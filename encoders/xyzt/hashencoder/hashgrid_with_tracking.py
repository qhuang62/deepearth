import enum
from math import ceil
from cachetools import cached
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd 

# Try to import the collision tracking backend
try:
    from .backend_tracking import _backend
    HAS_TRACKING = hasattr(_backend, 'hash_encode_forward_with_tracking') if _backend else False
except ImportError:
    _backend = None
    HAS_TRACKING = False
    print("[WARNING] Collision tracking CUDA backend not available. Install collision tracking extension.")


class _hash_encode_with_tracking(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx, inputs, embeddings, offsets, per_level_scale, base_resolution, calc_grad_inputs=False, track_collisions=False, grid_indices=None, collision_flags=None, max_tracking_examples=0, current_example_count=None):
        # inputs: [B, D], float in [0, 1]
        # embeddings: [sO, C], float
        # offsets: [L + 1], int
        # RETURN: [B, F], float

        inputs = inputs.contiguous()
        embeddings = embeddings.contiguous()
        offsets = offsets.contiguous()
        per_level_scale = per_level_scale.contiguous()
        base_resolution = base_resolution.contiguous()
        
        B, D = inputs.shape # batch size, coord dim
        L = offsets.shape[0] - 1 # level
        C = embeddings.shape[1] # embedding dim for each level
        per_level_scale = torch.log2(per_level_scale)

        # L first, optimize cache for cuda kernel, but needs an extra permute later
        outputs = torch.empty(L, B, C, device=inputs.device, dtype=inputs.dtype)

        if calc_grad_inputs:
            dy_dx = torch.empty(B, L * D * C, device=inputs.device, dtype=inputs.dtype)
        else:
            dy_dx = torch.empty(1, device=inputs.device, dtype=inputs.dtype)

        # Prepare collision tracking tensors
        if track_collisions and grid_indices is not None and collision_flags is not None and current_example_count is not None:
            grid_indices = grid_indices.contiguous()
            collision_flags = collision_flags.contiguous()
            current_example_count = current_example_count.contiguous()
            
            if HAS_TRACKING:
                _backend.hash_encode_forward_with_tracking(
                    inputs, embeddings, offsets, outputs, B, D, C, L, 
                    per_level_scale, base_resolution, calc_grad_inputs, dy_dx,
                    track_collisions, grid_indices, collision_flags, 
                    max_tracking_examples, current_example_count
                )
            else:
                # Fallback to regular encoding if tracking not available
                print("[WARNING] Collision tracking not available, falling back to regular encoding")
                _backend.hash_encode_forward(inputs, embeddings, offsets, outputs, B, D, C, L, per_level_scale, base_resolution, calc_grad_inputs, dy_dx)
        else:
            # Regular forward pass without tracking
            _backend.hash_encode_forward(inputs, embeddings, offsets, outputs, B, D, C, L, per_level_scale, base_resolution, calc_grad_inputs, dy_dx)

        # permute back to [B, L * C]
        outputs = outputs.permute(1, 0, 2).reshape(B, L * C)

        ctx.save_for_backward(inputs, embeddings, offsets, per_level_scale, base_resolution, dy_dx)
        ctx.dims = [B, D, C, L]
        ctx.calc_grad_inputs = calc_grad_inputs

        return outputs
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        # Use the same backward pass as the original implementation
        from .hashgrid import _hash_encode_second_backward
        
        inputs, embeddings, offsets, per_level_scale, base_resolution, dy_dx = ctx.saved_tensors
        B, D, C, L = ctx.dims
        calc_grad_inputs = ctx.calc_grad_inputs

        # grad: [B, L * C] --> [L, B, C]
        grad = grad.view(B, L, C).permute(1, 0, 2).contiguous()

        grad_inputs, grad_embeddings = _hash_encode_second_backward.apply(grad, inputs, embeddings, offsets, B, D, C, L, per_level_scale, base_resolution, calc_grad_inputs, dy_dx)

        if calc_grad_inputs:
            return grad_inputs, grad_embeddings, None, None, None, None, None, None, None, None, None
        else:
            return None, grad_embeddings, None, None, None, None, None, None, None, None, None


hash_encode_with_tracking = _hash_encode_with_tracking.apply


class HashEncoderWithTracking(nn.Module):
    """
    Enhanced HashEncoder with collision tracking capabilities.
    
    This class extends the original HashEncoder to support comprehensive
    collision tracking for research and analysis purposes.
    """
    
    def __init__(self, input_dim=3, num_levels=16, level_dim=2, per_level_scale=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=None, enable_collision_tracking=False, max_tracking_examples=1000000):
        super().__init__()

        if type(base_resolution) is int:
            base_resolution = np.array([base_resolution for _ in range(input_dim)], dtype=np.float64)
        else:
            assert len(base_resolution) == input_dim
            base_resolution = np.array(base_resolution, dtype=np.float64)

        # the finest resolution desired at the last level, if provided, overridee per_level_scale
        if desired_resolution is not None:
            if type(desired_resolution) is int:
                desired_resolution = np.array([desired_resolution for _ in range(input_dim)], dtype=np.float64)
            else:
                assert len(desired_resolution) == input_dim
            per_level_scale = np.exp2(np.log2(desired_resolution / base_resolution) / (num_levels - 1))
        else:
            # Handle both scalar and array per_level_scale
            if type(per_level_scale) is int or type(per_level_scale) is float:
                per_level_scale = np.array([per_level_scale for _ in range(input_dim)], dtype=np.float64)
            else:
                assert len(per_level_scale) == input_dim
                per_level_scale = np.array(per_level_scale, dtype=np.float64)

        self.input_dim = input_dim # coord dims, 2 or 3
        self.num_levels = num_levels # num levels, each level multiply resolution by 2
        self.level_dim = level_dim # encode channels per level
        self.log2_hashmap_size = log2_hashmap_size
        self.enable_collision_tracking = enable_collision_tracking
        self.max_tracking_examples = max_tracking_examples
        
        self.output_dim = num_levels * level_dim

        if level_dim % 2 != 0:
            print('[WARN] detected HashGrid level_dim % 2 != 0, which will cause very slow backward is also enabled fp16! (maybe fix later)')

        # allocate parameters
        offsets = []
        offset = 0
        self.max_params = 2 ** log2_hashmap_size
        for i in range(num_levels):
            resolution = np.ceil(base_resolution * per_level_scale ** i)
            params_in_level = min(self.max_params, np.prod(resolution)) # limit max number
            #params_in_level = np.ceil(params_in_level / 8) * 8 # make divisible
            offsets.append(offset)
            offset += int(params_in_level)
        offsets.append(offset)
        offsets = torch.from_numpy(np.array(offsets, dtype=np.int32))
        self.register_buffer('offsets', offsets)
        self.register_buffer('per_level_scale', torch.tensor(per_level_scale, dtype=torch.float32))
        self.register_buffer('base_resolution', torch.tensor(base_resolution, dtype=torch.float32))
        
        self.n_params = offsets[-1] * level_dim

        # parameters
        self.embeddings = nn.Parameter(torch.empty(offset, level_dim))

        # Collision tracking tensors (allocated on-demand)
        self.collision_tracking_data = None
        self.current_example_count = None
        
        self.reset_parameters()

    def reset_parameters(self):
        # With large hash tables and high collision ratios, we need stronger initialization
        # to ensure gradients flow properly
        std = 1e-1  # Increased to 0.1 for better gradient flow with large hash tables
        self.embeddings.data.uniform_(-std, std)

    def _initialize_collision_tracking(self, device):
        """Initialize collision tracking tensors."""
        if not self.enable_collision_tracking or self.collision_tracking_data is not None:
            return
        
        print(f"[HashEncoderWithTracking] Initializing collision tracking for {self.max_tracking_examples:,} examples")
        
        # Allocate tracking tensors
        self.collision_tracking_data = {
            'grid_indices': torch.zeros((self.max_tracking_examples, self.num_levels, self.input_dim), 
                                      dtype=torch.int16, device=device),
            'collision_flags': torch.zeros((self.max_tracking_examples, self.num_levels), 
                                         dtype=torch.bool, device=device),
        }
        
        # Global counter for tracking
        self.current_example_count = torch.zeros(1, dtype=torch.uint32, device=device)
        
        # Calculate memory usage
        memory_mb = (
            self.collision_tracking_data['grid_indices'].numel() * 2 +  # int16
            self.collision_tracking_data['collision_flags'].numel() * 1 + # bool
            4  # uint32 counter
        ) / (1024 * 1024)
        
        print(f"[HashEncoderWithTracking] Collision tracking memory: {memory_mb:.1f} MB")

    def get_collision_stats(self):
        """Get collision statistics from tracked data."""
        if not self.enable_collision_tracking or self.collision_tracking_data is None:
            return {"error": "Collision tracking not enabled or not initialized"}
        
        current_count = min(self.current_example_count.item(), self.max_tracking_examples)
        if current_count == 0:
            return {"error": "No examples tracked yet"}
        
        # Get collision flags for tracked examples
        collision_flags = self.collision_tracking_data['collision_flags'][:current_count]  # [examples, levels]
        
        # Calculate statistics per level
        total_per_level = current_count  # Each example contributes to each level
        collisions_per_level = collision_flags.sum(dim=0).cpu().numpy()  # Sum across examples
        collision_rates = collisions_per_level / max(total_per_level, 1)
        
        stats = {
            'summary': {
                'total_examples_tracked': current_count,
                'max_tracking_capacity': self.max_tracking_examples,
                'num_levels': self.num_levels,
                'input_dimensions': self.input_dim
            },
            'collision_analysis': {
                'collisions_per_level': collisions_per_level.tolist(),
                'collision_rates_per_level': collision_rates.tolist(),
                'total_collisions': int(collisions_per_level.sum()),
                'overall_collision_rate': float(collisions_per_level.sum() / (total_per_level * self.num_levels)),
                'levels_with_collisions': int((collision_rates > 0).sum()),
                'levels_without_collisions': int((collision_rates == 0).sum())
            }
        }
        
        return stats

    def export_collision_data(self, output_path):
        """Export collision tracking data to file."""
        if not self.enable_collision_tracking or self.collision_tracking_data is None:
            print("[HashEncoderWithTracking] No collision data to export")
            return False
        
        current_count = min(self.current_example_count.item(), self.max_tracking_examples)
        if current_count == 0:
            print("[HashEncoderWithTracking] No examples tracked")
            return False
        
        # Prepare data for export
        export_data = {
            'metadata': {
                'num_levels': self.num_levels,
                'input_dimensions': self.input_dim,
                'hashmap_size': 2 ** self.log2_hashmap_size,
                'examples_tracked': current_count,
                'max_capacity': self.max_tracking_examples
            },
            'grid_indices': self.collision_tracking_data['grid_indices'][:current_count].cpu().numpy(),
            'collision_flags': self.collision_tracking_data['collision_flags'][:current_count].cpu().numpy(),
            'statistics': self.get_collision_stats()
        }
        
        # Save to file
        torch.save(export_data, output_path)
        print(f"[HashEncoderWithTracking] Collision data exported to {output_path}")
        print(f"[HashEncoderWithTracking] {current_count:,} examples exported")
        return True

    def __repr__(self):
        tracking_info = f", collision_tracking={'enabled' if self.enable_collision_tracking else 'disabled'}"
        if self.enable_collision_tracking:
            tracking_info += f" (max_examples={self.max_tracking_examples:,})"
        
        return f"HashEncoderWithTracking: input_dim={self.input_dim} num_levels={self.num_levels} level_dim={self.level_dim} base_resolution={self.base_resolution} per_level_scale={self.per_level_scale} params={tuple(self.embeddings.shape)}{tracking_info}"

    def forward(self, inputs, size=1, track_collisions=None):
        # inputs: [..., input_dim], normalized real world positions in [-size, size]
        # return: [..., num_levels * level_dim]

        inputs = (inputs + size) / (2 * size) # map to [0, 1]
        
        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.input_dim)

        # Determine if we should track collisions for this forward pass
        should_track = track_collisions if track_collisions is not None else self.enable_collision_tracking
        
        if should_track and HAS_TRACKING:
            # Initialize tracking tensors if needed
            self._initialize_collision_tracking(inputs.device)
            
            # Forward pass with collision tracking
            outputs = hash_encode_with_tracking(
                inputs, self.embeddings, self.offsets, self.per_level_scale, 
                self.base_resolution, inputs.requires_grad, 
                True,  # track_collisions
                self.collision_tracking_data['grid_indices'],
                self.collision_tracking_data['collision_flags'],
                self.max_tracking_examples,
                self.current_example_count
            )
        else:
            # Regular forward pass without tracking
            from .hashgrid import hash_encode
            outputs = hash_encode(inputs, self.embeddings, self.offsets, self.per_level_scale, self.base_resolution, inputs.requires_grad)
        
        outputs = outputs.view(prefix_shape + [self.output_dim])
        return outputs