"""
Earth4D Hash Collision Tracking System - Design Implementation
================================================================

This module provides the design and implementation framework for tracking
hash collisions in Earth4D's multi-resolution spatiotemporal encoding.

Key Requirements from Lance's specifications:
1. Track grid indices for all 4 grids (xyz, xyt, yzt, xzt) 
2. Store both original and normalized coordinates
3. Memory-efficient int16 storage (486 bytes per example)
4. Export capabilities for CSV/JSON analysis
5. Real-time statistical reporting
6. Optional tracking flag (off by default)
"""

import torch
import torch.nn as nn
import numpy as np
import json
import csv
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CollisionTrackingConfig:
    """Configuration for hash collision tracking."""
    
    # Tracking parameters
    enabled: bool = False                    # Collision tracking flag
    max_examples: int = 1_000_000           # Maximum examples to track
    track_coordinates: bool = True           # Store original/normalized coords
    track_collision_type: bool = True        # Track direct vs hash indexing
    
    # Memory management
    use_circular_buffer: bool = True         # Overwrite old data when full
    warning_threshold: float = 0.9           # Warn when memory usage hits 90%
    
    # Export settings  
    export_csv: bool = True                  # Enable CSV export
    export_json: bool = True                 # Enable JSON metadata export
    csv_batch_size: int = 100_000           # Batch size for CSV writing


class CollisionTracker:
    """
    Memory-efficient tracking system for Earth4D hash collision analysis.
    
    Tracks grid indices for all 4 grid spaces across all resolution levels,
    enabling comprehensive collision pattern analysis and visualization.
    """
    
    def __init__(self, 
                 config: CollisionTrackingConfig,
                 spatial_levels: int = 24,
                 temporal_levels: int = 19,
                 device: str = 'cuda'):
        """
        Initialize collision tracking infrastructure.
        
        Args:
            config: Collision tracking configuration
            spatial_levels: Number of spatial encoding levels
            temporal_levels: Number of temporal encoding levels  
            device: Device for tensor allocation
        """
        self.config = config
        self.spatial_levels = spatial_levels
        self.temporal_levels = temporal_levels
        self.device = device
        self.current_count = 0
        self.total_processed = 0
        
        if not config.enabled:
            return
            
        # Calculate memory requirements
        self._calculate_memory_requirements()
        
        # Allocate tracking tensors
        self._allocate_tracking_tensors()
        
        # Initialize collision statistics
        self._init_collision_stats()
        
        print(f"[CollisionTracker] Initialized for {config.max_examples:,} examples")
        print(f"[CollisionTracker] Estimated memory: {self.estimated_memory_mb:.1f} MB")
    
    def _calculate_memory_requirements(self):
        """Calculate memory requirements for tracking tensors."""
        bytes_per_example = (
            self.spatial_levels * 3 * 2 +      # XYZ grid indices (int16)
            self.temporal_levels * 3 * 2 * 3 +  # XYT, YZT, XZT grid indices (int16)
            4 * 4 +                             # Original coordinates (float32)
            4 * 4                               # Normalized coordinates (float32)
        )
        
        if self.config.track_collision_type:
            bytes_per_example += (
                self.spatial_levels * 1 +       # XYZ collision flags (bool)
                self.temporal_levels * 1 * 3    # XYT, YZT, XZT collision flags (bool)
            )
        
        total_bytes = bytes_per_example * self.config.max_examples
        self.estimated_memory_mb = total_bytes / (1024 * 1024)
        
        print(f"[CollisionTracker] {bytes_per_example} bytes per example")
        print(f"[CollisionTracker] {total_bytes:,} total bytes ({self.estimated_memory_mb:.1f} MB)")
    
    def _allocate_tracking_tensors(self):
        """Allocate GPU tensors for collision tracking."""
        max_examples = self.config.max_examples
        
        # Grid index tracking tensors
        self.tracking_data = {
            # Grid indices for each space (int16 to save memory)
            'xyz_indices': torch.zeros((max_examples, self.spatial_levels, 3), 
                                     dtype=torch.int16, device=self.device),
            'xyt_indices': torch.zeros((max_examples, self.temporal_levels, 3),
                                     dtype=torch.int16, device=self.device),
            'yzt_indices': torch.zeros((max_examples, self.temporal_levels, 3),
                                     dtype=torch.int16, device=self.device), 
            'xzt_indices': torch.zeros((max_examples, self.temporal_levels, 3),
                                     dtype=torch.int16, device=self.device),
        }
        
        if self.config.track_coordinates:
            self.tracking_data.update({
                # Original coordinates (lat, lon, elev, time)
                'original_coords': torch.zeros((max_examples, 4), 
                                             dtype=torch.float32, device=self.device),
                # Normalized coordinates (x_norm, y_norm, z_norm, t_norm)
                'normalized_coords': torch.zeros((max_examples, 4),
                                               dtype=torch.float32, device=self.device),
            })
        
        if self.config.track_collision_type:
            self.tracking_data.update({
                # Collision flags (True = hash used, False = direct indexing)
                'xyz_collisions': torch.zeros((max_examples, self.spatial_levels),
                                            dtype=torch.bool, device=self.device),
                'xyt_collisions': torch.zeros((max_examples, self.temporal_levels),
                                            dtype=torch.bool, device=self.device),
                'yzt_collisions': torch.zeros((max_examples, self.temporal_levels), 
                                            dtype=torch.bool, device=self.device),
                'xzt_collisions': torch.zeros((max_examples, self.temporal_levels),
                                            dtype=torch.bool, device=self.device),
            })
    
    def _init_collision_stats(self):
        """Initialize collision statistics tracking."""
        self.stats = {
            'total_examples': 0,
            'examples_per_grid': {'xyz': 0, 'xyt': 0, 'yzt': 0, 'xzt': 0},
            'collisions_per_level': {
                'xyz': [0] * self.spatial_levels,
                'xyt': [0] * self.temporal_levels,
                'yzt': [0] * self.temporal_levels,
                'xzt': [0] * self.temporal_levels,
            },
            'direct_indexing_per_level': {
                'xyz': [0] * self.spatial_levels,
                'xyt': [0] * self.temporal_levels,
                'yzt': [0] * self.temporal_levels,
                'xzt': [0] * self.temporal_levels,
            }
        }
    
    def record_collision_data(self, 
                            example_idx: int,
                            grid_type: str,
                            level: int,
                            grid_indices: torch.Tensor,
                            is_collision: bool = False):
        """
        Record collision data for a specific grid and level.
        
        Args:
            example_idx: Index of the training example
            grid_type: Grid type ('xyz', 'xyt', 'yzt', 'xzt')  
            level: Resolution level within the grid
            grid_indices: 3D grid coordinates [x, y, z] or [x, y, t], etc.
            is_collision: Whether hash collision occurred (vs direct indexing)
        """
        if not self.config.enabled or example_idx >= self.config.max_examples:
            return
        
        # Get the appropriate tracking tensor
        indices_key = f'{grid_type}_indices'
        if indices_key in self.tracking_data:
            # Store grid indices (convert to int16 for memory efficiency)
            self.tracking_data[indices_key][example_idx, level] = grid_indices.to(torch.int16)
        
        # Track collision type if enabled
        if self.config.track_collision_type:
            collision_key = f'{grid_type}_collisions'
            if collision_key in self.tracking_data:
                self.tracking_data[collision_key][example_idx, level] = is_collision
        
        # Update statistics
        self._update_stats(grid_type, level, is_collision)
    
    def record_coordinates(self,
                          example_idx: int, 
                          original_coords: torch.Tensor,
                          normalized_coords: torch.Tensor):
        """
        Record original and normalized coordinates for an example.
        
        Args:
            example_idx: Index of the training example
            original_coords: Original [lat, lon, elev, time] coordinates
            normalized_coords: Normalized [x, y, z, t] coordinates
        """
        if not self.config.enabled or not self.config.track_coordinates:
            return
            
        if example_idx >= self.config.max_examples:
            return
        
        self.tracking_data['original_coords'][example_idx] = original_coords
        self.tracking_data['normalized_coords'][example_idx] = normalized_coords
    
    def _update_stats(self, grid_type: str, level: int, is_collision: bool):
        """Update collision statistics."""
        self.stats['examples_per_grid'][grid_type] += 1
        
        if is_collision:
            self.stats['collisions_per_level'][grid_type][level] += 1
        else:
            self.stats['direct_indexing_per_level'][grid_type][level] += 1
    
    def get_collision_statistics(self) -> Dict:
        """
        Get comprehensive collision statistics.
        
        Returns:
            Dictionary containing collision analysis statistics
        """
        if not self.config.enabled:
            return {"error": "Collision tracking not enabled"}
        
        stats = {
            'summary': {
                'total_examples_tracked': min(self.current_count, self.config.max_examples),
                'total_examples_processed': self.total_processed,
                'memory_usage_mb': self.estimated_memory_mb,
                'tracking_enabled': True
            },
            'grid_statistics': {},
            'collision_rates': {}
        }
        
        # Calculate statistics for each grid
        for grid_type in ['xyz', 'xyt', 'yzt', 'xzt']:
            levels = self.spatial_levels if grid_type == 'xyz' else self.temporal_levels
            
            collisions = self.stats['collisions_per_level'][grid_type]
            direct = self.stats['direct_indexing_per_level'][grid_type]
            
            total_per_level = [c + d for c, d in zip(collisions, direct)]
            collision_rates = [c / max(t, 1) for c, t in zip(collisions, total_per_level)]
            
            stats['grid_statistics'][grid_type] = {
                'levels': levels,
                'total_accesses': sum(total_per_level),
                'total_collisions': sum(collisions),
                'total_direct': sum(direct),
                'overall_collision_rate': sum(collisions) / max(sum(total_per_level), 1)
            }
            
            stats['collision_rates'][grid_type] = {
                'per_level': collision_rates,
                'collisions_per_level': collisions,
                'direct_per_level': direct
            }
        
        return stats
    
    def export_to_csv(self, output_path: str) -> bool:
        """
        Export collision data to CSV format for analysis.
        
        Args:
            output_path: Path for CSV output file
            
        Returns:
            Success status
        """
        if not self.config.enabled or not self.config.export_csv:
            print("[CollisionTracker] CSV export not enabled")
            return False
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header
                header = ['example_id']
                
                # Coordinate columns
                if self.config.track_coordinates:
                    header.extend(['lat', 'lon', 'elev', 'time'])
                    header.extend(['x_norm', 'y_norm', 'z_norm', 't_norm'])
                
                # Grid index columns
                for grid in ['xyz', 'xyt', 'yzt', 'xzt']:
                    levels = self.spatial_levels if grid == 'xyz' else self.temporal_levels
                    for level in range(levels):
                        header.extend([f'{grid}_L{level}_x', f'{grid}_L{level}_y', f'{grid}_L{level}_z'])
                        if self.config.track_collision_type:
                            header.append(f'{grid}_L{level}_collision')
                
                writer.writerow(header)
                
                # Write data in batches
                num_examples = min(self.current_count, self.config.max_examples)
                batch_size = self.config.csv_batch_size
                
                for start_idx in range(0, num_examples, batch_size):
                    end_idx = min(start_idx + batch_size, num_examples)
                    
                    for example_idx in range(start_idx, end_idx):
                        row = [example_idx]
                        
                        # Add coordinates
                        if self.config.track_coordinates:
                            orig_coords = self.tracking_data['original_coords'][example_idx].cpu().numpy()
                            norm_coords = self.tracking_data['normalized_coords'][example_idx].cpu().numpy()
                            row.extend(orig_coords.tolist())
                            row.extend(norm_coords.tolist())
                        
                        # Add grid indices
                        for grid in ['xyz', 'xyt', 'yzt', 'xzt']:
                            levels = self.spatial_levels if grid == 'xyz' else self.temporal_levels
                            indices_tensor = self.tracking_data[f'{grid}_indices'][example_idx].cpu().numpy()
                            
                            for level in range(levels):
                                row.extend(indices_tensor[level].tolist())
                                
                                if self.config.track_collision_type:
                                    collision_flag = self.tracking_data[f'{grid}_collisions'][example_idx, level].item()
                                    row.append(collision_flag)
                        
                        writer.writerow(row)
                
            print(f"[CollisionTracker] CSV exported to {output_path}")
            print(f"[CollisionTracker] {num_examples:,} examples exported")
            return True
            
        except Exception as e:
            print(f"[CollisionTracker] CSV export failed: {e}")
            return False
    
    def export_metadata_json(self, output_path: str) -> bool:
        """
        Export metadata and configuration to JSON for analysis.
        
        Args:
            output_path: Path for JSON output file
            
        Returns:
            Success status
        """
        if not self.config.enabled or not self.config.export_json:
            print("[CollisionTracker] JSON export not enabled")
            return False
        
        metadata = {
            'earth4d_configuration': {
                'spatial_levels': self.spatial_levels,
                'temporal_levels': self.temporal_levels,
                'total_levels': self.spatial_levels + 3 * self.temporal_levels,
                'grid_spaces': ['xyz', 'xyt', 'yzt', 'xzt']
            },
            'tracking_configuration': {
                'max_examples': self.config.max_examples,
                'track_coordinates': self.config.track_coordinates,
                'track_collision_type': self.config.track_collision_type,
                'estimated_memory_mb': self.estimated_memory_mb
            },
            'collision_statistics': self.get_collision_statistics(),
            'data_format': {
                'csv_columns_explanation': {
                    'example_id': 'Sequential example index',
                    'lat': 'Original latitude (degrees)',
                    'lon': 'Original longitude (degrees)', 
                    'elev': 'Original elevation (meters)',
                    'time': 'Original time (normalized [0,1])',
                    'x_norm': 'Normalized ECEF X coordinate',
                    'y_norm': 'Normalized ECEF Y coordinate',
                    'z_norm': 'Normalized ECEF Z coordinate',
                    't_norm': 'Normalized time coordinate',
                    'grid_L#_x': 'X grid index at level #',
                    'grid_L#_y': 'Y grid index at level #',
                    'grid_L#_z': 'Z grid index at level #',
                    'grid_L#_collision': 'Collision flag (True=hash, False=direct)'
                }
            }
        }
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"[CollisionTracker] Metadata exported to {output_path}")
            return True
            
        except Exception as e:
            print(f"[CollisionTracker] JSON export failed: {e}")
            return False


# Integration hooks for modifying Earth4D
class Earth4DCollisionIntegration:
    """
    Integration utilities for adding collision tracking to Earth4D.
    
    This class provides the interface between the collision tracker
    and the existing Earth4D implementation.
    """
    
    @staticmethod
    def modify_hashgrid_forward(original_forward_fn):
        """
        Decorator to modify HashEncoder forward pass for collision tracking.
        
        Args:
            original_forward_fn: Original forward function to wrap
            
        Returns:
            Modified forward function with collision tracking
        """
        def forward_with_tracking(self, inputs, size=1, tracker=None, example_indices=None, grid_type=None):
            # Call original forward function
            outputs = original_forward_fn(self, inputs, size)
            
            # Add collision tracking if enabled
            if tracker and tracker.config.enabled and example_indices is not None and grid_type is not None:
                # TODO: Extract grid indices from CUDA kernel
                # This requires CUDA modification to return grid indices
                pass
            
            return outputs
        
        return forward_with_tracking
    
    @staticmethod
    def create_collision_aware_earth4d(Earth4DClass):
        """
        Create a collision-aware version of Earth4D class.
        
        Args:
            Earth4DClass: Original Earth4D class to extend
            
        Returns:
            Modified Earth4D class with collision tracking
        """
        class CollisionAwareEarth4D(Earth4DClass):
            def __init__(self, *args, collision_config=None, **kwargs):
                super().__init__(*args, **kwargs)
                
                # Initialize collision tracker
                if collision_config is None:
                    collision_config = CollisionTrackingConfig()
                
                self.collision_tracker = CollisionTracker(
                    config=collision_config,
                    spatial_levels=24,  # TODO: Extract from actual config
                    temporal_levels=19,  # TODO: Extract from actual config
                    device=str(next(self.parameters()).device)
                )
                
                self.example_count = 0
            
            def forward(self, coords, track_collisions=True):
                # Store coordinates for tracking
                if self.collision_tracker.config.enabled and track_collisions:
                    batch_size = coords.shape[0]
                    
                    for i in range(batch_size):
                        if self.example_count < self.collision_tracker.config.max_examples:
                            # Extract original coordinates (assuming lat/lon/elev/time format)
                            original_coord = coords[i].cpu()
                            
                            # TODO: Get normalized coordinates from preprocessing
                            normalized_coord = original_coord  # Placeholder
                            
                            self.collision_tracker.record_coordinates(
                                self.example_count, original_coord, normalized_coord
                            )
                        
                        self.example_count += 1
                    
                    self.collision_tracker.total_processed += batch_size
                
                # Call original forward with collision tracking hooks
                return super().forward(coords)
            
            def get_collision_stats(self):
                """Get collision statistics."""
                return self.collision_tracker.get_collision_statistics()
            
            def export_collision_data(self, output_dir: str):
                """Export collision data for analysis."""
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                csv_path = output_dir / "collision_data.csv"
                json_path = output_dir / "collision_metadata.json"
                
                csv_success = self.collision_tracker.export_to_csv(csv_path)
                json_success = self.collision_tracker.export_metadata_json(json_path)
                
                return csv_success and json_success
        
        return CollisionAwareEarth4D


# Example usage
if __name__ == "__main__":
    # Configuration for collision tracking
    config = CollisionTrackingConfig(
        enabled=True,
        max_examples=1_000_000,
        track_coordinates=True,
        track_collision_type=True,
        export_csv=True,
        export_json=True
    )
    
    # Initialize tracker
    tracker = CollisionTracker(config)
    
    # Simulate collision data recording
    print("Collision tracking system designed and ready for implementation!")
    print(f"Memory requirement: {tracker.estimated_memory_mb:.1f} MB for {config.max_examples:,} examples")