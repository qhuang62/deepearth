"""
Earth4D with Collision Tracking: Enhanced Grid4D Encoder for Hash Collision Analysis
====================================================================================

This module extends the original Earth4D encoder with comprehensive collision tracking
capabilities, enabling detailed analysis of hash collision patterns across all 4 grid
spaces (xyz, xyt, yzt, xzt) at all resolution levels.

Key Features:
- Tracks grid indices for all 4 grid spaces
- Records collision vs direct indexing patterns  
- Memory-efficient int16 storage
- Real-time statistical analysis
- CSV/JSON export for scientific analysis
- Configurable tracking limits

Author: Earth4D Collision Research Team
License: MIT
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Union, Tuple, Dict
import warnings
import json
import csv
from pathlib import Path

# Import the collision tracking hash encoder
try:
    from .hashencoder.hashgrid_with_tracking import HashEncoderWithTracking
    TRACKING_AVAILABLE = True
except ImportError:
    try:
        # Fallback to regular hash encoder
        from .hashencoder.hashgrid import HashEncoder
        HashEncoderWithTracking = HashEncoder
        TRACKING_AVAILABLE = False
        warnings.warn("Collision tracking not available, using regular HashEncoder")
    except ImportError:
        warnings.warn("HashEncoder not found. Please install the hash encoding library.")
        HashEncoderWithTracking = None
        TRACKING_AVAILABLE = False


class CollisionTrackingConfig:
    """Configuration for Earth4D collision tracking."""
    
    def __init__(self,
                 enabled: bool = False,
                 max_examples: int = 1_000_000,
                 track_coordinates: bool = True,
                 export_csv: bool = True,
                 export_json: bool = True,
                 output_dir: str = "./collision_analysis"):
        self.enabled = enabled
        self.max_examples = max_examples
        self.track_coordinates = track_coordinates
        self.export_csv = export_csv
        self.export_json = export_json
        self.output_dir = output_dir


class Earth4DCollisionTracker:
    """
    Comprehensive collision tracking system for Earth4D's 4 grid spaces.
    
    Tracks collision patterns across:
    - XYZ grid (spatial): 24 levels, 4M hash table
    - XYT grid (temporal): 19 levels, 256K hash table  
    - YZT grid (temporal): 19 levels, 256K hash table
    - XZT grid (temporal): 19 levels, 256K hash table
    """
    
    def __init__(self, config: CollisionTrackingConfig, device: str = 'cuda'):
        self.config = config
        self.device = device
        self.example_count = 0
        
        if not config.enabled:
            return
        
        # Initialize coordinate tracking
        if config.track_coordinates:
            self.coordinates = {
                'original': torch.zeros((config.max_examples, 4), dtype=torch.float32, device=device),
                'normalized': torch.zeros((config.max_examples, 4), dtype=torch.float32, device=device)
            }
        else:
            self.coordinates = None
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'grid_stats': {
                'xyz': {'total_accesses': 0, 'collisions': 0},
                'xyt': {'total_accesses': 0, 'collisions': 0},
                'yzt': {'total_accesses': 0, 'collisions': 0},
                'xzt': {'total_accesses': 0, 'collisions': 0}
            }
        }
        
        print(f"[Earth4DCollisionTracker] Initialized for {config.max_examples:,} examples")
    
    def record_coordinates(self, batch_coords_original, batch_coords_normalized):
        """Record coordinate information for a batch."""
        if not self.config.enabled or not self.config.track_coordinates:
            return
        
        batch_size = batch_coords_original.shape[0]
        end_idx = min(self.example_count + batch_size, self.config.max_examples)
        actual_batch_size = end_idx - self.example_count
        
        if actual_batch_size > 0:
            self.coordinates['original'][self.example_count:end_idx] = batch_coords_original[:actual_batch_size]
            self.coordinates['normalized'][self.example_count:end_idx] = batch_coords_normalized[:actual_batch_size]
        
        self.example_count = end_idx
        self.stats['total_processed'] += batch_size
    
    def get_collision_statistics(self, encoders: Dict) -> Dict:
        """Get comprehensive collision statistics from all encoders."""
        if not self.config.enabled:
            return {"error": "Collision tracking not enabled"}
        
        stats = {
            'summary': {
                'total_examples_tracked': self.example_count,
                'total_examples_processed': self.stats['total_processed'],
                'max_tracking_capacity': self.config.max_examples,
                'tracking_enabled': True
            },
            'grid_statistics': {},
            'memory_usage': {}
        }
        
        # Get statistics from each encoder
        for grid_name, encoder in encoders.items():
            if hasattr(encoder, 'get_collision_stats'):
                grid_stats = encoder.get_collision_stats()
                stats['grid_statistics'][grid_name] = grid_stats
                
                # Calculate memory usage
                if hasattr(encoder, 'collision_tracking_data') and encoder.collision_tracking_data is not None:
                    memory_mb = sum(
                        tensor.numel() * tensor.element_size() 
                        for tensor in encoder.collision_tracking_data.values()
                    ) / (1024 * 1024)
                    stats['memory_usage'][grid_name] = f"{memory_mb:.1f} MB"
        
        # Add coordinate tracking memory if enabled
        if self.coordinates is not None:
            coord_memory_mb = sum(
                tensor.numel() * tensor.element_size() 
                for tensor in self.coordinates.values()
            ) / (1024 * 1024)
            stats['memory_usage']['coordinates'] = f"{coord_memory_mb:.1f} MB"
        
        return stats
    
    def export_comprehensive_analysis(self, encoders: Dict, output_dir: str = None) -> bool:
        """Export comprehensive collision analysis to CSV and JSON."""
        if not self.config.enabled:
            print("[Earth4DCollisionTracker] Collision tracking not enabled")
            return False
        
        output_dir = Path(output_dir or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        success = True
        
        # Export CSV data
        if self.config.export_csv:
            csv_path = output_dir / "earth4d_collision_data.csv"
            success &= self._export_csv(encoders, csv_path)
        
        # Export JSON metadata
        if self.config.export_json:
            json_path = output_dir / "earth4d_collision_metadata.json"
            success &= self._export_json(encoders, json_path)
        
        return success
    
    def _export_csv(self, encoders: Dict, csv_path: Path) -> bool:
        """Export collision data to CSV format."""
        try:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Build header
                header = ['example_id']
                
                # Coordinate columns
                if self.coordinates is not None:
                    header.extend(['lat', 'lon', 'elev', 'time'])
                    header.extend(['x_norm', 'y_norm', 'z_norm', 't_norm'])
                
                # Grid data columns
                for grid_name in ['xyz', 'xyt', 'yzt', 'xzt']:
                    if grid_name in encoders:
                        encoder = encoders[grid_name]
                        if hasattr(encoder, 'num_levels') and hasattr(encoder, 'input_dim'):
                            levels = encoder.num_levels
                            dims = encoder.input_dim
                            for level in range(levels):
                                for dim in range(dims):
                                    header.append(f'{grid_name}_L{level}_dim{dim}')
                                header.append(f'{grid_name}_L{level}_collision')
                
                writer.writerow(header)
                
                # Write data rows
                num_examples = min(self.example_count, self.config.max_examples)
                
                for example_idx in range(num_examples):
                    row = [example_idx]
                    
                    # Add coordinates
                    if self.coordinates is not None:
                        orig_coords = self.coordinates['original'][example_idx].cpu().numpy()
                        norm_coords = self.coordinates['normalized'][example_idx].cpu().numpy()
                        row.extend(orig_coords.tolist())
                        row.extend(norm_coords.tolist())
                    
                    # Add grid data
                    for grid_name in ['xyz', 'xyt', 'yzt', 'xzt']:
                        if grid_name in encoders:
                            encoder = encoders[grid_name]
                            if (hasattr(encoder, 'collision_tracking_data') and 
                                encoder.collision_tracking_data is not None and
                                example_idx < encoder.current_example_count.item()):
                                
                                # Grid indices
                                grid_indices = encoder.collision_tracking_data['grid_indices'][example_idx].cpu().numpy()
                                collision_flags = encoder.collision_tracking_data['collision_flags'][example_idx].cpu().numpy()
                                
                                for level in range(encoder.num_levels):
                                    for dim in range(encoder.input_dim):
                                        row.append(int(grid_indices[level, dim]))
                                    row.append(bool(collision_flags[level]))
                    
                    writer.writerow(row)
                
                print(f"[Earth4DCollisionTracker] CSV exported: {csv_path}")
                print(f"[Earth4DCollisionTracker] {num_examples:,} examples exported")
                return True
                
        except Exception as e:
            print(f"[Earth4DCollisionTracker] CSV export failed: {e}")
            return False
    
    def _export_json(self, encoders: Dict, json_path: Path) -> bool:
        """Export metadata and statistics to JSON."""
        try:
            metadata = {
                'earth4d_configuration': {
                    'grid_spaces': list(encoders.keys()),
                    'total_grids': len(encoders),
                    'coordinate_system': 'ECEF_normalized'
                },
                'collision_tracking_config': {
                    'enabled': self.config.enabled,
                    'max_examples': self.config.max_examples,
                    'track_coordinates': self.config.track_coordinates,
                    'examples_tracked': self.example_count,
                    'examples_processed': self.stats['total_processed']
                },
                'grid_configurations': {},
                'collision_statistics': self.get_collision_statistics(encoders),
                'analysis_timestamp': str(torch.cuda.Event().record()),
                'data_format_documentation': {
                    'csv_columns': {
                        'example_id': 'Sequential example index',
                        'lat': 'Original latitude (degrees WGS84)',
                        'lon': 'Original longitude (degrees WGS84)',
                        'elev': 'Original elevation (meters above sea level)',
                        'time': 'Original normalized time [0,1]',
                        'x_norm': 'Normalized ECEF X coordinate [-1,1]',
                        'y_norm': 'Normalized ECEF Y coordinate [-1,1]',
                        'z_norm': 'Normalized ECEF Z coordinate [-1,1]',
                        't_norm': 'Normalized time coordinate [0,1]',
                        'grid_L#_dim#': 'Grid index at level # for dimension #',
                        'grid_L#_collision': 'Collision flag (True=hash, False=direct)'
                    },
                    'grid_spaces_explanation': {
                        'xyz': 'Pure spatial encoding (ECEF coordinates)',
                        'xyt': 'Equatorial plane + time (X-Y plane + T)',
                        'yzt': 'Meridional plane + time (Y-Z plane + T)', 
                        'xzt': 'Prime meridian plane + time (X-Z plane + T)'
                    }
                }
            }
            
            # Add encoder-specific configurations
            for grid_name, encoder in encoders.items():
                if hasattr(encoder, 'num_levels'):
                    metadata['grid_configurations'][grid_name] = {
                        'num_levels': encoder.num_levels,
                        'input_dimensions': encoder.input_dim,
                        'features_per_level': encoder.level_dim,
                        'log2_hashmap_size': encoder.log2_hashmap_size,
                        'hashmap_size': 2 ** encoder.log2_hashmap_size,
                        'base_resolution': encoder.base_resolution.tolist() if hasattr(encoder.base_resolution, 'tolist') else encoder.base_resolution,
                        'per_level_scale': encoder.per_level_scale.tolist() if hasattr(encoder.per_level_scale, 'tolist') else encoder.per_level_scale
                    }
            
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            print(f"[Earth4DCollisionTracker] JSON metadata exported: {json_path}")
            return True
            
        except Exception as e:
            print(f"[Earth4DCollisionTracker] JSON export failed: {e}")
            return False


class Earth4DWithCollisionTracking(nn.Module):
    """
    Earth4D encoder with comprehensive collision tracking across all 4 grid spaces.
    
    Extends the original Earth4D implementation to provide detailed collision analysis
    for research into hash encoding patterns at planetary scale.
    """
    
    def __init__(self,
                 # Core encoder configuration (same as original Earth4D)
                 spatial_levels: int = 24,
                 temporal_levels: int = 19,
                 features_per_level: int = 2,
                 spatial_log2_hashmap_size: int = 22,
                 temporal_log2_hashmap_size: int = 18,
                 base_spatial_resolution: float = 16.0,
                 base_temporal_resolution: float = 8.0,
                 growth_factor: float = 2.0,
                 # Collision tracking configuration
                 collision_config: CollisionTrackingConfig = None,
                 verbose: bool = True):
        """
        Initialize Earth4D with collision tracking.
        
        Args:
            collision_config: Configuration for collision tracking (None = disabled)
            All other args: Same as original Earth4D
        """
        super().__init__()
        
        self.spatial_levels = spatial_levels
        self.temporal_levels = temporal_levels
        self.features_per_level = features_per_level
        self.verbose = verbose
        
        # Initialize collision tracking
        if collision_config is None:
            collision_config = CollisionTrackingConfig(enabled=False)
        
        self.collision_config = collision_config
        self.collision_tracker = Earth4DCollisionTracker(collision_config)
        
        # WGS84 ellipsoid parameters for coordinate conversion
        self.WGS84_A = 6378137.0  # Semi-major axis in meters
        self.WGS84_F = 1.0 / 298.257223563  # Flattening
        self.WGS84_E2 = 2 * self.WGS84_F - self.WGS84_F**2  # First eccentricity squared
        
        # Calculate max resolutions
        spatial_max_res = int(base_spatial_resolution * (growth_factor ** (spatial_levels - 1)))
        temporal_base_res = [int(base_temporal_resolution)] * 3
        temporal_max_res = [int(base_temporal_resolution * (growth_factor ** (temporal_levels - 1)))] * 3
        
        # Initialize the 4 encoders with collision tracking
        enable_tracking = collision_config.enabled and TRACKING_AVAILABLE
        max_examples = collision_config.max_examples if enable_tracking else 0
        
        # XYZ encoder (spatial)
        self.xyz_encoder = HashEncoderWithTracking(
            input_dim=3,
            num_levels=spatial_levels,
            level_dim=features_per_level,
            per_level_scale=growth_factor,
            base_resolution=int(base_spatial_resolution),
            log2_hashmap_size=spatial_log2_hashmap_size,
            desired_resolution=spatial_max_res,
            enable_collision_tracking=enable_tracking,
            max_tracking_examples=max_examples
        )
        
        # XYT encoder (temporal projection 1)
        self.xyt_encoder = HashEncoderWithTracking(
            input_dim=3,
            num_levels=temporal_levels,
            level_dim=features_per_level,
            per_level_scale=growth_factor,
            base_resolution=temporal_base_res,
            log2_hashmap_size=temporal_log2_hashmap_size,
            desired_resolution=temporal_max_res,
            enable_collision_tracking=enable_tracking,
            max_tracking_examples=max_examples
        )
        
        # YZT encoder (temporal projection 2)
        self.yzt_encoder = HashEncoderWithTracking(
            input_dim=3,
            num_levels=temporal_levels,
            level_dim=features_per_level,
            per_level_scale=growth_factor,
            base_resolution=temporal_base_res,
            log2_hashmap_size=temporal_log2_hashmap_size,
            desired_resolution=temporal_max_res,
            enable_collision_tracking=enable_tracking,
            max_tracking_examples=max_examples
        )
        
        # XZT encoder (temporal projection 3)
        self.xzt_encoder = HashEncoderWithTracking(
            input_dim=3,
            num_levels=temporal_levels,
            level_dim=features_per_level,
            per_level_scale=growth_factor,
            base_resolution=temporal_base_res,
            log2_hashmap_size=temporal_log2_hashmap_size,
            desired_resolution=temporal_max_res,
            enable_collision_tracking=enable_tracking,
            max_tracking_examples=max_examples
        )
        
        # Calculate output dimensions
        self.spatial_dim = spatial_levels * features_per_level
        self.temporal_dim = temporal_levels * features_per_level * 3  # 3 projections
        self.output_dim = self.spatial_dim + self.temporal_dim
        
        if verbose:
            self._print_configuration_info()
    
    def _print_configuration_info(self):
        """Print comprehensive configuration information."""
        print("\n" + "="*80)
        print("EARTH4D WITH COLLISION TRACKING - CONFIGURATION")
        print("="*80)
        
        print(f"\nCORE CONFIGURATION:")
        print(f"  Spatial levels: {self.spatial_levels}")
        print(f"  Temporal levels: {self.temporal_levels}")
        print(f"  Output dimension: {self.output_dim}D")
        print(f"    - Spatial: {self.spatial_dim}D")
        print(f"    - Temporal: {self.temporal_dim}D")
        
        print(f"\nCOLLISION TRACKING:")
        print(f"  Enabled: {self.collision_config.enabled}")
        if self.collision_config.enabled:
            print(f"  Max examples: {self.collision_config.max_examples:,}")
            print(f"  Track coordinates: {self.collision_config.track_coordinates}")
            print(f"  Backend available: {TRACKING_AVAILABLE}")
            
            # Calculate memory requirements
            memory_per_example = (
                self.spatial_levels * 3 * 2 +      # xyz grid indices
                self.temporal_levels * 3 * 2 * 3 + # 3 temporal grids
                (4 * 4 * 2 if self.collision_config.track_coordinates else 0)  # coordinates
            )
            total_memory_mb = memory_per_example * self.collision_config.max_examples / (1024 * 1024)
            print(f"  Estimated memory: {total_memory_mb:.1f} MB")
        
        print("="*80 + "\n")
    
    def get_encoders_dict(self) -> Dict:
        """Get dictionary of all encoders for analysis."""
        return {
            'xyz': self.xyz_encoder,
            'xyt': self.xyt_encoder,
            'yzt': self.yzt_encoder,
            'xzt': self.xzt_encoder
        }
    
    def forward(self, coords: torch.Tensor, track_collisions: bool = None) -> torch.Tensor:
        """
        Forward pass with collision tracking.
        
        Args:
            coords: Input coordinates [lat, lon, elev, time]
            track_collisions: Override collision tracking for this batch
            
        Returns:
            Concatenated features from all 4 grid spaces
        """
        batch_size = coords.shape[0]
        
        # Store original coordinates for tracking
        if self.collision_config.enabled and self.collision_config.track_coordinates:
            original_coords = coords.clone()
        
        # Convert lat/lon/elev to ECEF and normalize
        lat = coords[..., 0]
        lon = coords[..., 1]
        elev = coords[..., 2]
        time = coords[..., 3]
        
        # Convert to ECEF using WGS84
        lat_rad = torch.deg2rad(lat)
        lon_rad = torch.deg2rad(lon)
        
        sin_lat = torch.sin(lat_rad)
        cos_lat = torch.cos(lat_rad)
        sin_lon = torch.sin(lon_rad)
        cos_lon = torch.cos(lon_rad)
        
        # WGS84 ellipsoid conversion
        N = self.WGS84_A / torch.sqrt(1 - self.WGS84_E2 * sin_lat * sin_lat)
        
        x = (N + elev) * cos_lat * cos_lon
        y = (N + elev) * cos_lat * sin_lon
        z = (N * (1 - self.WGS84_E2) + elev) * sin_lat
        
        # Normalize ECEF to [-1, 1]
        norm_factor = 6400000.0  # 6400km in meters
        x_norm = x / norm_factor
        y_norm = y / norm_factor
        z_norm = z / norm_factor
        
        # Stack normalized coordinates
        norm_coords = torch.stack([x_norm, y_norm, z_norm, time], dim=-1)
        
        # Record coordinates for collision tracking
        if self.collision_config.enabled and self.collision_config.track_coordinates:
            self.collision_tracker.record_coordinates(original_coords, norm_coords)
        
        # Extract coordinate components
        xyz = norm_coords[..., :3]
        
        # Scale time dimension for temporal projections
        xyz_scaled = norm_coords[..., :3]
        t_scaled = (norm_coords[..., 3:] * 2 - 1) * 0.9  # Scale to ~[-0.9, 0.9]
        xyzt_scaled = torch.cat([xyz_scaled, t_scaled], dim=-1)
        
        # Create 3D projections
        xyt = torch.cat([xyzt_scaled[..., :2], xyzt_scaled[..., 3:]], dim=-1)
        yzt = xyzt_scaled[..., 1:]
        xzt = torch.cat([xyzt_scaled[..., :1], xyzt_scaled[..., 2:]], dim=-1)
        
        # Encode with collision tracking
        should_track = track_collisions if track_collisions is not None else self.collision_config.enabled
        
        spatial_features = self.xyz_encoder(xyz, size=1.0, track_collisions=should_track)
        xyt_features = self.xyt_encoder(xyt, size=1.0, track_collisions=should_track)
        yzt_features = self.yzt_encoder(yzt, size=1.0, track_collisions=should_track)
        xzt_features = self.xzt_encoder(xzt, size=1.0, track_collisions=should_track)
        
        # Concatenate all features
        temporal_features = torch.cat([xyt_features, yzt_features, xzt_features], dim=-1)
        return torch.cat([spatial_features, temporal_features], dim=-1)
    
    def get_collision_statistics(self) -> Dict:
        """Get comprehensive collision statistics from all encoders."""
        return self.collision_tracker.get_collision_statistics(self.get_encoders_dict())
    
    def export_collision_analysis(self, output_dir: str = None) -> bool:
        """Export comprehensive collision analysis."""
        return self.collision_tracker.export_comprehensive_analysis(
            self.get_encoders_dict(), output_dir
        )
    
    def print_collision_summary(self):
        """Print a summary of collision statistics."""
        if not self.collision_config.enabled:
            print("[Earth4DWithCollisionTracking] Collision tracking not enabled")
            return
        
        stats = self.get_collision_statistics()
        
        print("\n" + "="*60)
        print("EARTH4D COLLISION ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Examples tracked: {stats['summary']['total_examples_tracked']:,}")
        print(f"Examples processed: {stats['summary']['total_examples_processed']:,}")
        
        print(f"\nCOLLISION RATES BY GRID:")
        for grid_name, grid_stats in stats['grid_statistics'].items():
            if 'error' not in grid_stats and 'collision_analysis' in grid_stats:
                collision_rate = grid_stats['collision_analysis']['overall_collision_rate']
                total_collisions = grid_stats['collision_analysis']['total_collisions']
                print(f"  {grid_name.upper()}: {collision_rate:.1%} ({total_collisions:,} collisions)")
        
        print("="*60 + "\n")


# Example usage and testing
if __name__ == "__main__":
    print("Earth4D with Collision Tracking - Ready for Hash Collision Analysis!")
    
    # Example configuration
    collision_config = CollisionTrackingConfig(
        enabled=True,
        max_examples=100000,  # Start with smaller number for testing
        track_coordinates=True,
        export_csv=True,
        export_json=True,
        output_dir="./collision_analysis"
    )
    
    # Initialize Earth4D with collision tracking
    earth4d = Earth4DWithCollisionTracking(
        collision_config=collision_config,
        verbose=True
    )
    
    print(f"Output dimension: {earth4d.output_dim}")
    print(f"Collision tracking: {'Enabled' if collision_config.enabled else 'Disabled'}")
    print(f"Backend available: {TRACKING_AVAILABLE}")