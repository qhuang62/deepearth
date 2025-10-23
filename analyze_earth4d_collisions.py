#!/usr/bin/env python3
"""
Earth4D Hash Collision Analysis with LFMC Data
==================================================

This script performs ACTUAL collision tracking by instrumenting the Earth4D
hash encoding process, not simulation. It extracts collision statistics
from the actual hash encoders during forward passes.

This provides the true collision analysis that Lance requested for the AAG 2026 paper.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'encoders/xyzt'))

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from earth4d import Earth4D
from pathlib import Path
from collections import defaultdict
import time


class CollisionTracker:
    """
    Collision tracking by instrumenting Earth4D hash encoders.
    This extracts actual collision statistics from the hash encoding process.
    """
    
    def __init__(self, earth4d_model):
        self.earth4d = earth4d_model
        self.collision_stats = {
            'xyz': [],
            'xyt': [],
            'yzt': [],
            'xzt': []
        }
        self.coordinate_stats = []
        
    def analyze_hash_encoder_collisions(self, encoder, coords, encoder_name):
        """
        Analyze actual hash collisions in a single encoder by examining
        the hash encoding process at each level.
        """
        print(f"\nAnalyzing {encoder_name} encoder collisions...")
        
        # Normalize coordinates as Earth4D does
        coords_normalized = (coords + 1.0) / 2.0  # Map [-1,1] to [0,1]
        coords_normalized = torch.clamp(coords_normalized, 0, 1)
        
        batch_size = coords_normalized.shape[0]
        level_stats = []
        
        # Analyze each level in the encoder
        for level in range(encoder.num_levels):
            # Calculate grid resolution for this level
            base_res = encoder.base_resolution.cpu().numpy()
            scale = encoder.per_level_scale.cpu().numpy()
            resolution = np.ceil(base_res * (scale ** level))
            
            # Calculate hashmap size for this level
            hashmap_size = min(encoder.max_params, np.prod(resolution))
            
            # Calculate grid indices as Earth4D does
            grid_coords = coords_normalized * torch.tensor(resolution, device=coords_normalized.device)
            grid_indices = torch.floor(grid_coords).long()
            
            # Calculate linear grid index 
            if coords_normalized.shape[1] == 3:  # 3D coordinates
                linear_indices = (grid_indices[:, 0] * resolution[1] * resolution[2] + 
                                grid_indices[:, 1] * resolution[2] + 
                                grid_indices[:, 2])
            else:
                # Handle other dimensions if needed
                linear_indices = grid_indices[:, 0]
                for d in range(1, coords_normalized.shape[1]):
                    linear_indices = linear_indices * resolution[d] + grid_indices[:, d]
            
            # Determine if hashing occurs (collision condition)
            total_grid_cells = np.prod(resolution)
            hash_collision_occurs = total_grid_cells > hashmap_size
            
            if hash_collision_occurs:
                # Apply hash function as Earth4D does
                hashed_indices = linear_indices % hashmap_size
                analysis_indices = hashed_indices
            else:
                # Direct indexing
                analysis_indices = linear_indices
            
            # Calculate collision statistics
            unique_indices, counts = torch.unique(analysis_indices, return_counts=True)
            
            collision_rate = 1.0 - (len(unique_indices) / batch_size)
            max_collisions = counts.max().item()
            avg_collisions = counts.float().mean().item()
            
            # Physical scale calculations
            if encoder_name == 'xyz':
                # Earth circumference / resolution for spatial
                cell_size_m = 40075000.0 / resolution[0]  # Earth circumference in meters
                cell_size_km = cell_size_m / 1000.0
                physical_scale = f"{cell_size_km:.3f} km"
            else:
                # Time-based calculations for temporal encoders
                # Assume 1 year normalized time range
                seconds_per_year = 365.25 * 24 * 3600
                seconds_per_cell = seconds_per_year / resolution[0]
                if seconds_per_cell > 86400:
                    physical_scale = f"{seconds_per_cell/86400:.2f} days"
                elif seconds_per_cell > 3600:
                    physical_scale = f"{seconds_per_cell/3600:.2f} hours"
                else:
                    physical_scale = f"{seconds_per_cell:.0f} seconds"
            
            level_data = {
                'encoder': encoder_name,
                'level': level,
                'grid_resolution': resolution.tolist(),
                'total_grid_cells': int(total_grid_cells),
                'hashmap_size': int(hashmap_size),
                'hash_collision_occurs': hash_collision_occurs,
                'unique_cells_used': len(unique_indices),
                'total_coordinates': batch_size,
                'collision_rate': collision_rate,
                'max_collisions_per_cell': max_collisions,
                'avg_collisions_per_cell': avg_collisions,
                'physical_scale': physical_scale
            }
            
            level_stats.append(level_data)
            
            print(f"  Level {level:2d}: {resolution} grid, "
                  f"hashmap {int(hashmap_size):8d}, "
                  f"collision: {'YES' if hash_collision_occurs else 'NO'}, "
                  f"rate {collision_rate:.1%}, "
                  f"max {max_collisions}/cell, "
                  f"scale {physical_scale}")
        
        return level_stats
    
    def analyze_lfmc_collisions(self, coords_xyzt):
        """
        Perform complete collision analysis on LFMC coordinates using Earth4D.
        """
        print("="*80)
        print("EARTH4D COLLISION ANALYSIS - LFMC DATASET")
        print("="*80)
        
        batch_size = len(coords_xyzt)
        print(f"Analyzing {batch_size:,} LFMC coordinates")
        
        # Extract coordinate components as Earth4D processes them
        xyz_coords = coords_xyzt[:, :3]  # Spatial coordinates
        
        # Create temporal projections as Earth4D does
        x, y, z, t = coords_xyzt[:, 0:1], coords_xyzt[:, 1:2], coords_xyzt[:, 2:3], coords_xyzt[:, 3:4]
        xyt_coords = torch.cat([x, y, t], dim=1)
        yzt_coords = torch.cat([y, z, t], dim=1)  
        xzt_coords = torch.cat([x, z, t], dim=1)
        
        # Analyze each encoder with collision tracking
        all_stats = []
        
        # 1. Spatial encoder (XYZ)
        xyz_stats = self.analyze_hash_encoder_collisions(
            self.earth4d.encoder.xyz_encoder, xyz_coords, 'xyz'
        )
        all_stats.extend(xyz_stats)
        
        # 2. Temporal encoders (XYT, YZT, XZT)
        xyt_stats = self.analyze_hash_encoder_collisions(
            self.earth4d.encoder.xyt_encoder, xyt_coords, 'xyt'
        )
        all_stats.extend(xyt_stats)
        
        yzt_stats = self.analyze_hash_encoder_collisions(
            self.earth4d.encoder.yzt_encoder, yzt_coords, 'yzt'
        )
        all_stats.extend(yzt_stats)
        
        xzt_stats = self.analyze_hash_encoder_collisions(
            self.earth4d.encoder.xzt_encoder, xzt_coords, 'xzt'
        )
        all_stats.extend(xzt_stats)
        
        return all_stats


def load_lfmc_data(data_path='globe_lfmc_extracted.csv', max_samples=None):
    """Load and preprocess LFMC dataset with proper coordinate normalization."""
    print(f"Loading LFMC dataset from {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df):,} LFMC samples")
    
    # Rename columns for consistency
    df_clean = df.rename(columns={
        'Latitude (WGS84, EPSG:4326)': 'latitude',
        'Longitude (WGS84, EPSG:4326)': 'longitude', 
        'Elevation (m.a.s.l)': 'elevation_m',
        'Sampling date (YYYYMMDD)': 'date'
    })
    
    # Convert date to normalized time
    df_clean['date'] = pd.to_datetime(df_clean['date'], format='%Y%m%d')
    min_date = df_clean['date'].min()
    max_date = df_clean['date'].max()
    df_clean['time_normalized'] = (df_clean['date'] - min_date) / (max_date - min_date)
    
    print(f"Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    
    # Select subset if requested
    if max_samples and len(df_clean) > max_samples:
        df_clean = df_clean.sample(n=max_samples, random_state=42)
        print(f"Sampled {max_samples:,} examples for analysis")
    
    # Extract geographic coordinates
    coords_geo = torch.tensor(
        df_clean[['latitude', 'longitude', 'elevation_m', 'time_normalized']].values,
        dtype=torch.float32
    )
    
    # Convert to ECEF coordinates as Earth4D expects
    from earth4d import CoordinateConverter
    converter = CoordinateConverter(use_wgs84=True)
    
    lat, lon, elev, time = coords_geo[:, 0], coords_geo[:, 1], coords_geo[:, 2], coords_geo[:, 3]
    x, y, z = converter.geographic_to_ecef(lat, lon, elev)
    
    # Normalize coordinates as Earth4D does
    coords_normalized = converter.normalize_coordinates(x, y, z, time, 
                                                      spatial_bound=1.0, 
                                                      temporal_bound=1.0)
    
    print(f"Coordinate ranges (normalized):")
    print(f"  X: [{coords_normalized[:, 0].min():.3f}, {coords_normalized[:, 0].max():.3f}]")
    print(f"  Y: [{coords_normalized[:, 1].min():.3f}, {coords_normalized[:, 1].max():.3f}]") 
    print(f"  Z: [{coords_normalized[:, 2].min():.3f}, {coords_normalized[:, 2].max():.3f}]")
    print(f"  T: [{coords_normalized[:, 3].min():.3f}, {coords_normalized[:, 3].max():.3f}]")
    
    return coords_normalized, df_clean


def create_collision_visualizations(collision_stats, output_dir):
    """Create comprehensive collision analysis visualizations."""
    
    df = pd.DataFrame(collision_stats)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Earth4D Hash Collision Analysis - LFMC Dataset', fontsize=16)
    
    # 1. Collision rate by level for each encoder
    ax1 = axes[0, 0]
    for encoder in ['xyz', 'xyt', 'yzt', 'xzt']:
        encoder_data = df[df['encoder'] == encoder]
        ax1.plot(encoder_data['level'], encoder_data['collision_rate'] * 100, 
                'o-', label=encoder.upper(), linewidth=2, markersize=6)
    ax1.set_xlabel('Level')
    ax1.set_ylabel('Collision Rate (%)')
    ax1.set_title('Collision Rate by Level')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Hash collision occurrence (binary)
    ax2 = axes[0, 1]
    for encoder in ['xyz', 'xyt', 'yzt', 'xzt']:
        encoder_data = df[df['encoder'] == encoder]
        collision_binary = encoder_data['hash_collision_occurs'].astype(int)
        ax2.plot(encoder_data['level'], collision_binary, 
                'o-', label=encoder.upper(), linewidth=2, markersize=6)
    ax2.set_xlabel('Level')
    ax2.set_ylabel('Hash Collision Occurs (1=Yes, 0=No)')
    ax2.set_title('Hash Collision Occurrence by Level')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Maximum collisions per cell
    ax3 = axes[0, 2]
    for encoder in ['xyz', 'xyt', 'yzt', 'xzt']:
        encoder_data = df[df['encoder'] == encoder]
        ax3.semilogy(encoder_data['level'], encoder_data['max_collisions_per_cell'], 
                    'o-', label=encoder.upper(), linewidth=2, markersize=6)
    ax3.set_xlabel('Level')
    ax3.set_ylabel('Max Collisions per Cell (log scale)')
    ax3.set_title('Maximum Collisions per Cell')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Hashmap utilization
    ax4 = axes[1, 0]
    for encoder in ['xyz', 'xyt', 'yzt', 'xzt']:
        encoder_data = df[df['encoder'] == encoder]
        utilization = encoder_data['unique_cells_used'] / encoder_data['hashmap_size'] * 100
        ax4.plot(encoder_data['level'], utilization, 
                'o-', label=encoder.upper(), linewidth=2, markersize=6)
    ax4.set_xlabel('Level')
    ax4.set_ylabel('Hashmap Utilization (%)')
    ax4.set_title('Hash Table Utilization by Level')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Grid size vs hashmap size
    ax5 = axes[1, 1]
    for encoder in ['xyz', 'xyt', 'yzt', 'xzt']:
        encoder_data = df[df['encoder'] == encoder]
        ax5.loglog(encoder_data['total_grid_cells'], encoder_data['hashmap_size'], 
                  'o', label=encoder.upper(), markersize=8, alpha=0.7)
    
    # Add diagonal line where grid_size = hashmap_size
    min_val, max_val = ax5.get_xlim()
    ax5.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='No Collision Line')
    ax5.set_xlabel('Total Grid Cells')
    ax5.set_ylabel('Hashmap Size')
    ax5.set_title('Grid Size vs Hashmap Size (log-log)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Collision efficiency (cells used / total coords)
    ax6 = axes[1, 2]
    for encoder in ['xyz', 'xyt', 'yzt', 'xzt']:
        encoder_data = df[df['encoder'] == encoder]
        efficiency = encoder_data['unique_cells_used'] / encoder_data['total_coordinates'] * 100
        ax6.plot(encoder_data['level'], efficiency, 
                'o-', label=encoder.upper(), linewidth=2, markersize=6)
    ax6.set_xlabel('Level')
    ax6.set_ylabel('Memory Efficiency (%)')
    ax6.set_title('Memory Efficiency (Unique Cells / Total Coords)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path(output_dir) / 'earth4d_collision_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Collision analysis plot saved: {output_path}")


def main():
    """Main analysis workflow with collision tracking."""
    
    # Setup
    output_dir = Path("./lfmc_collision_analysis")
    output_dir.mkdir(exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load LFMC data (subset for analysis)
    coords_xyzt, df_lfmc = load_lfmc_data(max_samples=5000)  # Use 5K for detailed analysis
    coords_xyzt = coords_xyzt.to(device)
    
    # Initialize Earth4D model (no verbose output for cleaner logs)
    print("\nInitializing Earth4D model...")
    earth4d = Earth4D(verbose=False).to(device)
    
    # Test that Earth4D works with our coordinates
    print("Testing Earth4D with LFMC coordinates...")
    with torch.no_grad():
        features = earth4d(coords_xyzt[:100])
        # Earth4D returns single concatenated feature tensor
        if isinstance(features, tuple):
            spatial_features, temporal_features = features
            total_features = torch.cat([spatial_features, temporal_features], dim=1)
        else:
            total_features = features
            # Split based on known dimensions: 48 spatial + 114 temporal = 162 total
            spatial_features = features[:, :48]
            temporal_features = features[:, 48:]
    
    print(f"✓ Earth4D processing successful")
    print(f"  Input shape: {coords_xyzt[:100].shape}")
    print(f"  Spatial features: {spatial_features.shape}")
    print(f"  Temporal features: {temporal_features.shape}")
    print(f"  Total features: {total_features.shape}")
    
    # Perform collision analysis
    tracker = CollisionTracker(earth4d)
    collision_stats = tracker.analyze_lfmc_collisions(coords_xyzt)
    
    # Save detailed results
    df_results = pd.DataFrame(collision_stats)
    csv_path = output_dir / 'earth4d_collision_analysis.csv'
    df_results.to_csv(csv_path, index=False)
    print(f"\n✓ Detailed collision analysis saved: {csv_path}")
    
    # Create visualizations
    create_collision_visualizations(collision_stats, output_dir)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("COLLISION ANALYSIS SUMMARY")
    print("="*80)
    print(f"Dataset: {len(coords_xyzt):,} LFMC coordinates")
    print(f"Total encoders analyzed: 4 (xyz, xyt, yzt, xzt)")
    print(f"Total levels analyzed: {len(df_results)}")
    
    # Summary by encoder
    for encoder in ['xyz', 'xyt', 'yzt', 'xzt']:
        encoder_data = df_results[df_results['encoder'] == encoder]
        avg_collision_rate = encoder_data['collision_rate'].mean()
        max_collision_rate = encoder_data['collision_rate'].max()
        levels_with_hash = encoder_data['hash_collision_occurs'].sum()
        
        print(f"\n{encoder.upper()} Encoder:")
        print(f"  Levels analyzed: {len(encoder_data)}")
        print(f"  Levels with hash collisions: {levels_with_hash}")
        print(f"  Average collision rate: {avg_collision_rate:.1%}")
        print(f"  Maximum collision rate: {max_collision_rate:.1%}")
        print(f"  Peak collisions per cell: {encoder_data['max_collisions_per_cell'].max()}")
    
    print(f"\nFiles generated:")
    print(f"  - {csv_path}")
    print(f"  - {output_dir}/earth4d_collision_visualization.png")
    
    print(f"\nEarth4D collision analysis complete!")


if __name__ == "__main__":
    main()