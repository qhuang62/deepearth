#!/usr/bin/env python3
"""
Hash Distribution Analysis for Team Lead's High-Level Statistics
================================================================

Analyzes hash usage patterns beyond binary collision rates:
1. Hash activation frequency distribution
2. Scaling behavior with different hash table sizes
3. Point density analysis
4. Spatial/temporal range effects

Usage:
    python hash_distribution_analysis.py --collision-data lfmc_collision_results/lfmc_data/collision_data.pt
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import argparse
from collections import Counter

def load_collision_data(data_path):
    """Load collision data and metadata."""
    print(f"Loading data from {data_path}")
    data = torch.load(data_path, map_location='cpu')
    
    # Load coordinates for range analysis
    coordinates = data['coordinates']['normalized'].numpy()
    
    # Load hash indices
    hash_indices = {}
    for grid in ['xyz', 'xyt', 'yzt', 'xzt']:
        hash_indices[grid] = data['hash_indices'][grid].numpy()
    
    return coordinates, hash_indices

def analyze_hash_activation_distribution(hash_indices, grid_name, level, hash_table_size):
    """Analyze how many times each hash index is activated."""
    level_indices = hash_indices[:, level]
    
    # Count activations per hash index
    activation_counts = Counter(level_indices)
    
    # Statistics
    total_points = len(level_indices)
    unique_hashes = len(activation_counts)
    utilization = unique_hashes / hash_table_size
    
    # Distribution analysis
    activation_values = list(activation_counts.values())
    max_activations = max(activation_values)
    mean_activations = np.mean(activation_values)
    std_activations = np.std(activation_values)
    
    # Percentiles
    p50 = np.percentile(activation_values, 50)
    p90 = np.percentile(activation_values, 90)
    p99 = np.percentile(activation_values, 99)
    
    return {
        'grid': grid_name,
        'level': level,
        'total_points': total_points,
        'unique_hashes': unique_hashes,
        'hash_table_size': hash_table_size,
        'utilization_percent': utilization * 100,
        'max_activations': max_activations,
        'mean_activations': mean_activations,
        'std_activations': std_activations,
        'activation_p50': p50,
        'activation_p90': p90,
        'activation_p99': p99,
        'activation_distribution': activation_counts
    }

def simulate_hash_table_scaling(hash_indices, grid_name, level):
    """Simulate performance with different hash table sizes."""
    level_indices = hash_indices[:, level]
    original_size = 2**22
    
    results = []
    for y in [16, 18, 20, 21, 22, 23]:
        table_size = 2**y
        
        if table_size <= original_size:
            # Map indices to smaller table
            scaled_indices = level_indices % table_size
        else:
            # For larger tables, use original indices
            scaled_indices = level_indices
            
        unique_count = len(np.unique(scaled_indices))
        collision_rate = (len(scaled_indices) - unique_count) / len(scaled_indices)
        utilization = unique_count / table_size
        
        results.append({
            'hash_bits': y,
            'table_size': table_size,
            'unique_hashes': unique_count,
            'collision_rate': collision_rate,
            'utilization_percent': utilization * 100
        })
    
    return results

def analyze_coordinate_ranges(coordinates, hash_indices):
    """Analyze spatial and temporal coordinate ranges."""
    # Assuming coordinates are [x, y, z, t] normalized
    x, y, z, t = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], coordinates[:, 3]
    
    ranges = {
        'x_range': (float(x.min()), float(x.max())),
        'y_range': (float(y.min()), float(y.max())),
        'z_range': (float(z.min()), float(z.max())),
        'temporal_range': (float(t.min()), float(t.max())),
        'total_points': len(coordinates),
        'temporal_span_years': (float(t.max()) - float(t.min())) * 200,  # Assuming 200-year normalization
    }
    
    # Estimate point density (very rough)
    x_span = ranges['x_range'][1] - ranges['x_range'][0]
    y_span = ranges['y_range'][1] - ranges['y_range'][0]
    z_span = ranges['z_range'][1] - ranges['z_range'][0]
    t_span = ranges['temporal_span_years']
    
    # Rough volume estimate (not geographically accurate)
    volume_estimate = x_span * y_span * z_span * t_span
    if volume_estimate > 0:
        ranges['estimated_density_points_per_unit'] = ranges['total_points'] / volume_estimate
    
    return ranges

def create_distribution_plots(results, output_dir):
    """Create visualization plots for hash distribution analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Hash table scaling plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    grids = ['xyz', 'xyt', 'yzt', 'xzt']
    for i, grid in enumerate(grids):
        ax = axes[i//2, i%2]
        
        if grid in results['hash_table_scaling']:
            scaling_data = results['hash_table_scaling'][grid]
            bits = [d['hash_bits'] for d in scaling_data]
            collision_rates = [d['collision_rate'] * 100 for d in scaling_data]
            utilizations = [d['utilization_percent'] for d in scaling_data]
            
            ax2 = ax.twinx()
            
            line1 = ax.plot(bits, collision_rates, 'r-o', label='Collision Rate %')
            line2 = ax2.plot(bits, utilizations, 'b-s', label='Hash Utilization %')
            
            ax.set_xlabel('Hash Table Size (2^x bits)')
            ax.set_ylabel('Collision Rate %', color='r')
            ax2.set_ylabel('Hash Utilization %', color='b')
            ax.set_title(f'{grid.upper()} Grid Scaling')
            ax.grid(True, alpha=0.3)
            
            # Legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='center right')
    
    plt.suptitle('Hash Table Size Scaling Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'hash_table_scaling_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # 2. Activation distribution plot for Level 4
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for i, grid in enumerate(grids):
        ax = axes[i//2, i%2]
        
        if grid in results['level_4_distributions']:
            dist_data = results['level_4_distributions'][grid]
            activation_counts = list(dist_data['activation_distribution'].values())
            
            ax.hist(activation_counts, bins=50, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Activations per Hash Index')
            ax.set_ylabel('Number of Hash Indices')
            ax.set_title(f'{grid.upper()} - Activation Distribution\n'
                        f'Max: {dist_data["max_activations"]}, '
                        f'Mean: {dist_data["mean_activations"]:.1f}')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Hash Index Activation Distribution (Level 4)', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'activation_distribution_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Hash Distribution Analysis')
    parser.add_argument('--collision-data', required=True, help='Path to collision_data.pt')
    parser.add_argument('--output-dir', default='hash_distribution_analysis', help='Output directory')
    args = parser.parse_args()
    
    print("Hash Distribution Analysis for Team Lead")
    print("=" * 50)
    
    # Load data
    coordinates, hash_indices = load_collision_data(args.collision_data)
    
    # Analyze coordinate ranges
    print("\n1. Analyzing coordinate ranges...")
    coord_analysis = analyze_coordinate_ranges(coordinates, hash_indices)
    
    # Analyze hash table scaling for all grids
    print("\n2. Analyzing hash table scaling...")
    scaling_results = {}
    for grid in ['xyz', 'xyt', 'yzt', 'xzt']:
        scaling_results[grid] = simulate_hash_table_scaling(hash_indices[grid], grid, level=4)
    
    # Analyze activation distributions for Level 4
    print("\n3. Analyzing hash activation distributions...")
    level_4_distributions = {}
    for grid in ['xyz', 'xyt', 'yzt', 'xzt']:
        level_4_distributions[grid] = analyze_hash_activation_distribution(
            hash_indices[grid], grid, level=4, hash_table_size=2**22
        )
    
    # Compile results
    results = {
        'coordinate_analysis': coord_analysis,
        'hash_table_scaling': scaling_results,
        'level_4_distributions': level_4_distributions,
        'summary_insights': {
            'spatial_clustering': "Severe spatial clustering due to CONUS-only coverage",
            'temporal_span': f"{coord_analysis['temporal_span_years']:.1f} years (limited vs 200-year capacity)",
            'hash_utilization': "Extremely low hash table utilization suggests smaller tables needed",
            'point_density': "High local density but very sparse global coverage"
        }
    }
    
    # Create plots
    print("\n4. Creating visualization plots...")
    create_distribution_plots(results, args.output_dir)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Counter):
            return dict(obj)
        return obj
    
    # Clean results for JSON
    clean_results = {}
    for key, value in results.items():
        if key == 'level_4_distributions':
            clean_results[key] = {}
            for grid, data in value.items():
                clean_data = {}
                for k, v in data.items():
                    if k == 'activation_distribution':
                        # Only save summary stats, not full distribution
                        continue
                    clean_data[k] = convert_for_json(v)
                clean_results[key][grid] = clean_data
        else:
            clean_results[key] = convert_for_json(value)
    
    with open(output_dir / 'hash_distribution_analysis.json', 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 50)
    print("HASH DISTRIBUTION ANALYSIS COMPLETED")
    print("=" * 50)
    print(f"Results saved to: {output_dir}")
    print("\nKey Findings:")
    print(f"• Temporal span: {coord_analysis['temporal_span_years']:.1f} years")
    print(f"• Total points: {coord_analysis['total_points']:,}")
    print("• Hash table scaling analysis shows optimal sizes:")
    for grid in ['xyz', 'xyt', 'yzt', 'xzt']:
        best_size = None
        best_utilization = 0
        for scale_data in scaling_results[grid]:
            if 1 <= scale_data['utilization_percent'] <= 10:  # Sweet spot
                if scale_data['utilization_percent'] > best_utilization:
                    best_utilization = scale_data['utilization_percent']
                    best_size = scale_data['hash_bits']
        if best_size:
            print(f"  • {grid.upper()}: 2^{best_size} (current: 2^22)")
        else:
            print(f"  • {grid.upper()}: Consider 2^16 or smaller (current: 2^22)")

if __name__ == "__main__":
    main()