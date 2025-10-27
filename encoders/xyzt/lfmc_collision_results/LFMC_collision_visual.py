#!/usr/bin/env python3
"""
Patched LFMC Hash Collision Heatmap Generator
=============================================

Improvements included:
- Fast vectorized mapping from 1D hash indices -> 2D counts
- Robust smoothing & log-quantile scaling with guards for sparse levels
- Fixed layout & safer table styling
- CLI flags for smoothing, scaling, grid size, preview, shared colorbar, save-counts
- Saves raw counts (.npy) optionally for fast re-rendering

Usage:
python LFMC_collision_visual.py \
  --collision-data /home/qhuang62/deepearth/encoders/xyzt/lfmc_collision_results/lfmc_data/collision_data.pt \
  --output-dir /home/qhuang62/deepearth/encoders/xyzt/lfmc_collision_results/collision_heatmaps \
  --grid-size 2000

"""
import argparse
from pathlib import Path
import json
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import ndimage
from tqdm import tqdm

# -------------------------
# Data loading / utilities
# -------------------------
def load_collision_data(data_path):
    """Load collision data from PT file and return CPU tensors/arrays."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Collision data file not found: {data_path}")
    print(f"Loading collision data from {data_path}")
    data = torch.load(str(data_path), map_location='cpu')

    # Coordinates (optional, for metadata)
    coordinates = None
    if 'coordinates' in data and 'normalized' in data['coordinates']:
        coordinates = data['coordinates']['normalized']
        n_points = int(coordinates.shape[0])
        print(f"Loaded {n_points:,} coordinate samples")
    else:
        # infer from hash_indices if coordinates missing
        sample_grid = list(data['hash_indices'].keys())[0]
        n_points = int(data['hash_indices'][sample_grid].shape[0])
        print(f"Coordinates missing in file; inferred {n_points:,} samples from hash indices")

    # Hash indices per grid (expected tensor shape: [N_points, N_levels])
    hash_indices = {}
    for grid in ['xyz', 'xyt', 'yzt', 'xzt']:
        if grid not in data['hash_indices']:
            raise KeyError(f"Expected grid '{grid}' in data['hash_indices']")
        tensor = data['hash_indices'][grid]
        # ensure CPU tensor
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu()
        # convert to numpy for faster processing
        try:
            arr = tensor.numpy()
        except Exception:
            # fallback: convert via torch
            arr = torch.as_tensor(tensor).cpu().numpy()
        hash_indices[grid] = arr.astype(np.int64)
        print(f"{grid.upper()} grid: {arr.shape[0]:,} samples × {arr.shape[1]} levels")

    return coordinates, hash_indices, n_points

# -------------------------
# Core processing functions
# -------------------------
def create_2d_hash_distribution(hash_indices_1d, hash_table_size, grid_size=2000):
    """
    Vectorized mapping of 1D hash indices -> 2D count grid.
    For sparse hash usage, we create a compact visualization showing only the used hash range.
    """
    idx = np.asarray(hash_indices_1d, dtype=np.int64)
    # Remove invalid indices
    mask_valid = (idx >= 0) & (idx < hash_table_size)
    if not np.any(mask_valid):
        return np.zeros((grid_size, grid_size), dtype=np.int32)
    idx = idx[mask_valid]
    
    if len(idx) == 0:
        return np.zeros((grid_size, grid_size), dtype=np.int32)
    
    unique_indices = np.unique(idx)
    utilization = len(unique_indices) / (grid_size * grid_size)
    
    if utilization < 0.001:  # Less than 0.1% utilization - create compact visualization
        # Calculate optimal compact grid size to show all used hashes
        n_unique = len(unique_indices)
        compact_size = min(grid_size, max(int(math.ceil(math.sqrt(n_unique * 10))), 50))
        
        counts = np.zeros((compact_size, compact_size), dtype=np.int32)
        
        # Place hash collisions in a compact grid pattern
        for i, hash_val in enumerate(unique_indices):
            if i >= compact_size * compact_size:
                break
            row = i // compact_size
            col = i % compact_size
            collision_count = np.sum(idx == hash_val)
            counts[row, col] = collision_count
        
        # If compact grid is smaller than target, pad it to center
        if compact_size < grid_size:
            padded_counts = np.zeros((grid_size, grid_size), dtype=np.int32)
            start_row = (grid_size - compact_size) // 2
            start_col = (grid_size - compact_size) // 2
            padded_counts[start_row:start_row+compact_size, start_col:start_col+compact_size] = counts
            return padded_counts
        
        return counts
    else:
        # Standard hash table mapping for normal utilization
        n = min(int(math.ceil(math.sqrt(hash_table_size))), grid_size)
        row = idx // n
        col = idx % n
        # Clip to grid bounds
        valid_mask = (row < n) & (col < n)
        row = row[valid_mask]
        col = col[valid_mask]
        
        counts = np.zeros((n, n), dtype=np.int32)
        np.add.at(counts, (row, col), 1)
        
        # Pad to target grid size if needed
        if n < grid_size:
            padded_counts = np.zeros((grid_size, grid_size), dtype=np.int32)
            padded_counts[:n, :n] = counts
            return padded_counts
        
        return counts

def apply_smoothing_and_scaling(count_grid, smoothing_sigma=1.0, scaling='log_quantile'):
    """
    Smooth + scale the count grid for display.
    Returns a float array normalized to [0,1] for plotting.
    """
    # smoothing (presentation-only)
    if smoothing_sigma and smoothing_sigma > 0:
        smoothed = ndimage.gaussian_filter(count_grid.astype(np.float32), sigma=smoothing_sigma)
    else:
        smoothed = count_grid.astype(np.float32)

    # empty guard
    if smoothed.max() == 0:
        return smoothed.astype(np.float32)

    if scaling == 'log':
        out = np.log1p(smoothed)
        vmax = out.max()
        return (out / vmax).astype(np.float32) if vmax > 0 else out.astype(np.float32)

    if scaling == 'log_quantile':
        out = np.log1p(smoothed)
        nonzero = out[out > 0]
        if nonzero.size == 0:
            return out.astype(np.float32)
        p99 = np.percentile(nonzero, 99)
        p99 = max(p99, 1e-6)
        scaled = np.clip(out / p99, 0.0, 1.0)
        return scaled.astype(np.float32)

    if scaling == 'quantile':
        nonzero = smoothed[smoothed > 0]
        if nonzero.size == 0:
            return smoothed.astype(np.float32)
        p99 = np.percentile(nonzero, 99)
        p99 = max(p99, 1e-6)
        return np.clip(smoothed / p99, 0.0, 1.0).astype(np.float32)

    # linear fallback
    vmax = smoothed.max()
    return (smoothed / vmax).astype(np.float32) if vmax > 0 else smoothed.astype(np.float32)

def compute_basic_stats(count_grid, level_indices, hash_table_size):
    """Compute core statistics from the raw count grid and level indices."""
    total_points = int(len(level_indices))
    unique_hashes = int(len(np.unique(level_indices)))
    collision_rate = (total_points - unique_hashes) / total_points if total_points > 0 else 0.0
    max_collisions = int(count_grid.max()) if count_grid.size > 0 else 0
    nonzero_cells = int((count_grid > 0).sum())
    sparsity = 1.0 - (nonzero_cells / count_grid.size) if count_grid.size > 0 else 1.0
    return {
        'total_points': total_points,
        'unique_hashes': unique_hashes,
        'collision_rate': collision_rate,
        'max_collisions': max_collisions,
        'nonzero_cells': nonzero_cells,
        'total_cells': int(count_grid.size),
        'sparsity': sparsity
    }

# -------------------------
# Plotting helpers (sparse-aware)
# -------------------------
def _is_extremely_sparse(count_grid, threshold_cells=50, fraction=1e-4):
    """Heuristic: true if nonzero cells < threshold or below fraction of grid."""
    nonzero = int((count_grid > 0).sum())
    total = count_grid.size
    return (nonzero < threshold_cells) or (nonzero / total < fraction)

def _plot_processed(ax, count_grid, smoothing, scaling, sparse_guard=True):
    """
    Plot a processed view on given axis. For extremely sparse grids, avoid smoothing +
    quantile normalization — use log1p + LogNorm to preserve hotspots.
    Returns the plotted image object and a small dict of info.
    """
    if sparse_guard and _is_extremely_sparse(count_grid):
        # Diagnostic-friendly: avoid smoothing/quantile normalization
        proc = np.log1p(count_grid.astype(np.float32))
        if np.any(proc > 0):
            vmin = max(proc[proc > 0].min(), 1e-6)
            vmax = np.percentile(proc[proc > 0], 99.5)
            vmax = max(vmax, vmin * 1.001)
            im = ax.imshow(proc, cmap='turbo', origin='lower', aspect='equal', norm=LogNorm(vmin=vmin, vmax=vmax))
        else:
            im = ax.imshow(proc, cmap='turbo', origin='lower', aspect='equal')
        info = {'mode': 'log1p+LogNorm', 'nonzero': int((count_grid > 0).sum())}
        return im, info

    # Default (presentation) pipeline
    proc = apply_smoothing_and_scaling(count_grid, smoothing_sigma=smoothing, scaling=scaling)
    # Use simple linear display (proc is normalized to [0,1] for log_quantile)
    im = ax.imshow(proc, cmap='turbo', origin='lower', aspect='equal', vmin=0.0, vmax=1.0)
    info = {'mode': f'{scaling}+smoothing', 'nonzero': int((count_grid > 0).sum())}
    return im, info

# -------------------------
# Plotting functions
# -------------------------
def create_comparison(all_grids_data, output_dir, level=4,
                               grid_size=1200, smoothing=1.5, scaling='log_quantile',
                               save_counts=False, shared_colorbar=False):
    """
    Create a 2x2 comparison figure for a single level with a summary table row.
    Returns path and stats.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.22], hspace=0.25)

    grids = ['xyz', 'xyt', 'yzt', 'xzt']
    grid_titles = ['XYZ (Spatial)', 'XYT (X-Y-Time)', 'YZT (Y-Z-Time)', 'XZT (X-Z-Time)']

    stats = []
    processed_images = []
    raw_max_vals = []

    # First pass: compute raw count grids and stats (to optionally compute a shared vmax)
    raw_count_grids = {}
    for grid in grids:
        hash_indices_grid, hash_table_size = all_grids_data[grid]
        level_indices = hash_indices_grid[:, level]
        count_grid = create_2d_hash_distribution(level_indices, hash_table_size, grid_size=grid_size)
        raw_count_grids[grid] = (count_grid, level_indices, hash_table_size)
        raw_max_vals.append(int(count_grid.max()))

        if save_counts:
            np.save(Path(output_dir) / f'{grid}_level_{level:02d}_counts.npy', count_grid)

    # determine shared vmax if requested (percentile across nonzero values of all grids)
    shared_vmax = None
    if shared_colorbar:
        all_nonzero = np.concatenate([g[0][g[0] > 0].ravel() for g in raw_count_grids.values() if np.any(g[0] > 0)] or [np.array([1])])
        if all_nonzero.size > 0:
            shared_vmax = np.percentile(all_nonzero, 99.5)
            shared_vmax = max(shared_vmax, 1.0)

    # Plot panels
    for i, (grid, title) in enumerate(zip(grids, grid_titles)):
        r = i // 2
        c = i % 2
        ax = fig.add_subplot(gs[r, c])

        count_grid, level_indices, hash_table_size = raw_count_grids[grid]

        # Plot with sparse-guard
        im, info = _plot_processed(ax, count_grid, smoothing=smoothing, scaling=scaling, sparse_guard=True)
        processed_images.append(im)

        st = compute_basic_stats(count_grid, level_indices, hash_table_size)
        st['grid'] = grid
        st.update(info)
        stats.append(st)

        # Plot processed image colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Normalized collision density / log-counts', fontsize=9)

        ax.set_title(f'{title} — L{level} — {st["collision_rate"]:.1%} collisions', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    # summary table (bottom spanning both columns)
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')
    headers = ['Grid', 'Collision Rate', 'Max Collisions/Hash', 'Hash Utilization', 'Sparsity']
    table_data = []
    for st in stats:
        util = st['nonzero_cells'] / st['total_cells'] if st['total_cells'] > 0 else 0.0
        table_data.append([st['grid'].upper(), f"{st['collision_rate']:.1%}", f"{st['max_collisions']:,}",
                           f"{util:.1%}", f"{st['sparsity']:.1%}"])

    table = ax_table.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    # style header row (colLabels are the top row in Matplotlib table)
    for j in range(len(headers)):
        # (row=0 are column labels)
        cell = table[0, j]
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#333333')

    outpath = Path(output_dir) / f'comparison_level_{level:02d}.png'
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return str(outpath), stats

def create_multi_level_comparison(all_grids_data, output_dir,
                                          levels=(0,2,4,8,12,20), grid_size=800,
                                          smoothing=1.0, scaling='log_quantile', save_counts=False):
    """Create a vertical panel of levels × 4 grids with small colorbars."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    levels = list(levels)
    nrows = len(levels)
    ncols = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    all_stats = {}
    for row_idx, level in enumerate(tqdm(levels, desc='Levels')):
        for col_idx, grid in enumerate(['xyz','xyt','yzt','xzt']):
            ax = axes[row_idx, col_idx]
            hash_indices_grid, hash_table_size = all_grids_data[grid]
            level_indices = hash_indices_grid[:, level]
            count_grid = create_2d_hash_distribution(level_indices, hash_table_size, grid_size=grid_size)

            # sparse-safe plotting
            im, info = _plot_processed(ax, count_grid, smoothing=smoothing, scaling=scaling, sparse_guard=True)

            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(grid.upper(), fontsize=10, weight='bold')

            # annotate leftmost column with level and collision rate
            if col_idx == 0:
                st = compute_basic_stats(count_grid, level_indices, hash_table_size)
                ax.set_ylabel(f'L{level}\n{st["collision_rate"]:.1%}', fontsize=9, weight='bold')

            cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.01)
            cbar.ax.tick_params(labelsize=6)

            # optionally save raw counts
            if save_counts:
                np.save(Path(output_dir) / f'{grid}_level_{level:02d}_counts.npy', count_grid)

            all_stats.setdefault(f'level_{level}', []).append({
                'grid': grid,
                'level': level,
                'collision_rate': compute_basic_stats(count_grid, level_indices, hash_table_size)['collision_rate'],
                'unique_hashes': compute_basic_stats(count_grid, level_indices, hash_table_size)['unique_hashes']
            })

    # place title above figure with reserved space
    plt.suptitle('Earth4D Hash Collision Evolution Across Levels', fontsize=14, y=0.995, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    outpath = Path(output_dir) / 'multilevel_comparison.png'
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(outpath), all_stats

def create_detailed_level_heatmap(hash_indices_grid, level, grid_name, hash_table_size,
                                  output_dir, grid_size=2000, smoothing=1.0, scaling='log_quantile'):
    """Create a 2x2 comparison of processing methods for a single grid-level."""
    output_dir = Path(output_dir)
    detail_dir = output_dir / 'detailed' / grid_name
    detail_dir.mkdir(parents=True, exist_ok=True)

    level_indices = hash_indices_grid[:, level]
    count_grid = create_2d_hash_distribution(level_indices, hash_table_size, grid_size=grid_size)

    methods = [
        ('Raw Counts', lambda x: x.astype(np.float32)),
        ('Log Scale', lambda x: np.log1p(x.astype(np.float32))),
        ('Log + Smoothing', lambda x: apply_smoothing_and_scaling(x, smoothing_sigma=1.0, scaling='log')),
        ('Log-Quantile + Smooth', lambda x: apply_smoothing_and_scaling(x, smoothing_sigma=1.5, scaling='log_quantile'))
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    for i, (name, fn) in enumerate(methods):
        proc = fn(count_grid)
        ax = axes[i]
        # robust vmax
        if proc.size > 0 and np.any(proc > 0):
            vmax = np.percentile(proc[proc > 0], 99.5)
            if vmax <= 0:
                vmax = proc.max()
            im = ax.imshow(proc, cmap='turbo', origin='lower', aspect='equal', vmax=vmax)
        else:
            im = ax.imshow(proc, cmap='turbo', origin='lower', aspect='equal')
        ax.set_title(name, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = plt.colorbar(im, ax=ax, shrink=0.7)
        cbar.ax.tick_params(labelsize=8)

    st = compute_basic_stats(count_grid, level_indices, hash_table_size)
    fig.suptitle(f'Processing Method Comparison - {grid_name.upper()} L{level} — '
                 f'{st["collision_rate"]:.1%} collision, {st["unique_hashes"]:,} unique, max {st["max_collisions"]:,}', fontsize=14)
    plt.tight_layout()
    outpath = detail_dir / f'{grid_name}_level_{level:02d}_comparison.png'
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(outpath)

# -------------------------
# CLI / Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description='Generate LFMC hash collision heatmaps')
    parser.add_argument('--collision-data', required=True, help='Path to collision_data.pt file')
    parser.add_argument('--output-dir', default='lfmc_heatmaps_improved', help='Output directory')
    parser.add_argument('--detailed-levels', nargs='+', type=int, default=list(range(24)),
                        help='Levels for detailed processing comparison (default: all 0-23)')
    parser.add_argument('--grid-size', type=int, default=2000, help='Side length for 2D mapping (max), team lead requested ~2000x2000')
    parser.add_argument('--smoothing', type=float, default=1.5, help='Gaussian smoothing sigma (presentation)')
    parser.add_argument('--scaling', choices=['log', 'log_quantile', 'quantile', 'linear'], default='log_quantile',
                        help='Scaling transformation for visualization')
    parser.add_argument('--preview', action='store_true', help='Run a small preview (reduced grid size) and exit')
    parser.add_argument('--preview-size', type=int, default=300, help='Grid size for preview mode')
    parser.add_argument('--shared-cbar', action='store_true', help='Use shared color range across panels')
    parser.add_argument('--save-counts', action='store_true', help='Save raw count grids as .npy for reuse')
    args = parser.parse_args()

    print("\nLFMC Hash Collision Heatmap Generator\n" + "="*72)
    print(f"Collision data: {args.collision_data}")
    print(f"Output dir: {args.output_dir}")
    print(f"Grid size: {args.grid_size}, Smoothing: {args.smoothing}, Scaling: {args.scaling}")
    if args.preview:
        print("Preview mode: ON (will use reduced grid size and produce quick visuals)")
    print("="*72)

    coordinates, hash_indices, n_points = load_collision_data(args.collision_data)

    # hash_table_size defaults (2^22) unless you want to change
    hash_table_sizes = {'xyz': 2**22, 'xyt': 2**22, 'yzt': 2**22, 'xzt': 2**22}

    all_grids_data = {}
    for grid in ['xyz','xyt','yzt','xzt']:
        if grid not in hash_indices:
            raise KeyError(f"Missing hash indices for {grid}")
        all_grids_data[grid] = (hash_indices[grid], hash_table_sizes[grid])

    # Preview quick-run
    if args.preview:
        small_dir = Path(args.output_dir) / 'preview'
        small_dir.mkdir(parents=True, exist_ok=True)
        gs = min(args.preview_size, args.grid_size)
        print(f"Creating preview comparison (L4) at {gs}×{gs} ...")
        ppath, pstats = create_comparison(all_grids_data, small_dir, level=4,
                                                  grid_size=gs, smoothing=args.smoothing,
                                                  scaling=args.scaling, save_counts=args.save_counts,
                                                  shared_colorbar=args.shared_cbar)
        print("Preview saved to:", ppath)
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Single-level comparison (default Level 4)
    print("\nCreating Level 4 comparison...")
    comparison_path, stats = create_comparison(all_grids_data, out_dir, level=4,
                                                       grid_size=args.grid_size, smoothing=args.smoothing,
                                                       scaling=args.scaling, save_counts=args.save_counts,
                                                       shared_colorbar=args.shared_cbar)
    print(f"✅ Comparison saved: {comparison_path}")

    # Multi-level evolution figure (ALL 24 levels as requested by team lead)
    print("\nCreating multi-level comparison for ALL 24 levels...")
    level_list = list(range(24))  # All levels 0-23 as team lead requested
    multilevel_path, multilevel_stats = create_multi_level_comparison(all_grids_data, out_dir,
                                                                               levels=level_list, grid_size=1200,  # Use fixed size for multi-level readability
                                                                               smoothing=max(0.8, args.smoothing - 0.5),
                                                                               scaling=args.scaling,
                                                                               save_counts=args.save_counts)
    print(f"✅ Multi-level comparison saved: {multilevel_path}")

    # Detailed per-level processing comparisons
    print(f"\nCreating detailed processing comparisons for levels: {args.detailed_levels}")
    detailed_paths = {}
    for grid in ['xyz','xyt','yzt','xzt']:
        detailed_paths[grid] = []
        hash_indices_grid, hash_table_size = all_grids_data[grid]
        for level in args.detailed_levels:
            if level < hash_indices_grid.shape[1]:
                dp = create_detailed_level_heatmap(hash_indices_grid, level, grid, hash_table_size,
                                                   out_dir, grid_size=args.grid_size,
                                                   smoothing=args.smoothing, scaling=args.scaling)
                detailed_paths[grid].append(dp)
                print(f"  ✅ {grid.upper()} L{level}: {dp}")

    # Save summary JSON
    stats = {
        'single_level_stats': stats,
        'multilevel_stats': multilevel_stats,
        'detailed_paths': detailed_paths,
        'params': {
            'grid_size': args.grid_size,
            'smoothing': args.smoothing,
            'scaling': args.scaling,
            'preview': args.preview,
            'shared_colorbar': args.shared_cbar
        }
    }
    stats_path = out_dir / 'analysis_summary.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n" + "="*72)
    print("HEATMAP GENERATION COMPLETED")
    print("="*72)
    print(f"Saved outputs to: {out_dir}")
    print(f"Summary JSON: {stats_path}")

if __name__ == "__main__":
    main()
