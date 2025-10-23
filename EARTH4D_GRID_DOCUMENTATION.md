# Earth4D Grid Spaces Documentation

## Overview
Earth4D uses a decomposed 4D hash encoding approach with **4 separate 3D grid spaces** to efficiently encode spatiotemporal coordinates (x, y, z, t). This document provides detailed technical documentation for hash collision profiling research.

## The 4 Grid Spaces

### 1. XYZ Grid (Spatial Encoder)
- **Purpose**: Pure spatial encoding of 3D ECEF coordinates
- **Input Coordinates**: `[x, y, z]` from `xyzt[..., :3]`  
- **Default Configuration**:
  - `spatial_levels = 24`
  - `features_per_level = 2`
  - `spatial_log2_hashmap_size = 23` (8.4M entries)
  - `base_spatial_resolution = 16.0`
  - `growth_factor = 2.0`

- **Resolution Range**: ~100km (level 0) → ~9.5cm (level 23)
- **Hash Table**: Single 8.4M entry table
- **Output Dimension**: 48D (24 levels × 2 features)

### 2. XYT Grid (Temporal Projection 1)
- **Purpose**: Encodes X-Y equatorial plane + time dynamics
- **Input Coordinates**: `[x, y, t]` from `xyzt[..., [0,1,3]]`
- **Coordinate Construction**:
  ```python
  xyt = torch.cat([xyzt_scaled[..., :2], xyzt_scaled[..., 3:]], dim=-1)
  ```
- **Default Configuration**:
  - `temporal_levels = 19`
  - `features_per_level = 2` 
  - `temporal_log2_hashmap_size = 18` (256K entries)
  - `base_temporal_resolution = 8.0`
  - `growth_factor = 2.0`

- **Resolution Range**: ~73 days (level 0) → ~0.84 hours (level 18)
- **Hash Table**: 256K entry table
- **Output Dimension**: 38D (19 levels × 2 features)

### 3. YZT Grid (Temporal Projection 2)  
- **Purpose**: Encodes Y-Z meridional plane + time dynamics
- **Input Coordinates**: `[y, z, t]` from `xyzt[..., 1:]`
- **Coordinate Construction**:
  ```python
  yzt = xyzt_scaled[..., 1:]  # Direct slice [y, z, t]
  ```
- **Configuration**: Same as XYT grid
- **Hash Table**: Separate 256K entry table
- **Output Dimension**: 38D (19 levels × 2 features)

### 4. XZT Grid (Temporal Projection 3)
- **Purpose**: Encodes X-Z prime meridian plane + time dynamics  
- **Input Coordinates**: `[x, z, t]` from `xyzt[..., [0,2,3]]`
- **Coordinate Construction**:
  ```python
  xzt = torch.cat([xyzt_scaled[..., :1], xyzt_scaled[..., 2:]], dim=-1)
  ```
- **Configuration**: Same as XYT and YZT grids
- **Hash Table**: Separate 256K entry table  
- **Output Dimension**: 38D (19 levels × 2 features)

## Grid Configuration Summary

| Grid | Levels | Hash Size | Memory | Output Dim | Purpose |
|------|--------|-----------|---------|------------|---------|
| XYZ  | 24     | 2^22 (4M) | ~32MB  | 48D        | Pure spatial |
| XYT  | 19     | 2^18 (256K)| ~8MB   | 38D        | Equatorial + time |
| YZT  | 19     | 2^18 (256K)| ~8MB   | 38D        | Meridional + time |
| XZT  | 19     | 2^18 (256K)| ~8MB   | 38D        | Prime meridian + time |
| **Total** | **81** | **5.75M** | **~56MB** | **162D** | **Full 4D encoding** |

## Hash Collision Analysis Framework

### Memory Requirements for Collision Tracking

For each training example, we track:
- **4 grids** × **levels per grid** × **3 coordinate indices**
- Storage: int16 (2 bytes per index)

**Per Example Memory**:
- XYZ grid: 24 levels × 3 indices × 2 bytes = 144 bytes
- XYT grid: 19 levels × 3 indices × 2 bytes = 114 bytes  
- YZT grid: 19 levels × 3 indices × 2 bytes = 114 bytes
- XZT grid: 19 levels × 3 indices × 2 bytes = 114 bytes
- **Total per example**: 486 bytes

**For 1M examples**: 486MB (within 552MB target)

### Grid Index Tracking Structure

```python
# Proposed tracking tensors (allocated in VRAM)
collision_data = {
    'xyz_indices': torch.zeros((N_examples, 24, 3), dtype=torch.int16, device='cuda'),
    'xyt_indices': torch.zeros((N_examples, 19, 3), dtype=torch.int16, device='cuda'), 
    'yzt_indices': torch.zeros((N_examples, 19, 3), dtype=torch.int16, device='cuda'),
    'xzt_indices': torch.zeros((N_examples, 19, 3), dtype=torch.int16, device='cuda'),
    'original_coords': torch.zeros((N_examples, 4), dtype=torch.float32, device='cuda'),
    'normalized_coords': torch.zeros((N_examples, 4), dtype=torch.float32, device='cuda'),
    'example_count': 0  # Current number of tracked examples
}
```

## Hash Function Analysis

### Collision Patterns by Level

**Direct Indexing (No Collisions)**:
- XYZ: Levels 0-10 (coarse resolutions, stride ≤ 4M)
- XYT/YZT/XZT: Levels 0-8 (coarse resolutions, stride ≤ 256K)

**Hash-Based Indexing (Collisions Expected)**:
- XYZ: Levels 11-23 (fine resolutions, stride > 4M)
- XYT/YZT/XZT: Levels 9-18 (fine resolutions, stride > 256K)

### Expected Collision Ratios

Using the formula: `grid_cells = ceil(base_resolution * growth_factor^level)^3`

**XYZ Grid (4M hash table)**:
- Level 15: ~4.9km resolution, ~530:1 collision ratio
- Level 20: ~76m resolution, ~33,000:1 collision ratio  
- Level 23: ~9.5cm resolution, ~33M:1 collision ratio

**Temporal Grids (256K hash table)**:
- Level 12: ~1.3hr resolution, ~65:1 collision ratio
- Level 15: ~10min resolution, ~4,000:1 collision ratio
- Level 18: ~50sec resolution, ~260,000:1 collision ratio

## Key Insights for Research

1. **Sparsity Advantage**: Earth data is naturally sparse - collisions between empty grid cells don't affect performance

2. **Multi-Scale Context**: Coarse levels (no collisions) provide global context, fine levels (with collisions) add local detail

3. **Hash Function Quality**: XOR with primes should provide uniform distribution, but real Earth coordinates may have spatial clustering

4. **Learned Disambiguation**: The MLP decoder learns to resolve collisions using multi-scale context from all levels

## Implementation Notes

### CUDA Modification Points

1. **`get_grid_index` function**: Return both the hash index AND the original grid coordinates
2. **`kernel_grid` function**: Store grid indices for collision tracking  
3. **Hash vs Direct Indexing**: Track which method was used per level
4. **Memory Management**: Use circular buffer when tracking limit exceeded

### Python Interface Extensions

1. **Earth4D Constructor**: Add collision tracking parameters
2. **Export Functions**: CSV and JSON output for analysis
3. **Statistical Reporting**: Real-time collision metrics
4. **Visualization Tools**: Grid distribution and collision heatmaps

This documentation provides the foundation for implementing comprehensive hash collision profiling in Earth4D.