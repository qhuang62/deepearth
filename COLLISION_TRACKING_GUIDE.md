# Earth4D Collision Tracking - Complete Setup and Usage Guide

## Overview

This guide provides step-by-step instructions for setting up and using the Earth4D collision tracking system for comprehensive hash collision analysis across all 4 grid spaces (xyz, xyt, yzt, xzt).

## Prerequisites

### System Requirements
- **GPU**: NVIDIA GPU with CUDA Compute Capability 7.0+ (RTX 20xx series or newer)
- **CUDA**: CUDA 11.0+ or 12.0+
- **Memory**: 8GB+ GPU memory recommended for large-scale analysis
- **Python**: 3.7+ with PyTorch 2.0+

### Python Dependencies
```bash
pip install torch>=2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas numpy matplotlib tqdm pybind11
```

## Installation

### Step 1: Verify CUDA Installation
```bash
nvcc --version
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 2: Compile the Collision Tracking CUDA Extension
```bash
cd encoders/xyzt/hashencoder
python setup_with_tracking.py build_ext --inplace
```

**Expected Output:**
```
Compiling for CUDA architecture: 75
running build_ext
building 'hashencoder_cuda_tracking' extension
...
creating hashencoder_cuda_tracking.cpython-310-x86_64-linux-gnu.so
```

### Step 3: Verify Installation
```bash
python -c "from hashencoder.hashgrid_with_tracking import HashEncoderWithTracking; print('✓ Installation successful')"
```

### Step 4: Run Real Collision Analysis
```bash
cd ../../..  # Back to deepearth root
python analyze_earth4d_collisions.py
```

## Testing and Validation

### Real Earth4D Collision Analysis
```bash
# Run complete collision analysis on LFMC dataset
python analyze_earth4d_collisions.py
```

### Custom Dataset Test
```bash
# Test with your own coordinates CSV by modifying analyze_earth4d_collisions.py
# Update the script to load your dataset instead of LFMC data
```

## Usage Examples

### Basic Collision Tracking
```python
from encoders.xyzt.earth4d_with_tracking import Earth4DWithCollisionTracking, CollisionTrackingConfig
import torch

# Configure collision tracking
config = CollisionTrackingConfig(
    enabled=True,
    max_examples=100000,
    track_coordinates=True,
    export_csv=True,
    export_json=True,
    output_dir="./collision_analysis"
)

# Initialize Earth4D with collision tracking
earth4d = Earth4DWithCollisionTracking(
    collision_config=config,
    spatial_levels=24,      # Full planetary resolution
    temporal_levels=19,     # 200-year temporal coverage
    verbose=True
)

# Move to GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
earth4d = earth4d.to(device)

# Example coordinates: [latitude, longitude, elevation_m, time_normalized]
coordinates = torch.tensor([
    [40.7128, -74.0060, 100, 0.5],  # New York, year 2000
    [51.5074, -0.1278, 50, 0.75],   # London, year 2050
    [35.6762, 139.6503, 20, 0.25],  # Tokyo, year 1950
], device=device)

# Forward pass with collision tracking
features = earth4d(coordinates, track_collisions=True)
print(f"Output features shape: {features.shape}")  # [3, 162]

# Get collision statistics
stats = earth4d.get_collision_statistics()
print(f"Collision rates by grid:")
for grid_name, grid_stats in stats['grid_statistics'].items():
    if 'collision_analysis' in grid_stats:
        rate = grid_stats['collision_analysis']['overall_collision_rate']
        print(f"  {grid_name.upper()}: {rate:.1%}")

# Export comprehensive analysis
earth4d.export_collision_analysis("./my_collision_analysis")
```

### Batch Processing Large Datasets
```python
import pandas as pd

# Load your dataset
df = pd.read_csv('your_coordinates.csv')
coordinates = torch.tensor(df[['lat', 'lon', 'elev', 'time']].values, dtype=torch.float32)

# Process in batches
batch_size = 1000
total_examples = len(coordinates)

print(f"Processing {total_examples:,} examples in batches of {batch_size:,}")

for i in range(0, total_examples, batch_size):
    batch_end = min(i + batch_size, total_examples)
    batch_coords = coordinates[i:batch_end].to(device)
    
    # Process batch
    features = earth4d(batch_coords)
    
    if (i // batch_size + 1) % 100 == 0:
        print(f"Processed {batch_end:,} / {total_examples:,} examples")

# Print final statistics
earth4d.print_collision_summary()
```

### Advanced Configuration
```python
# High-resolution configuration for detailed analysis
config = CollisionTrackingConfig(
    enabled=True,
    max_examples=1_000_000,    # Track 1M examples
    track_coordinates=True,
    export_csv=True,
    export_json=True,
    output_dir="./detailed_analysis"
)

# Custom Earth4D configuration
earth4d = Earth4DWithCollisionTracking(
    collision_config=config,
    spatial_levels=28,                    # Higher resolution (0.006m)
    temporal_levels=24,                   # Higher temporal resolution (3.2 min)
    spatial_log2_hashmap_size=24,         # 16M entries (4GB)
    temporal_log2_hashmap_size=20,        # 1M entries
    features_per_level=2,
    growth_factor=2.0,
    verbose=True
)
```

## Analysis and Visualization

### Export Data for Analysis
The collision tracking system exports two files:
1. **CSV Data** (`earth4d_collision_analysis.csv`): Raw collision data for statistical analysis
2. **PNG Visualization** (`earth4d_collision_visualization.png`): Comprehensive collision analysis plots

### CSV Column Format
```
example_id,lat,lon,elev,time,x_norm,y_norm,z_norm,t_norm,
xyz_L0_dim0,xyz_L0_dim1,xyz_L0_dim2,xyz_L0_collision,
xyz_L1_dim0,xyz_L1_dim1,xyz_L1_dim2,xyz_L1_collision,
...
xyt_L0_dim0,xyt_L0_dim1,xyt_L0_dim2,xyt_L0_collision,
...
```

### Analyzing Results with Pandas
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load collision data
df = pd.read_csv('./lfmc_collision_analysis/earth4d_collision_analysis.csv')

# Analyze collision patterns by level
collision_cols = [col for col in df.columns if col.endswith('_collision')]
collision_rates = df[collision_cols].mean()

# Plot collision rates by level
plt.figure(figsize=(12, 8))
collision_rates.plot(kind='bar')
plt.title('Hash Collision Rates by Grid Level')
plt.xlabel('Grid Level')
plt.ylabel('Collision Rate')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('collision_rates_by_level.png', dpi=300)
plt.show()

# Spatial distribution analysis
plt.figure(figsize=(10, 6))
plt.scatter(df['lon'], df['lat'], c=df['xyz_L23_collision'], cmap='RdYlBu', alpha=0.5)
plt.colorbar(label='Collision at Finest Level (0.095m)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Spatial Distribution of Hash Collisions')
plt.savefig('collision_spatial_distribution.png', dpi=300)
plt.show()
```

## Troubleshooting

### Common Issues

**1. CUDA Compilation Errors**
```bash
# Solution: Check CUDA version compatibility
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# Try different CUDA architecture
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"
```

**2. Out of Memory Errors**
```python
# Solution: Reduce batch size or max_examples
config = CollisionTrackingConfig(
    enabled=True,
    max_examples=50000,  # Reduce from 1M
    ...
)

# Or use smaller model configuration
earth4d = Earth4DWithCollisionTracking(
    spatial_levels=20,        # Reduce from 24
    temporal_levels=16,       # Reduce from 19
    collision_config=config
)
```

**3. Import Errors**
```bash
# Solution: Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or add path in script
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'encoders/xyzt'))
```

**4. Slow Performance**
```python
# Solution: Use mixed precision and larger batches
earth4d = earth4d.half()  # Use FP16
batch_size = 5000         # Increase batch size
```

### Debug Mode
```bash
# Run with debug output
CUDA_LAUNCH_BLOCKING=1 python analyze_earth4d_collisions.py
```

## Expected Results

### Memory Usage
- **1M examples**: ~486 MB (within 552 MB target)
- **Grid tracking**: ~150 MB per grid space
- **Coordinate tracking**: ~32 MB for 1M examples

### Performance
- **Regular Earth4D**: ~10,000 examples/second
- **With collision tracking**: ~8,000 examples/second (20% overhead)
- **Export**: ~100,000 examples exported in CSV format

### Collision Rates (Typical)
- **XYZ Grid**: 15-25% (higher at fine levels)
- **Temporal Grids**: 10-20% (varies by temporal distribution)
- **Fine levels (L20+)**: 50-90% collision rates (expected due to hash table size)


## References

- [Earth4D Documentation](./encoders/xyzt/README.md)
- [NVIDIA InstantNGP](https://github.com/NVlabs/instant-ngp)
- [Hash Collision Analysis Theory](./EARTH4D_GRID_DOCUMENTATION.md)
