# Earth4D Collision Analysis - Setup and Usage Guide

## Overview

This guide provides instructions for running Earth4D hash collision analysis using real LFMC data. The analysis examines collision patterns across all 4 grid spaces (xyz, xyt, yzt, xzt) using the working `analyze_earth4d_collisions.py` implementation.

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

### Step 2: Verify Earth4D Installation
```bash
cd encoders/xyzt/hashencoder
# Ensure the original CUDA extension is compiled
ls hashencoder_cuda.so
```

### Step 3: Run Collision Analysis
```bash
cd ../../..  # Back to deepearth root
python analyze_earth4d_collisions.py
```

## Running the Analysis

### Basic Collision Analysis
```bash
# Run complete collision analysis on LFMC dataset
python analyze_earth4d_collisions.py
```

### Custom Dataset Analysis
```bash
# To use your own dataset, modify analyze_earth4d_collisions.py:
# 1. Update the data_path in load_lfmc_data() function
# 2. Ensure your CSV has columns: latitude, longitude, elevation, time
# 3. Run the script normally
python analyze_earth4d_collisions.py
```

## Usage Examples

### Understanding the Analysis Script
```python
# The analyze_earth4d_collisions.py script works as follows:

from encoders.xyzt.earth4d import Earth4D
import torch

# 1. Initialize standard Earth4D (uses working CUDA extension)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
earth4d = Earth4D(verbose=False).to(device)

# 2. Load LFMC dataset
coords_xyzt = load_lfmc_data('globe_lfmc_extracted.csv', max_samples=5000)

# 3. Create collision tracker that analyzes hash patterns
tracker = CollisionTracker(earth4d)

# 4. Run collision analysis across all 4 grids
collision_stats = tracker.analyze_all_encoders(coords_xyzt)

# 5. Export results
# - CSV: lfmc_collision_analysis/earth4d_collision_analysis.csv
# - PNG: lfmc_collision_analysis/earth4d_collision_visualization.png
```

### Modifying the Analysis
```python
# To analyze different datasets or parameters, edit analyze_earth4d_collisions.py:

# 1. Change dataset size:
coords_xyzt = load_lfmc_data('your_dataset.csv', max_samples=10000)

# 2. Modify Earth4D configuration:
earth4d = Earth4D(
    spatial_levels=24,      # Spatial resolution levels
    temporal_levels=19,     # Temporal resolution levels
    verbose=True           # Enable detailed output
)

# 3. Change output directory:
output_dir = "./custom_collision_analysis"

# 4. The script automatically processes data in batches and
#    generates collision statistics for all 4 grids (xyz, xyt, yzt, xzt)
```

### Expected Output
```
================================================================================
EARTH4D COLLISION ANALYSIS - LFMC DATASET
================================================================================
Analyzing 5,000 LFMC coordinates

Analyzing xyz encoder collisions...
Analyzing xyt encoder collisions...
Analyzing yzt encoder collisions...
Analyzing xzt encoder collisions...

================================================================================
COLLISION ANALYSIS SUMMARY
================================================================================
Dataset: 5,000 LFMC coordinates
Total encoders analyzed: 4 (xyz, xyt, yzt, xzt)
Total levels analyzed: 81

XYZ Encoder:
  Levels analyzed: 24
  Levels with hash collisions: 20
  Average collision rate: 96.9%
  Maximum collision rate: 99.8%

Files generated:
  - lfmc_collision_analysis/earth4d_collision_analysis.csv
  - lfmc_collision_analysis/earth4d_collision_visualization.png
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
# Solution: Reduce dataset size in analyze_earth4d_collisions.py
# Edit line ~380:
coords_xyzt = load_lfmc_data('globe_lfmc_extracted.csv', max_samples=1000)

# Or use smaller Earth4D configuration:
earth4d = Earth4D(
    spatial_levels=20,        # Reduce from 24
    temporal_levels=16,       # Reduce from 19
    verbose=False
)
```

**3. Import Errors**
```bash
# Solution: Ensure you're in the correct directory
cd /path/to/deepearth
python analyze_earth4d_collisions.py

# Or check if Earth4D can be imported:
python -c "from encoders.xyzt.earth4d import Earth4D; print('✓ Earth4D available')"
```

**4. Slow Performance**
```bash
# Solution: Ensure CUDA is being used
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"

# Or reduce dataset size for faster testing:
# Edit analyze_earth4d_collisions.py to use max_samples=1000
```

### Debug Mode
```bash
# Run with debug output
CUDA_LAUNCH_BLOCKING=1 python analyze_earth4d_collisions.py
```

## Expected Results

### Performance
- **Analysis speed**: ~2-3 minutes for 5,000 LFMC coordinates
- **Memory usage**: Standard Earth4D GPU memory requirements
- **Output files**: CSV (~2MB) and PNG visualization (~500KB)

### Collision Rates (LFMC Dataset Results)
- **XYZ Grid**: ~96.9% average collision rate
- **XYT Grid**: ~86.9% average collision rate  
- **YZT Grid**: ~85.4% average collision rate
- **XZT Grid**: ~86.6% average collision rate

## Key Files

### Main Analysis Script
- **`analyze_earth4d_collisions.py`** - Complete working collision analysis implementation

### Documentation  
- **`EARTH4D_GRID_DOCUMENTATION.md`** - Technical specifications of Earth4D's 4 grid spaces
- **`COLLISION_TRACKING_GUIDE.md`** - This usage guide

### Required Data
- **`globe_lfmc_extracted.csv`** - LFMC dataset for collision analysis

### Output Files
- **`lfmc_collision_analysis/earth4d_collision_analysis.csv`** - Detailed collision statistics
- **`lfmc_collision_analysis/earth4d_collision_visualization.png`** - Analysis plots

## References

- [NVIDIA InstantNGP](https://github.com/NVlabs/instant-ngp)
- [Hash Collision Analysis Theory](./EARTH4D_GRID_DOCUMENTATION.md)
