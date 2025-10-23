#!/bin/bash
# Earth4D Collision Tracking Setup Script
# ========================================
# This script sets up the complete collision tracking environment

set -e  # Exit on any error

echo "Earth4D Collision Tracking Setup"
echo "===================================="

# Check CUDA installation
echo "Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    echo "✓ CUDA compiler found: $(nvcc --version | grep release)"
else
    echo "CUDA compiler not found. Please install CUDA toolkit."
    exit 1
fi

if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA driver found"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1
else
    echo "NVIDIA driver not found"
    exit 1
fi

# Set CUDA_HOME if not set
if [ -z "$CUDA_HOME" ]; then
    # Try common CUDA installation paths
    for cuda_path in /usr/local/cuda /opt/cuda /usr/local/cuda-* /opt/cuda-*; do
        if [ -d "$cuda_path" ]; then
            export CUDA_HOME="$cuda_path"
            echo "✓ Set CUDA_HOME to $CUDA_HOME"
            break
        fi
    done
    
    if [ -z "$CUDA_HOME" ]; then
        echo "Could not find CUDA installation. Please set CUDA_HOME manually:"
        echo "export CUDA_HOME=/path/to/cuda"
        exit 1
    fi
fi

# Add CUDA to PATH
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

echo "✓ CUDA environment configured"

# Check Python dependencies
echo "Checking Python dependencies..."
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__} found')"
python3 -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')"

if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "PyTorch CUDA not available. Please install PyTorch with CUDA support:"
    echo "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    exit 1
fi

# Install additional dependencies
echo "Installing Python dependencies..."
pip install pandas numpy matplotlib tqdm pybind11 -q

# Navigate to hashencoder directory
cd encoders/xyzt/hashencoder

# Compile original hash encoder
echo "Compiling original hash encoder..."
if [ ! -f "hashencoder_cuda.so" ]; then
    python3 setup.py build_ext --inplace
    echo "✓ Original hash encoder compiled"
else
    echo "✓ Original hash encoder already compiled"
fi

# Compile collision tracking extension
echo "Compiling collision tracking extension..."
if [ -f "setup_with_tracking.py" ]; then
    python3 setup_with_tracking.py build_ext --inplace
    echo "✓ Collision tracking extension compiled"
else
    echo "Collision tracking setup script not found, skipping..."
fi

# Go back to root directory
cd ../../..

# Test the installation
echo "Testing installation..."
python3 -c "
try:
    from encoders.xyzt.hashencoder.hashgrid import HashEncoder
    print('✓ Original HashEncoder imported successfully')
except Exception as e:
    print(f'Failed to import HashEncoder: {e}')
    exit(1)

try:
    from encoders.xyzt.earth4d import Earth4D
    print('✓ Original Earth4D imported successfully')
except Exception as e:
    print(f'Failed to import Earth4D: {e}')
    exit(1)

# Test basic functionality
import torch
encoder = Earth4D(verbose=False)
coords = torch.tensor([[40.7, -74.0, 100, 0.5]])
if torch.cuda.is_available():
    encoder = encoder.cuda()
    coords = coords.cuda()
features = encoder(coords)
print(f'✓ Basic Earth4D test passed: {features.shape}')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "Setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Run real collision analysis: python analyze_earth4d_collisions.py"
    echo "2. Analyze real LFMC collision data: ensure globe_lfmc_extracted.csv is available"
    echo "3. Read the guide: cat COLLISION_TRACKING_GUIDE.md"
    echo ""
    echo "For any issues, check the troubleshooting section in COLLISION_TRACKING_GUIDE.md"
else
    echo ""
    echo "Setup failed. Please check the error messages above."
    echo "Common issues:"
    echo "1. CUDA_HOME not set correctly"
    echo "2. PyTorch not compiled with CUDA support"
    echo "3. Incompatible CUDA/PyTorch versions"
    echo ""
    echo "See COLLISION_TRACKING_GUIDE.md for detailed troubleshooting."
fi