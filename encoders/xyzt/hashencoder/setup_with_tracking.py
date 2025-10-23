#!/usr/bin/env python3
"""
Setup script for Earth4D Hash Encoder with Collision Tracking
=============================================================

This script compiles the CUDA extension for collision tracking functionality.

Usage:
    python setup_with_tracking.py build_ext --inplace
"""

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import pybind11
import torch
import os

# Get CUDA compute capability
def get_cuda_arch():
    """Get CUDA architecture for compilation."""
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return f'{major}{minor}'
    return '75'  # Default to RTX 20xx series

cuda_arch = get_cuda_arch()
print(f"Compiling for CUDA architecture: {cuda_arch}")

# Define the extension
ext_modules = [
    CUDAExtension(
        name='hashencoder_cuda_tracking',
        sources=[
            'src/hashencoder_with_tracking.cu',
            'src/bindings_with_tracking.cpp',
        ],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++17'],
            'nvcc': [
                '-O3',
                '-std=c++17',
                f'-gencode=arch=compute_{cuda_arch},code=sm_{cuda_arch}',
                '-U__CUDA_NO_HALF_OPERATORS__',
                '-U__CUDA_NO_HALF_CONVERSIONS__',
                '-U__CUDA_NO_HALF2_OPERATORS__',
                '--use_fast_math',
                '--extended-lambda',
                '--expt-relaxed-constexpr'
            ]
        },
        include_dirs=[
            pybind11.get_cmake_dir(),
        ],
    )
]

setup(
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)