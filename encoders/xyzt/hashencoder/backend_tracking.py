import os
import sys

# Try to import the installed CUDA extension with collision tracking
try:
    import hashencoder_cuda_tracking as _backend
except ImportError:
    # Try to import from local directory
    import os
    import sys
    module_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(module_dir, 'hashencoder_cuda_tracking.cpython-312-x86_64-linux-gnu.so')

    if os.path.exists(module_path):
        # Load the compiled module directly
        import importlib.util
        spec = importlib.util.spec_from_file_location("hashencoder_cuda_tracking", module_path)
        _backend = importlib.util.module_from_spec(spec)
        sys.modules["hashencoder_cuda_tracking"] = _backend
        spec.loader.exec_module(_backend)
    else:
        print(f"Warning: hashencoder_cuda_tracking.so not found at {module_path}")
        print("Available files in directory:")
        for f in os.listdir(module_dir):
            if 'hashencoder' in f:
                print(f"  {f}")
        
        # Check if we can find any tracking extension
        for ext_name in ['hashencoder_cuda_tracking.cpython-312-x86_64-linux-gnu.so', 
                         'hashencoder_cuda_tracking.so']:
            alt_path = os.path.join(module_dir, ext_name)
            if os.path.exists(alt_path):
                print(f"Found tracking extension at: {alt_path}")
                spec = importlib.util.spec_from_file_location("hashencoder_cuda_tracking", alt_path)
                _backend = importlib.util.module_from_spec(spec)
                sys.modules["hashencoder_cuda_tracking"] = _backend
                spec.loader.exec_module(_backend)
                break
        else:
            _backend = None

__all__ = ['_backend']