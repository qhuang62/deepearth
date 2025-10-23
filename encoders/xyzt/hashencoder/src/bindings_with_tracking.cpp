#include <torch/extension.h>
#include "hashencoder_with_tracking.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Original functions (for backward compatibility)
    m.def("hash_encode_forward", &hash_encode_forward, "Hash encode forward");
    m.def("hash_encode_backward", &hash_encode_backward, "Hash encode backward");
    
    // New functions with collision tracking
    m.def("hash_encode_forward_with_tracking", &hash_encode_forward_with_tracking, "Hash encode forward with collision tracking");
    
    // Note: We don't need backward pass tracking for collision analysis,
    // since we only care about the forward pass grid indices
}