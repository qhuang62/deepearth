#ifndef _HASH_ENCODE_WITH_TRACKING_H
#define _HASH_ENCODE_WITH_TRACKING_H

#include <stdint.h>
#include <torch/torch.h>
#include <torch/extension.h>

// Original hash encoding functions (unchanged)
// inputs: [B, D], float, in [0, 1]
// embeddings: [sO, C], float
// offsets: [L + 1], uint32_t
// outputs: [B, L * C], float
// H: base resolution
void hash_encode_forward(const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, at::Tensor outputs, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const at::Tensor per_level_scale, const at::Tensor base_resolution, const bool calc_grad_inputs, at::Tensor dy_dx);
void hash_encode_backward(const at::Tensor grad, const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, at::Tensor grad_embeddings, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const at::Tensor per_level_scale, const at::Tensor base_resolution, const bool calc_grad_inputs, const at::Tensor dy_dx, at::Tensor grad_inputs);

// New hash encoding functions with collision tracking
// Additional parameters:
// track_collisions: bool - whether to enable collision tracking
// grid_indices: [max_examples, L, D], int16 - grid coordinates for each example/level 
// collision_flags: [max_examples, L], bool - collision flags for each example/level
// max_tracking_examples: uint32 - maximum number of examples to track
// current_example_count: [1], uint32 - current number of tracked examples (global counter)
void hash_encode_forward_with_tracking(
    const at::Tensor inputs, 
    const at::Tensor embeddings, 
    const at::Tensor offsets, 
    at::Tensor outputs, 
    const uint32_t B, 
    const uint32_t D, 
    const uint32_t C, 
    const uint32_t L, 
    const at::Tensor per_level_scale, 
    const at::Tensor base_resolution, 
    const bool calc_grad_inputs, 
    at::Tensor dy_dx,
    const bool track_collisions,
    at::Tensor grid_indices,
    at::Tensor collision_flags,
    const uint32_t max_tracking_examples,
    at::Tensor current_example_count
);

#endif