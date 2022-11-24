
#ifndef DALI_OPERATORS_DECODER_NVJPEG_L3_DECODER_H_
#define DALI_OPERATORS_DECODER_NVJPEG_L3_DECODER_H_

#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define shard 64

namespace dali {

void preprocess_l3_decode(uint8_t *output, const uint8_t *input, int in_size, void *dev_input, cudaStream_t stream);

}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_NVJPEG_L3_DECODER_H_

