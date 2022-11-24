
#include "dali/operators/decoder/nvjpeg/l3_decoder.h"
#include "dali/core/error_handling.h"
#include "dali/core/byte_io.h"


namespace dali {

namespace {


constexpr int kOffsetWidth = sizeof(int);
constexpr int kOffsetHeight = kOffsetWidth + sizeof(int);

/*
constexpr int kOffsetShard = kOffsetHeight + sizeof(int);

char ReadShard(const uint8_t *data) {
  return ReadValueLE<char>(data + kOffsetShard);
}
*/

int ReadHeight(const uint8_t *data) {
  return ReadValueLE<int>(data + kOffsetHeight);
}

int ReadWidth(const uint8_t *data) {
  return ReadValueLE<int>(data + kOffsetWidth);
}

}  // namespace


__device__ inline char paeth_filter(unsigned char a, unsigned char b, unsigned char c) {

    int p = (int)(a + c - b);

    if (a == 0) {
        int pb = abs(p - b);
        int pc = abs(p - c);

        if (pb <= pc) return b;
        else          return c;
    } else if (c == 0) {
        int pa = abs(p - a);
        int pb = abs(p - b);

        if (pa <= pb) return a;
        else          return b;

    } else {
        int pa = abs(p - a);
        int pb = abs(p - b);
        int pc = abs(p - c);

        if (pa <= pb && pa <= pc) return a;
        else if (pa <= pc)        return b;
        else                      return c;
    }

    return -1;
}


__device__ int bd_loop_dec(unsigned char* bd, char* paeth) {
    int nbit_entry = (bd[0] & 0x0F);
    char base = ((bd[0] & 0xF0) >> 4) + ((bd[1] & 0x0F) << 4);

    int sbit = 12 + threadIdx.x * nbit_entry;
    int ebit = 12 + (threadIdx.x + 1) * nbit_entry - 1;

    int sbound = floorf(sbit / 8);
    int ebound = floorf(ebit / 8);

    int sstart = sbit - sbound * 8;
    int estart = ebit - ebound * 8;

    if (sbound == ebound) {
        unsigned char flag = 0;
        for (int j = 0; j < nbit_entry; j++) {
            flag += (1 << j);
        }
        flag = flag << sstart;
        paeth[threadIdx.x] = ((bd[sbound] & flag) >> sstart) + base;
    } else {
        int nbit_entry_a = 8 - sstart;
        int nbit_entry_b = estart + 1;

        unsigned char flag_a = 0;
        for (int j = 0; j < nbit_entry_a; j++) {
            flag_a += ((unsigned char)128 >> j);
        }

        unsigned char flag_b = 0;
        for (int j = 0; j < nbit_entry_b; j++) {
            flag_b += (1 << j);
        }

        paeth[threadIdx.x] = ((bd[sbound] & flag_a) >> sstart) + ((bd[ebound] & flag_b) << nbit_entry_a) + base;
    }

    return (int)roundf((12 + nbit_entry * 64) / 8 + 0.5);
}


__device__ void paeth_loop_dec(char* paeth, int i, int idx, unsigned char* unpack, int wd) {
    if (i == 0) {
        unpack[idx + 3 * threadIdx.x] = paeth[threadIdx.x];
    } else {
        if (threadIdx.x == 0) {
            unpack[idx + 3 * threadIdx.x] = paeth[threadIdx.x] + paeth_filter(0,
                                                                        unpack[idx + 3 * (threadIdx.x - wd)],
                                                                        unpack[idx + 3 * (threadIdx.x - wd + 1)]);
        } else if (threadIdx.x == (shard - 1)) {
            unpack[idx + 3 * threadIdx.x] = paeth[threadIdx.x] + paeth_filter(unpack[idx + 3 * (threadIdx.x - wd - 1)],
                                                                        unpack[idx + 3 * (threadIdx.x - wd)],
                                                                        0);
        } else {
            unpack[idx + 3 * threadIdx.x] = paeth[threadIdx.x] + paeth_filter(unpack[idx + 3 * (threadIdx.x - wd - 1)],
                                                                        unpack[idx + 3 * (threadIdx.x - wd)],
                                                                        unpack[idx + 3 * (threadIdx.x - wd + 1)]);
        }
    }
}


__global__ void decoder(unsigned char* input, int* offset_r, int* offset_g, int* offset_b,  unsigned char* output,
                        int wd, int ht, int num_wd, int num_ht) {
    int offset_idx = blockIdx.y * num_wd + blockIdx.x;

    int header_size = sizeof(short) * (num_wd * num_ht * 3) + 13;

    int pt_offset_r = offset_r[offset_idx] + header_size;
    int pt_offset_g = offset_g[offset_idx] + header_size + offset_r[num_wd * num_ht];
    int pt_offset_b = offset_b[offset_idx] + header_size + offset_r[num_wd * num_ht] + offset_g[num_wd * num_ht];

    char paeth_r[shard], paeth_g[shard], paeth_b[shard];

    for (int i = 0; i < shard; i++) {
        memset(paeth_r, 0, sizeof(char) * shard);
        memset(paeth_g, 0, sizeof(char) * shard);
        memset(paeth_b, 0, sizeof(char) * shard);

        int row_off_r = bd_loop_dec(&input[pt_offset_r], paeth_r);
        int row_off_g = bd_loop_dec(&input[pt_offset_g], paeth_g);
        int row_off_b = bd_loop_dec(&input[pt_offset_b], paeth_b);

        pt_offset_r += row_off_r;
        pt_offset_g += row_off_g;
        pt_offset_b += row_off_b;

        int idx = (blockIdx.y * shard + i) * (wd * 3) + (blockIdx.x * (shard * 3));

        paeth_loop_dec(paeth_r, i, idx, output, wd);
        paeth_loop_dec(paeth_g, i, idx + 1, output, wd);
        paeth_loop_dec(paeth_b, i, idx + 2, output, wd);

        __syncthreads();
    }
}


void preprocess_l3_decode(uint8_t *output, const uint8_t *input, int in_size, void *dev_input, cudaStream_t stream) {

  int wd = ReadWidth(input);
  int ht = ReadHeight(input);
  // int shard = (int)ReadShard(input);

  int num_wd = (int)(wd / shard);
  int num_ht = (int)(ht / shard);

  unsigned short *offset_cpu_in_r, *offset_cpu_in_g, *offset_cpu_in_b;
  int *offset_cpu_out_r, *offset_cpu_out_g, *offset_cpu_out_b;
  void *offset_dev_out_r, *offset_dev_out_g, *offset_dev_out_b;

  size_t size_offset = sizeof(short) * num_wd * num_ht;
  size_t size_offset_int = sizeof(int) * (num_wd * num_ht + 1);

  offset_cpu_in_r = (unsigned short *)malloc(size_offset);
  offset_cpu_in_g = (unsigned short *)malloc(size_offset);
  offset_cpu_in_b = (unsigned short *)malloc(size_offset);

  offset_cpu_out_r = (int *)malloc(size_offset_int);
  offset_cpu_out_g = (int *)malloc(size_offset_int);
  offset_cpu_out_b = (int *)malloc(size_offset_int);

  cudaMalloc(&offset_dev_out_r, size_offset_int);
  cudaMalloc(&offset_dev_out_g, size_offset_int);
  cudaMalloc(&offset_dev_out_b, size_offset_int);

  memcpy((void *)offset_cpu_in_r, (void *)&input[13], size_offset);
  memcpy((void *)offset_cpu_in_g, (void *)&input[13 + (num_wd * num_ht) * 2], size_offset);
  memcpy((void *)offset_cpu_in_b, (void *)&input[13 + (num_wd * num_ht) * 4], size_offset);

  offset_cpu_out_r[0] = 0;
  offset_cpu_out_g[0] = 0;
  offset_cpu_out_b[0] = 0;
  for (int i = 0; i < num_wd * num_ht + 1; i++) {
      offset_cpu_out_r[i + 1] = offset_cpu_in_r[i] + offset_cpu_out_r[i];
      offset_cpu_out_g[i + 1] = offset_cpu_in_g[i] + offset_cpu_out_g[i];
      offset_cpu_out_b[i + 1] = offset_cpu_in_b[i] + offset_cpu_out_b[i];
  }

  free(offset_cpu_in_r);
  free(offset_cpu_in_g);
  free(offset_cpu_in_b);

  cudaMemcpyAsync(offset_dev_out_r, (void *)offset_cpu_out_r, size_offset_int, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(offset_dev_out_g, (void *)offset_cpu_out_g, size_offset_int, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(offset_dev_out_b, (void *)offset_cpu_out_b, size_offset_int, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(dev_input, (void *)input, in_size, cudaMemcpyHostToDevice, stream);

  void *dev_out;
  cudaMalloc(&dev_out, wd * ht * 3);

  dim3 blocks(num_wd, num_ht);
  dim3 threads(shard, 1);

  decoder<<<blocks, threads, 0, stream>>>((unsigned char *)dev_input,
                                          (int *)offset_dev_out_r, (int *)offset_dev_out_g, (int *)offset_dev_out_b,
                                          (unsigned char *)dev_out,
                                          wd, ht, num_wd, num_ht);

  cudaMemcpyAsync((void *)output, dev_out, wd * ht * 3, cudaMemcpyDeviceToHost, stream);
  cudaDeviceSynchronize();

  cudaFree(dev_out);
}


}  // namespace dali

