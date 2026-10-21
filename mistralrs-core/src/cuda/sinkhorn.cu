// Fused sinkhorn mixer for the DeepSeek-V4 hyper-connections.
//
// `logits` and `out` are contiguous `[n, hc, hc]` f32, element `(dst, src)` at
// `dst * hc + src`. Each thread owns one matrix and computes, in order:
//   a softmax over `dst`, plus `eps`;
//   a normalization over `src`;
//   `iters - 1` rounds of a normalization over `dst`, then over `src`.
// A normalization divides by the sum along its axis plus `eps`. This is the
// same sequence `dsv4_hc::sinkhorn_reference` runs as separate tensor ops; `hc`
// is 4 for DeepSeek-V4.1 and GLM-5.3, so one matrix is 16 values.

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#define SINKHORN_MAX_HC 16

__global__ void sinkhorn_f32_kernel(const float *__restrict__ logits,
                                    float *__restrict__ out, int64_t n, int hc,
                                    int iters, float eps) {
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const float *in = logits + idx * hc * hc;
  float *dst_out = out + idx * hc * hc;
  float m[SINKHORN_MAX_HC * SINKHORN_MAX_HC];

  for (int s = 0; s < hc; ++s) {
    float mx = -INFINITY;
    for (int d = 0; d < hc; ++d) {
      mx = fmaxf(mx, in[d * hc + s]);
    }
    float sum = 0.0f;
    for (int d = 0; d < hc; ++d) {
      const float e = expf(in[d * hc + s] - mx);
      m[d * hc + s] = e;
      sum += e;
    }
    for (int d = 0; d < hc; ++d) {
      m[d * hc + s] = m[d * hc + s] / sum + eps;
    }
  }

  for (int round = 0; round < iters; ++round) {
    if (round > 0) {
      for (int s = 0; s < hc; ++s) {
        float sum = 0.0f;
        for (int d = 0; d < hc; ++d) {
          sum += m[d * hc + s];
        }
        const float denom = sum + eps;
        for (int d = 0; d < hc; ++d) {
          m[d * hc + s] /= denom;
        }
      }
    }
    for (int d = 0; d < hc; ++d) {
      float sum = 0.0f;
      for (int s = 0; s < hc; ++s) {
        sum += m[d * hc + s];
      }
      const float denom = sum + eps;
      for (int s = 0; s < hc; ++s) {
        m[d * hc + s] /= denom;
      }
    }
  }

  for (int i = 0; i < hc * hc; ++i) {
    dst_out[i] = m[i];
  }
}

extern "C" int dsv4_sinkhorn_f32(const void *logits, void *out, int64_t n,
                                 int hc, int iters, float eps, int64_t stream) {
  if (hc <= 0 || hc > SINKHORN_MAX_HC || iters < 1 || n < 0) {
    return 1;
  }
  if (n == 0) {
    return 0;
  }
  const int threads = 128;
  const int64_t blocks = (n + threads - 1) / threads;
  sinkhorn_f32_kernel<<<(unsigned int)blocks, threads, 0,
                        reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const float *>(logits), static_cast<float *>(out), n, hc,
      iters, eps);
  return cudaGetLastError() == cudaSuccess ? 0 : 2;
}
