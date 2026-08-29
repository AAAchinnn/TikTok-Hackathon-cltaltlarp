// ---------------------------------------------------------------------------
// current.cu -- the kernel under iteration.
//
// v0: a tiled, online-softmax (FlashAttention-style) fused attention forward.
// It fuses QK^T, scaling, causal + key-padding masking, softmax and PV into a
// single kernel, so the [B, H, S, S] score matrix is never written to global
// memory. That is the one structural win over the reference implementation,
// which materializes it (and a second fp32 copy of it) per layer.
//
// This is deliberately a SIMPLE correct baseline, not a fast one. It leaves
// obvious headroom on purpose -- see ../MUTATION_MENU.md, whose entries map
// onto the numbered comments below. Known-suboptimal by construction:
//
//   (M1) K and V are read straight from global on every key tile; no shared
//        memory staging and no double buffering.
//   (M2) One score per warp-reduction: 5 shuffles of overhead per dot product.
//   (M3) No tensor cores. Everything runs on the FP32 pipe even in fp16.
//   (M4) The PV accumulation parallelizes over head_dim only, so with D=64 and
//        128 threads half the block idles.
//   (M5) The softmax rescale runs on BLOCK_M=16 threads while 112 idle.
//   (M6) Scalar (non-vectorized) global loads: no float4 / half2 packing.
//
// Interface (matches harness/common.py's CudaFusedTransformer):
//   fused_attention(q, k, v, valid_mask, causal, scale) -> out
//     q, k, v     [B, H, S, D] contiguous, float32 / float16 / bfloat16
//     valid_mask  [B, S] bool, or an empty tensor for "no padding"
//     causal      bool
//     scale       float (1/sqrt(head_dim))
//     out         [B, H, S, D], same dtype as q
//
// Correctness contract: bit-comparable-enough to the reference under
// abs<=2e-3 OR rel<=2e-2. Softmax accumulates in fp32 regardless of the input
// dtype, matching the reference's `torch.softmax(scores.float(), ...)`.
// ---------------------------------------------------------------------------

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cfloat>

namespace {

constexpr int kBlockM  = 16;   // query rows per block
constexpr int kBlockN  = 64;   // keys per tile
constexpr int kThreads = 128;  // 4 warps
constexpr int kMaxHeadDim = 128;

// Shared-memory budget, floats:
//   sq[BLOCK_M*D] + sacc[BLOCK_M*D] + sp[BLOCK_M*BLOCK_N] + 3*BLOCK_M
// D=64  ->  ( 1024 + 1024 + 1024 + 48) * 4 =  12.6 KB
// D=128 ->  ( 2048 + 2048 + 1024 + 48) * 4 =  20.7 KB
// Both fit the 48 KB default limit, so no cudaFuncSetAttribute opt-in.
inline size_t smem_bytes(int head_dim) {
  return sizeof(float) *
         (size_t)(2 * kBlockM * head_dim + kBlockM * kBlockN + 3 * kBlockM);
}

template <typename scalar_t>
__global__ void fused_attention_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const bool* __restrict__ valid_mask,  // [B, S] or nullptr
    scalar_t* __restrict__ out,
    const int batch,
    const int heads,
    const int seq_len,
    const int head_dim,
    const bool causal,
    const float scale) {
  const int tile_index = blockIdx.x;
  const int head       = blockIdx.y;
  const int batch_id   = blockIdx.z;
  const int row_base   = tile_index * kBlockM;

  const int tid    = threadIdx.x;
  const int lane   = tid & 31;
  const int warp   = tid >> 5;
  const int nwarps = kThreads >> 5;

  extern __shared__ float smem[];
  float* s_q    = smem;                                 // [kBlockM][D]
  float* s_acc  = s_q + kBlockM * head_dim;             // [kBlockM][D]
  float* s_prob = s_acc + kBlockM * head_dim;           // [kBlockM][kBlockN]
  float* s_max  = s_prob + kBlockM * kBlockN;           // [kBlockM] running max
  float* s_sum  = s_max + kBlockM;                      // [kBlockM] running denom
  float* s_corr = s_sum + kBlockM;                      // [kBlockM] rescale factor

  // Base offset of this (batch, head) slice. 64-bit: B*H*S*D overflows int32
  // at the larger shapes (e.g. 2*16*2048*64 is fine, but layer-stacked
  // indexing is not worth the risk).
  const long long qkv_base =
      ((long long)batch_id * heads + head) * (long long)seq_len * head_dim;

  // ---- load Q tile into shared, zero the accumulator (M6: scalar loads) ----
  for (int i = tid; i < kBlockM * head_dim; i += kThreads) {
    const int r = i / head_dim;
    const int d = i - r * head_dim;
    const int global_row = row_base + r;
    s_q[i] = (global_row < seq_len)
                 ? static_cast<float>(q[qkv_base + (long long)global_row * head_dim + d])
                 : 0.0f;
    s_acc[i] = 0.0f;
  }
  for (int r = tid; r < kBlockM; r += kThreads) {
    s_max[r] = -FLT_MAX;
    s_sum[r] = 0.0f;
    s_corr[r] = 0.0f;
  }
  __syncthreads();

  // With a causal mask no query in this tile can attend past the tile's last
  // row, so whole key tiles beyond that are skipped rather than masked away.
  int key_end = seq_len;
  if (causal) {
    key_end = min(seq_len, row_base + kBlockM);
  }

  for (int n0 = 0; n0 < key_end; n0 += kBlockN) {
    // ---- scores: one (row, key) dot product per warp iteration ----------
    // (M2) Lanes split the head_dim reduction and finish with a shuffle tree.
    // (M1) K is re-read from global here on every tile.
    for (int idx = warp; idx < kBlockM * kBlockN; idx += nwarps) {
      const int r = idx / kBlockN;
      const int j = idx - r * kBlockN;
      const int global_row = row_base + r;
      const int global_key = n0 + j;

      float dot = 0.0f;
      if (global_row < seq_len && global_key < seq_len) {
        const long long k_off = qkv_base + (long long)global_key * head_dim;
        for (int d = lane; d < head_dim; d += 32) {
          dot += s_q[r * head_dim + d] * static_cast<float>(k[k_off + d]);
        }
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_down_sync(0xffffffffu, dot, offset);
      }

      if (lane == 0) {
        bool active = (global_row < seq_len) && (global_key < seq_len);
        if (active && causal && global_key > global_row) {
          active = false;
        }
        if (active && valid_mask != nullptr &&
            !valid_mask[(long long)batch_id * seq_len + global_key]) {
          active = false;
        }
        // -FLT_MAX rather than -inf: it survives the exp() below as a clean
        // zero without risking inf-inf NaNs in the rescale.
        s_prob[r * kBlockN + j] = active ? dot * scale : -FLT_MAX;
      }
    }
    __syncthreads();

    // ---- online softmax rescale, one row per thread (M5) ----------------
    if (tid < kBlockM) {
      const int r = tid;
      const float prev_max = s_max[r];

      float tile_max = -FLT_MAX;
      for (int j = 0; j < kBlockN; ++j) {
        tile_max = fmaxf(tile_max, s_prob[r * kBlockN + j]);
      }
      const float new_max = fmaxf(prev_max, tile_max);

      // No finite score seen yet, here or previously: emit zeros and keep the
      // denominator at 0 so the epilogue writes 0 for a fully-masked row.
      if (new_max == -FLT_MAX) {
        for (int j = 0; j < kBlockN; ++j) {
          s_prob[r * kBlockN + j] = 0.0f;
        }
        s_corr[r] = 1.0f;
      } else {
        const float correction =
            (prev_max == -FLT_MAX) ? 0.0f : __expf(prev_max - new_max);
        float tile_sum = 0.0f;
        for (int j = 0; j < kBlockN; ++j) {
          const float score = s_prob[r * kBlockN + j];
          const float p = (score == -FLT_MAX) ? 0.0f : __expf(score - new_max);
          s_prob[r * kBlockN + j] = p;
          tile_sum += p;
        }
        s_max[r] = new_max;
        s_sum[r] = s_sum[r] * correction + tile_sum;
        s_corr[r] = correction;
      }
    }
    __syncthreads();

    // ---- accumulate P @ V, one head_dim column per thread (M4) ----------
    // V is loaded once per (key, column) and reused across all kBlockM rows.
    for (int d = tid; d < head_dim; d += kThreads) {
      for (int r = 0; r < kBlockM; ++r) {
        s_acc[r * head_dim + d] *= s_corr[r];
      }
      for (int j = 0; j < kBlockN; ++j) {
        const int global_key = n0 + j;
        if (global_key >= seq_len) {
          break;
        }
        const float value =
            static_cast<float>(v[qkv_base + (long long)global_key * head_dim + d]);
        for (int r = 0; r < kBlockM; ++r) {
          s_acc[r * head_dim + d] += s_prob[r * kBlockN + j] * value;
        }
      }
    }
    __syncthreads();
  }

  // ---- epilogue: normalize by the softmax denominator and store ----------
  for (int i = tid; i < kBlockM * head_dim; i += kThreads) {
    const int r = i / head_dim;
    const int d = i - r * head_dim;
    const int global_row = row_base + r;
    if (global_row >= seq_len) {
      continue;
    }
    const float denom = s_sum[r];
    // denom == 0 means every key was masked for this row. The reference zeroes
    // those positions via masked_fill, so we match by writing 0.
    const float result = (denom > 0.0f) ? (s_acc[i] / denom) : 0.0f;
    out[qkv_base + (long long)global_row * head_dim + d] =
        static_cast<scalar_t>(result);
  }
}

}  // namespace

torch::Tensor fused_attention(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor valid_mask,
    bool causal,
    double scale) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q/k/v must be CUDA tensors");
  TORCH_CHECK(q.dim() == 4, "expected q of shape [B, H, S, D], got ", q.sizes());
  TORCH_CHECK(k.sizes() == q.sizes() && v.sizes() == q.sizes(),
              "q/k/v must have identical shapes");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
              "q/k/v must share a dtype");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
              "q/k/v must be contiguous");

  const int batch    = q.size(0);
  const int heads    = q.size(1);
  const int seq_len  = q.size(2);
  const int head_dim = q.size(3);

  // Fail loudly rather than silently falling back to PyTorch: a silent
  // fallback would report a "speedup" that is really just the torch path.
  TORCH_CHECK(head_dim <= kMaxHeadDim,
              "head_dim ", head_dim, " exceeds kMaxHeadDim ", kMaxHeadDim,
              "; raise the constant and re-check the shared-memory budget");

  const bool* mask_ptr = nullptr;
  if (valid_mask.defined() && valid_mask.numel() > 0) {
    TORCH_CHECK(valid_mask.is_cuda(), "valid_mask must be a CUDA tensor");
    TORCH_CHECK(valid_mask.scalar_type() == torch::kBool, "valid_mask must be bool");
    TORCH_CHECK(valid_mask.dim() == 2 && valid_mask.size(0) == batch &&
                    valid_mask.size(1) == seq_len,
                "valid_mask must be [B, S], got ", valid_mask.sizes());
    TORCH_CHECK(valid_mask.is_contiguous(), "valid_mask must be contiguous");
    mask_ptr = valid_mask.data_ptr<bool>();
  }

  const at::cuda::CUDAGuard guard(q.device());
  auto out = torch::empty_like(q);

  const dim3 grid((seq_len + kBlockM - 1) / kBlockM, heads, batch);
  const dim3 block(kThreads);
  const size_t shared = smem_bytes(head_dim);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, q.scalar_type(), "fused_attention", [&] {
        fused_attention_kernel<scalar_t><<<grid, block, shared, stream>>>(
            q.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            mask_ptr,
            out.data_ptr<scalar_t>(),
            batch, heads, seq_len, head_dim, causal,
            static_cast<float>(scale));
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_attention", &fused_attention,
        "Fused multi-head attention forward (online softmax, causal + padding mask)",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("valid_mask"),
        py::arg("causal"), py::arg("scale"));
}
