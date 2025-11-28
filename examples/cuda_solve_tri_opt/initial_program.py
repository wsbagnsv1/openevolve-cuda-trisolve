python
"""
CUDA Kernel for Triangular Matrix Solve (solve_tri)

This module contains the CUDA kernel source code for solving triangular systems.
The goal is to optimize this kernel for performance on NVIDIA GPUs.
"""

# EVOLVE-BLOCK-START
cuda_source = r'''
#include "common.cuh"
#include "ggml.h"
#include "solve_tri.cuh"

#define MAX_N_FAST 64
#define MAX_K_FAST 32

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template <int n_template, int k_template>
static __global__ void solve_tri_f32_fast(const float * __restrict__ A,
                                          const float * __restrict__ B,
                                          float * __restrict__ X,
                                          const uint3  ne02,
                                          const size_t nb02,
                                          const size_t nb03,
                                          const size_t nb12,
                                          const size_t nb13,
                                          const size_t nb2,
                                          const size_t nb3,
                                          const int    n_arg,
                                          const int    k_arg) {
    const int n = n_template == 0 ? n_arg : n_template;
    const int k = k_template == 0 ? k_arg : k_template;

    const int batch_idx = blockIdx.x;
    const int lane      = threadIdx.x;
    const int col_idx   = threadIdx.y;

    if (col_idx >= k) {
        return;
    }

    const uint2   i02_i03 = fast_div_modulo(batch_idx, ne02);
    const int64_t i02     = i02_i03.y;
    const int64_t i03     = i02_i03.x;

    const float * const A_batch = (const float *) ((const char *) A + i02 * nb02 + i03 * nb03);
    const float * const B_batch = (const float *) ((const char *) B + i02 * nb12 + i03 * nb13);
    float *             X_batch = (float *) ((char *) X + i02 * nb2 + i03 * nb3);

    __shared__ float sA[MAX_N_FAST * MAX_N_FAST];
    __shared__ float sX[MAX_K_FAST * MAX_N_FAST];

    const int offset = threadIdx.x + threadIdx.y * blockDim.x;

#pragma unroll
    for (int i = 0; i < n * n; i += k * WARP_SIZE) {
        int i0 = i + offset;
        if (i0 < n * n) {
            sA[i0] = A_batch[i0];
        }
    }

    // Warp-wise load B to column-major sX
#pragma unroll 2
    for (int rr = 0; rr < 2; ++rr) {
        int row = rr * WARP_SIZE + lane;
        if (row < n) {
            sX[col_idx * n + row] = B_batch[row * k + col_idx];
        }
    }

    __syncthreads();

    const int half = WARP_SIZE;

    if (n <= half) {
#pragma unroll
        for (int row = 0; row < n; ++row) {
            float sum = 0.0f;
            int j = lane;
            if (j < row) {
                sum += sA[row * n + j] * sX[col_idx * n + j];
            }
            sum = warp_reduce_sum(sum);
            if (lane == 0) {
                const float diag = sA[row * n + row];
                const float b_val = sX[col_idx * n + row];
                const float idiv = 1.0f / diag;
                sX[col_idx * n + row] = fmaf(sum, -idiv, b_val * idiv);
            }
            __syncwarp();
        }
    } else {
        // First half
#pragma unroll
        for (int row = 0; row < half; ++row) {
            float sum = 0.0f;
            int j = lane;
            if (j < row) {
                sum += sA[row * n + j] * sX[col_idx * n + j];
            }
            sum = warp_reduce_sum(sum);
            if (lane == 0) {
                const float diag = sA[row * n + row];
                const float b_val = sX[col_idx * n + row];
                const float idiv = 1.0f / diag;
                sX[col_idx * n + row] = fmaf(sum, -idiv, b_val * idiv);
            }
            __syncwarp();
        }
        // Second half
#pragma unroll
        for (int row = half; row < n; ++row) {
            float sum = 0.0f;
            // First part: all active
            int j = lane;
            sum += sA[row * n + j] * sX[col_idx * n + j];
            // Second part
            j = half + lane;
            if (j < row) {
                sum += sA[row * n + j] * sX[col_idx * n + j];
            }
            sum = warp_reduce_sum(sum);
            if (lane == 0) {
                const float diag = sA[row * n + row];
                const float b_val = sX[col_idx * n + row];
                const float idiv = 1.0f / diag;
                sX[col_idx * n + row] = fmaf(sum, -idiv, b_val * idiv);
            }
            __syncwarp();
        }
    }

    // Warp-wise store from column-major sX to X
#pragma unroll 2
    for (int rr = 0; rr < 2; ++rr) {
        int row = rr * WARP_SIZE + lane;
        if (row < n) {
            X_batch[row * k + col_idx] = sX[col_idx * n + row];
        }
    }
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

// Launcher
static void solve_tri_f32_cuda(const float * A,
                               const float * B,
                               float *       X,
                               int           n,
                               int           k,
                               int64_t       ne02,
                               int64_t       ne03,
                               size_t        nb02,
                               size_t        nb03,
                               size_t        nb12,
                               size_t        nb13,
                               size_t        nb2,
                               size_t        nb3,
                               cudaStream_t  stream) {
    const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
    dim3        threads(WARP_SIZE, k);
    dim3        grid(ne02 * ne03);
    if (n == 64) {
        switch (k) {
            case 32:
                solve_tri_f32_fast<64, 32>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
                break;
            case 16:
                solve_tri_f32_fast<64, 16>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
                break;
            case 14:
                solve_tri_f32_fast<64, 14>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
                break;
            case 12:
                solve_tri_f32_fast<64, 12>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
                break;
            case 10:
                solve_tri_f32_fast<64, 10>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
                break;
            case 8:
                solve_tri_f32_fast<64, 8>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
                break;
            case 6:
                solve_tri_f32_fast<64, 6>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
                break;
            case 4:
                solve_tri_f32_fast<64, 4>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
                break;
            case 2:
                solve_tri_f32_fast<64, 2>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
                break;
            case 1:
                solve_tri_f32_fast<64, 1>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
                break;
            default:
                solve_tri_f32_fast<0, 0>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, n, k);
        }
    } else {
        solve_tri_f32_fast<0, 0>
            <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, n, k);
    }
}

void ggml_cuda_op_solve_tri(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];  // A
    const ggml_tensor * src1 = dst->src[1];  // B

    ggml_is_contiguous(src0);
    ggml_is_contiguous(src1);

    const int64_t n = src0->ne[0];
    const int64_t k = src1->ne[0];

    GGML_ASSERT(n <= 64);
    GGML_ASSERT(k <= 32);

    solve_tri_f32_cuda((const float *) src0->data, (const float *) src1->data, (float *) dst->data, n, k, src0->ne[2],
                       src0->ne[3], src0->nb[2], src0->nb[3], src1->nb[2], src1->nb[3], dst->nb[2], dst->nb[3],
                       ctx.stream());
}
'''
# EVOLVE-BLOCK-END
