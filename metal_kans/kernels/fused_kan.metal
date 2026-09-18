#include <metal_stdlib>
using namespace metal;

// ==============================================================================
// 1. CHEBYKAN FUSED TILED KERNEL (THREADGROUP SHARED MEMORY SRAM + 2x2 REGISTER BLOCKING)
// Fuses Chebyshev polynomial basis evaluation + Matrix Multiply-Accumulate in 1 pass.
// Avoids writing the intermediate B x (D_in * K) tensor to global VRAM.
// ==============================================================================

#define TILE_M 32
#define TILE_N 32
#define TILE_K 8

kernel void kan_cheby_tiled_deg4(
    device const float*  X         [[buffer(0)]], // [B, D_in]
    device const float*  W_cheby   [[buffer(1)]], // [D_out, D_in * 4]
    device const float*  W_base    [[buffer(2)]], // [D_out, D_in]
    device const float*  bias      [[buffer(3)]], // [D_out]
    device float*        Y         [[buffer(4)]], // [B, D_out]
    constant uint&       B         [[buffer(5)]],
    constant uint&       D_in      [[buffer(6)]],
    constant uint&       D_out     [[buffer(7)]],
    constant uint&       has_base  [[buffer(8)]],
    constant uint&       has_bias  [[buffer(9)]],
    uint2                tg_pos    [[threadgroup_position_in_grid]],
    uint2                t_pos     [[thread_position_in_threadgroup]]
) {
    uint ty = t_pos.y; // 0..15
    uint tx = t_pos.x; // 0..15

    uint row0 = tg_pos.y * TILE_M + ty * 2;
    uint row1 = row0 + 1;
    uint col0 = tg_pos.x * TILE_N + tx * 2;
    uint col1 = col0 + 1;

    float acc00 = (has_bias && col0 < D_out) ? bias[col0] : 0.0f;
    float acc01 = (has_bias && col1 < D_out) ? bias[col1] : 0.0f;
    float acc10 = (has_bias && col0 < D_out) ? bias[col0] : 0.0f;
    float acc11 = (has_bias && col1 < D_out) ? bias[col1] : 0.0f;

    threadgroup float  s_X[TILE_M][TILE_K];
    threadgroup float4 s_W[TILE_N][TILE_K];
    threadgroup float  s_Wb[TILE_N][TILE_K];

    uint num_k_tiles = (D_in + TILE_K - 1) / TILE_K;
    uint tid_flat = ty * 16 + tx; // 256 threads in threadgroup

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        // Collaborative load of X tile: 32 rows x 8 cols = 256 elements
        uint load_r = tid_flat / TILE_K;
        uint load_c = tid_flat % TILE_K;
        uint global_r = tg_pos.y * TILE_M + load_r;
        uint global_c = kt * TILE_K + load_c;
        s_X[load_r][load_c] = (global_r < B && global_c < D_in) ? X[global_r * D_in + global_c] : 0.0f;

        // Collaborative load of W tile: 32 rows x 8 cols = 256 float4 elements
        uint w_row = tid_flat / TILE_K;
        uint w_col = tid_flat % TILE_K;
        uint global_w_row = tg_pos.x * TILE_N + w_row;
        uint global_w_col = kt * TILE_K + w_col;
        if (global_w_row < D_out && global_w_col < D_in) {
            device const float4* w_ptr = (device const float4*)(W_cheby + (global_w_row * D_in + global_w_col) * 4);
            s_W[w_row][w_col] = *w_ptr;
            if (has_base) s_Wb[w_row][w_col] = W_base[global_w_row * D_in + global_w_col];
        } else {
            s_W[w_row][w_col] = float4(0.0f);
            if (has_base) s_Wb[w_row][w_col] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute 2x2 output tiles from fast on-chip SRAM
        for (uint k = 0; k < TILE_K; k++) {
            float x_val0 = s_X[ty * 2 + 0][k];
            float x_val1 = s_X[ty * 2 + 1][k];

            float c0 = clamp(x_val0, -1.0f, 1.0f);
            float c1 = clamp(x_val1, -1.0f, 1.0f);

            // Vectorized Chebyshev basis degree 4
            float4 b0 = float4(1.0f, c0, fma(2.0f * c0, c0, -1.0f), c0 * fma(4.0f * c0, c0, -3.0f));
            float4 b1 = float4(1.0f, c1, fma(2.0f * c1, c1, -1.0f), c1 * fma(4.0f * c1, c1, -3.0f));

            float4 w0 = s_W[tx * 2 + 0][k];
            float4 w1 = s_W[tx * 2 + 1][k];

            acc00 += dot(b0, w0);
            acc01 += dot(b0, w1);
            acc10 += dot(b1, w0);
            acc11 += dot(b1, w1);

            if (has_base) {
                float s0 = x_val0 / (1.0f + exp(-x_val0));
                float s1 = x_val1 / (1.0f + exp(-x_val1));
                acc00 = fma(s0, s_Wb[tx * 2 + 0][k], acc00);
                acc01 = fma(s0, s_Wb[tx * 2 + 1][k], acc01);
                acc10 = fma(s1, s_Wb[tx * 2 + 0][k], acc10);
                acc11 = fma(s1, s_Wb[tx * 2 + 1][k], acc11);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row0 < B && col0 < D_out) Y[row0 * D_out + col0] = acc00;
    if (row0 < B && col1 < D_out) Y[row0 * D_out + col1] = acc01;
    if (row1 < B && col0 < D_out) Y[row1 * D_out + col0] = acc10;
    if (row1 < B && col1 < D_out) Y[row1 * D_out + col1] = acc11;
}

// ==============================================================================
// 2. FASTKAN FUSED TILED KERNEL (GAUSSIAN RBF)
// ==============================================================================

kernel void kan_fastkan_tiled(
    device const float*  X         [[buffer(0)]],
    device const float*  W_rbf     [[buffer(1)]], // [D_out, D_in * num_centers]
    device const float*  W_base    [[buffer(2)]],
    device const float*  grid      [[buffer(3)]], // [num_centers]
    device const float*  bias      [[buffer(4)]],
    device float*        Y         [[buffer(5)]],
    constant uint&       B         [[buffer(6)]],
    constant uint&       D_in      [[buffer(7)]],
    constant uint&       D_out     [[buffer(8)]],
    constant uint&       num_centers [[buffer(9)]],
    constant float&      inv_denom [[buffer(10)]],
    constant uint&       has_base  [[buffer(11)]],
    constant uint&       has_bias  [[buffer(12)]],
    uint2                tg_pos    [[threadgroup_position_in_grid]],
    uint2                t_pos     [[thread_position_in_threadgroup]]
) {
    uint ty = t_pos.y;
    uint tx = t_pos.x;

    uint row0 = tg_pos.y * TILE_M + ty * 2;
    uint row1 = row0 + 1;
    uint col0 = tg_pos.x * TILE_N + tx * 2;
    uint col1 = col0 + 1;

    float acc00 = (has_bias && col0 < D_out) ? bias[col0] : 0.0f;
    float acc01 = (has_bias && col1 < D_out) ? bias[col1] : 0.0f;
    float acc10 = (has_bias && col0 < D_out) ? bias[col0] : 0.0f;
    float acc11 = (has_bias && col1 < D_out) ? bias[col1] : 0.0f;

    threadgroup float s_X[TILE_M][TILE_K];
    uint tid_flat = ty * 16 + tx;
    uint num_k_tiles = (D_in + TILE_K - 1) / TILE_K;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint load_r = tid_flat / TILE_K;
        uint load_c = tid_flat % TILE_K;
        uint global_r = tg_pos.y * TILE_M + load_r;
        uint global_c = kt * TILE_K + load_c;
        s_X[load_r][load_c] = (global_r < B && global_c < D_in) ? X[global_r * D_in + global_c] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_K; k++) {
            uint curr_din = kt * TILE_K + k;
            if (curr_din >= D_in) break;

            float x0 = s_X[ty * 2 + 0][k];
            float x1 = s_X[ty * 2 + 1][k];

            if (has_base) {
                float s0 = x0 / (1.0f + exp(-x0));
                float s1 = x1 / (1.0f + exp(-x1));
                if (col0 < D_out) {
                    float wb0 = W_base[col0 * D_in + curr_din];
                    acc00 = fma(s0, wb0, acc00);
                    acc10 = fma(s1, wb0, acc10);
                }
                if (col1 < D_out) {
                    float wb1 = W_base[col1 * D_in + curr_din];
                    acc01 = fma(s0, wb1, acc01);
                    acc11 = fma(s1, wb1, acc11);
                }
            }

            uint w_offset0 = col0 * (D_in * num_centers) + curr_din * num_centers;
            uint w_offset1 = col1 * (D_in * num_centers) + curr_din * num_centers;

            for (uint c = 0; c < num_centers; c++) {
                float mu = grid[c];
                float d0 = x0 - mu;
                float d1 = x1 - mu;
                float rbf0 = exp(-(d0 * d0) * inv_denom);
                float rbf1 = exp(-(d1 * d1) * inv_denom);

                if (col0 < D_out) {
                    float w = W_rbf[w_offset0 + c];
                    acc00 = fma(rbf0, w, acc00);
                    acc10 = fma(rbf1, w, acc10);
                }
                if (col1 < D_out) {
                    float w = W_rbf[w_offset1 + c];
                    acc01 = fma(rbf0, w, acc01);
                    acc11 = fma(rbf1, w, acc11);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row0 < B && col0 < D_out) Y[row0 * D_out + col0] = acc00;
    if (row0 < B && col1 < D_out) Y[row0 * D_out + col1] = acc01;
    if (row1 < B && col0 < D_out) Y[row1 * D_out + col0] = acc10;
    if (row1 < B && col1 < D_out) Y[row1 * D_out + col1] = acc11;
}

// ==============================================================================
// 3. RELUKAN FUSED TILED KERNEL (TENT BASIS)
// ==============================================================================

kernel void kan_relu_tiled(
    device const float*  X         [[buffer(0)]],
    device const float*  W_relu    [[buffer(1)]],
    device const float*  W_base    [[buffer(2)]],
    device const float*  grid      [[buffer(3)]],
    device const float*  bias      [[buffer(4)]],
    device float*        Y         [[buffer(5)]],
    constant uint&       B         [[buffer(6)]],
    constant uint&       D_in      [[buffer(7)]],
    constant uint&       D_out     [[buffer(8)]],
    constant uint&       num_grids [[buffer(9)]],
    constant float&      inv_h     [[buffer(10)]],
    constant uint&       has_base  [[buffer(11)]],
    constant uint&       has_bias  [[buffer(12)]],
    uint2                tg_pos    [[threadgroup_position_in_grid]],
    uint2                t_pos     [[thread_position_in_threadgroup]]
) {
    uint ty = t_pos.y;
    uint tx = t_pos.x;

    uint row0 = tg_pos.y * TILE_M + ty * 2;
    uint row1 = row0 + 1;
    uint col0 = tg_pos.x * TILE_N + tx * 2;
    uint col1 = col0 + 1;

    float acc00 = (has_bias && col0 < D_out) ? bias[col0] : 0.0f;
    float acc01 = (has_bias && col1 < D_out) ? bias[col1] : 0.0f;
    float acc10 = (has_bias && col0 < D_out) ? bias[col0] : 0.0f;
    float acc11 = (has_bias && col1 < D_out) ? bias[col1] : 0.0f;

    threadgroup float s_X[TILE_M][TILE_K];
    uint tid_flat = ty * 16 + tx;
    uint num_k_tiles = (D_in + TILE_K - 1) / TILE_K;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint load_r = tid_flat / TILE_K;
        uint load_c = tid_flat % TILE_K;
        uint global_r = tg_pos.y * TILE_M + load_r;
        uint global_c = kt * TILE_K + load_c;
        s_X[load_r][load_c] = (global_r < B && global_c < D_in) ? X[global_r * D_in + global_c] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_K; k++) {
            uint curr_din = kt * TILE_K + k;
            if (curr_din >= D_in) break;

            float x0 = s_X[ty * 2 + 0][k];
            float x1 = s_X[ty * 2 + 1][k];

            if (has_base) {
                float s0 = x0 / (1.0f + exp(-x0));
                float s1 = x1 / (1.0f + exp(-x1));
                if (col0 < D_out) {
                    float wb0 = W_base[col0 * D_in + curr_din];
                    acc00 = fma(s0, wb0, acc00);
                    acc10 = fma(s1, wb0, acc10);
                }
                if (col1 < D_out) {
                    float wb1 = W_base[col1 * D_in + curr_din];
                    acc01 = fma(s0, wb1, acc01);
                    acc11 = fma(s1, wb1, acc11);
                }
            }

            uint w_offset0 = col0 * (D_in * num_grids) + curr_din * num_grids;
            uint w_offset1 = col1 * (D_in * num_grids) + curr_din * num_grids;

            for (uint g = 0; g < num_grids; g++) {
                float mu = grid[g];
                float t0 = max(0.0f, 1.0f - fabs(x0 - mu) * inv_h);
                float t1 = max(0.0f, 1.0f - fabs(x1 - mu) * inv_h);

                if (col0 < D_out) {
                    float w = W_relu[w_offset0 + g];
                    acc00 = fma(t0, w, acc00);
                    acc10 = fma(t1, w, acc10);
                }
                if (col1 < D_out) {
                    float w = W_relu[w_offset1 + g];
                    acc01 = fma(t0, w, acc01);
                    acc11 = fma(t1, w, acc11);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row0 < B && col0 < D_out) Y[row0 * D_out + col0] = acc00;
    if (row0 < B && col1 < D_out) Y[row0 * D_out + col1] = acc01;
    if (row1 < B && col0 < D_out) Y[row1 * D_out + col0] = acc10;
    if (row1 < B && col1 < D_out) Y[row1 * D_out + col1] = acc11;
}

// ==============================================================================
// 4. WAVKAN FUSED TILED KERNEL (MEXICAN HAT / MORLET / DOG WAVELETS)
// ==============================================================================

kernel void kan_wavkan_tiled(
    device const float*  X            [[buffer(0)]],
    device const float*  W_wav        [[buffer(1)]],
    device const float*  W_base       [[buffer(2)]],
    device const float*  translation  [[buffer(3)]],
    device const float*  scale        [[buffer(4)]],
    device const float*  bias         [[buffer(5)]],
    device float*        Y            [[buffer(6)]],
    constant uint&       B            [[buffer(7)]],
    constant uint&       D_in         [[buffer(8)]],
    constant uint&       D_out        [[buffer(9)]],
    constant uint&       num_wavelets [[buffer(10)]],
    constant uint&       wavelet_type [[buffer(11)]],
    constant uint&       has_base     [[buffer(12)]],
    constant uint&       has_bias     [[buffer(13)]],
    uint2                tg_pos       [[threadgroup_position_in_grid]],
    uint2                t_pos        [[thread_position_in_threadgroup]]
) {
    uint ty = t_pos.y;
    uint tx = t_pos.x;

    uint row0 = tg_pos.y * TILE_M + ty * 2;
    uint row1 = row0 + 1;
    uint col0 = tg_pos.x * TILE_N + tx * 2;
    uint col1 = col0 + 1;

    float acc00 = (has_bias && col0 < D_out) ? bias[col0] : 0.0f;
    float acc01 = (has_bias && col1 < D_out) ? bias[col1] : 0.0f;
    float acc10 = (has_bias && col0 < D_out) ? bias[col0] : 0.0f;
    float acc11 = (has_bias && col1 < D_out) ? bias[col1] : 0.0f;

    threadgroup float s_X[TILE_M][TILE_K];
    uint tid_flat = ty * 16 + tx;
    uint num_k_tiles = (D_in + TILE_K - 1) / TILE_K;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint load_r = tid_flat / TILE_K;
        uint load_c = tid_flat % TILE_K;
        uint global_r = tg_pos.y * TILE_M + load_r;
        uint global_c = kt * TILE_K + load_c;
        s_X[load_r][load_c] = (global_r < B && global_c < D_in) ? X[global_r * D_in + global_c] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_K; k++) {
            uint curr_din = kt * TILE_K + k;
            if (curr_din >= D_in) break;

            float x0 = s_X[ty * 2 + 0][k];
            float x1 = s_X[ty * 2 + 1][k];

            if (has_base) {
                float s0 = x0 / (1.0f + exp(-x0));
                float s1 = x1 / (1.0f + exp(-x1));
                if (col0 < D_out) {
                    float wb0 = W_base[col0 * D_in + curr_din];
                    acc00 = fma(s0, wb0, acc00);
                    acc10 = fma(s1, wb0, acc10);
                }
                if (col1 < D_out) {
                    float wb1 = W_base[col1 * D_in + curr_din];
                    acc01 = fma(s0, wb1, acc01);
                    acc11 = fma(s1, wb1, acc11);
                }
            }

            uint w_offset0 = col0 * (D_in * num_wavelets) + curr_din * num_wavelets;
            uint w_offset1 = col1 * (D_in * num_wavelets) + curr_din * num_wavelets;
            uint param_offset = curr_din * num_wavelets;

            for (uint w = 0; w < num_wavelets; w++) {
                float tr = translation[param_offset + w];
                float sc = scale[param_offset + w];
                float inv_sc = 1.0f / (fabs(sc) + 1e-4f);

                float z0 = (x0 - tr) * inv_sc;
                float z1 = (x1 - tr) * inv_sc;

                float psi0, psi1;
                if (wavelet_type == 1) {
                    psi0 = cos(5.0f * z0) * exp(-0.5f * z0 * z0);
                    psi1 = cos(5.0f * z1) * exp(-0.5f * z1 * z1);
                } else if (wavelet_type == 2) {
                    psi0 = -z0 * exp(-0.5f * z0 * z0);
                    psi1 = -z1 * exp(-0.5f * z1 * z1);
                } else {
                    psi0 = (1.0f - z0 * z0) * exp(-0.5f * z0 * z0);
                    psi1 = (1.0f - z1 * z1) * exp(-0.5f * z1 * z1);
                }

                if (col0 < D_out) {
                    float weight = W_wav[w_offset0 + w];
                    acc00 = fma(psi0, weight, acc00);
                    acc10 = fma(psi1, weight, acc10);
                }
                if (col1 < D_out) {
                    float weight = W_wav[w_offset1 + w];
                    acc01 = fma(psi0, weight, acc01);
                    acc11 = fma(psi1, weight, acc11);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row0 < B && col0 < D_out) Y[row0 * D_out + col0] = acc00;
    if (row0 < B && col1 < D_out) Y[row0 * D_out + col1] = acc01;
    if (row1 < B && col0 < D_out) Y[row1 * D_out + col0] = acc10;
    if (row1 < B && col1 < D_out) Y[row1 * D_out + col1] = acc11;
}

// ==============================================================================
// 5. FOURIERKAN FUSED TILED KERNEL (TRIGONOMETRIC HARMONICS)
// ==============================================================================

kernel void kan_fourier_tiled(
    device const float*  X         [[buffer(0)]],
    device const float*  W_fourier [[buffer(1)]],
    device const float*  W_base    [[buffer(2)]],
    device const float*  bias      [[buffer(3)]],
    device float*        Y         [[buffer(4)]],
    constant uint&       B         [[buffer(5)]],
    constant uint&       D_in      [[buffer(6)]],
    constant uint&       D_out     [[buffer(7)]],
    constant uint&       num_freqs [[buffer(8)]],
    constant uint&       has_base  [[buffer(9)]],
    constant uint&       has_bias  [[buffer(10)]],
    uint2                tg_pos    [[threadgroup_position_in_grid]],
    uint2                t_pos     [[thread_position_in_threadgroup]]
) {
    uint ty = t_pos.y;
    uint tx = t_pos.x;

    uint row0 = tg_pos.y * TILE_M + ty * 2;
    uint row1 = row0 + 1;
    uint col0 = tg_pos.x * TILE_N + tx * 2;
    uint col1 = col0 + 1;

    float acc00 = (has_bias && col0 < D_out) ? bias[col0] : 0.0f;
    float acc01 = (has_bias && col1 < D_out) ? bias[col1] : 0.0f;
    float acc10 = (has_bias && col0 < D_out) ? bias[col0] : 0.0f;
    float acc11 = (has_bias && col1 < D_out) ? bias[col1] : 0.0f;

    threadgroup float s_X[TILE_M][TILE_K];
    uint tid_flat = ty * 16 + tx;
    uint num_k_tiles = (D_in + TILE_K - 1) / TILE_K;
    uint num_bases = 2 * num_freqs + 1;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint load_r = tid_flat / TILE_K;
        uint load_c = tid_flat % TILE_K;
        uint global_r = tg_pos.y * TILE_M + load_r;
        uint global_c = kt * TILE_K + load_c;
        s_X[load_r][load_c] = (global_r < B && global_c < D_in) ? X[global_r * D_in + global_c] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_K; k++) {
            uint curr_din = kt * TILE_K + k;
            if (curr_din >= D_in) break;

            float x0 = s_X[ty * 2 + 0][k];
            float x1 = s_X[ty * 2 + 1][k];

            if (has_base) {
                float s0 = x0 / (1.0f + exp(-x0));
                float s1 = x1 / (1.0f + exp(-x1));
                if (col0 < D_out) {
                    float wb0 = W_base[col0 * D_in + curr_din];
                    acc00 = fma(s0, wb0, acc00);
                    acc10 = fma(s1, wb0, acc10);
                }
                if (col1 < D_out) {
                    float wb1 = W_base[col1 * D_in + curr_din];
                    acc01 = fma(s0, wb1, acc01);
                    acc11 = fma(s1, wb1, acc11);
                }
            }

            uint w_offset0 = col0 * (D_in * num_bases) + curr_din * num_bases;
            uint w_offset1 = col1 * (D_in * num_bases) + curr_din * num_bases;

            if (col0 < D_out) {
                acc00 += W_fourier[w_offset0];
                acc10 += W_fourier[w_offset0];
            }
            if (col1 < D_out) {
                acc01 += W_fourier[w_offset1];
                acc11 += W_fourier[w_offset1];
            }

            for (uint f = 0; f < num_freqs; f++) {
                float freq = (float)(f + 1) * M_PI_F;
                float c0 = cos(freq * x0);
                float s0 = sin(freq * x0);
                float c1 = cos(freq * x1);
                float s1 = sin(freq * x1);

                uint idx_cos = 1 + f * 2;
                uint idx_sin = 2 + f * 2;

                if (col0 < D_out) {
                    acc00 = fma(c0, W_fourier[w_offset0 + idx_cos], acc00);
                    acc00 = fma(s0, W_fourier[w_offset0 + idx_sin], acc00);
                    acc10 = fma(c1, W_fourier[w_offset0 + idx_cos], acc10);
                    acc10 = fma(s1, W_fourier[w_offset0 + idx_sin], acc10);
                }
                if (col1 < D_out) {
                    acc01 = fma(c0, W_fourier[w_offset1 + idx_cos], acc01);
                    acc01 = fma(s0, W_fourier[w_offset1 + idx_sin], acc01);
                    acc11 = fma(c1, W_fourier[w_offset1 + idx_cos], acc11);
                    acc11 = fma(s1, W_fourier[w_offset1 + idx_sin], acc11);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row0 < B && col0 < D_out) Y[row0 * D_out + col0] = acc00;
    if (row0 < B && col1 < D_out) Y[row0 * D_out + col1] = acc01;
    if (row1 < B && col0 < D_out) Y[row1 * D_out + col0] = acc10;
    if (row1 < B && col1 < D_out) Y[row1 * D_out + col1] = acc11;
}

// ==============================================================================
// 6. JACOBIKAN FUSED TILED KERNEL (ORTHOGONAL RECURRENCE)
// ==============================================================================

kernel void kan_jacobi_tiled(
    device const float*  X         [[buffer(0)]],
    device const float*  W_jacobi  [[buffer(1)]],
    device const float*  W_base    [[buffer(2)]],
    device const float*  bias      [[buffer(3)]],
    device float*        Y         [[buffer(4)]],
    constant uint&       B         [[buffer(5)]],
    constant uint&       D_in      [[buffer(6)]],
    constant uint&       D_out     [[buffer(7)]],
    constant uint&       degree    [[buffer(8)]],
    constant float&      alpha     [[buffer(9)]],
    constant float&      beta      [[buffer(10)]],
    constant uint&       has_base  [[buffer(11)]],
    constant uint&       has_bias  [[buffer(12)]],
    uint2                tg_pos    [[threadgroup_position_in_grid]],
    uint2                t_pos     [[thread_position_in_threadgroup]]
) {
    uint ty = t_pos.y;
    uint tx = t_pos.x;

    uint row0 = tg_pos.y * TILE_M + ty * 2;
    uint row1 = row0 + 1;
    uint col0 = tg_pos.x * TILE_N + tx * 2;
    uint col1 = col0 + 1;

    float acc00 = (has_bias && col0 < D_out) ? bias[col0] : 0.0f;
    float acc01 = (has_bias && col1 < D_out) ? bias[col1] : 0.0f;
    float acc10 = (has_bias && col0 < D_out) ? bias[col0] : 0.0f;
    float acc11 = (has_bias && col1 < D_out) ? bias[col1] : 0.0f;

    threadgroup float s_X[TILE_M][TILE_K];
    uint tid_flat = ty * 16 + tx;
    uint num_k_tiles = (D_in + TILE_K - 1) / TILE_K;
    float a_b = alpha + beta;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint load_r = tid_flat / TILE_K;
        uint load_c = tid_flat % TILE_K;
        uint global_r = tg_pos.y * TILE_M + load_r;
        uint global_c = kt * TILE_K + load_c;
        s_X[load_r][load_c] = (global_r < B && global_c < D_in) ? X[global_r * D_in + global_c] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_K; k++) {
            uint curr_din = kt * TILE_K + k;
            if (curr_din >= D_in) break;

            float x0 = clamp(s_X[ty * 2 + 0][k], -1.0f, 1.0f);
            float x1 = clamp(s_X[ty * 2 + 1][k], -1.0f, 1.0f);

            if (has_base) {
                float s0 = x0 / (1.0f + exp(-x0));
                float s1 = x1 / (1.0f + exp(-x1));
                if (col0 < D_out) {
                    float wb0 = W_base[col0 * D_in + curr_din];
                    acc00 = fma(s0, wb0, acc00);
                    acc10 = fma(s1, wb0, acc10);
                }
                if (col1 < D_out) {
                    float wb1 = W_base[col1 * D_in + curr_din];
                    acc01 = fma(s0, wb1, acc01);
                    acc11 = fma(s1, wb1, acc11);
                }
            }

            uint w_offset0 = col0 * (D_in * degree) + curr_din * degree;
            uint w_offset1 = col1 * (D_in * degree) + curr_din * degree;

            float p0_prev2 = 1.0f;
            float p1_prev2 = 1.0f;
            if (col0 < D_out) {
                acc00 = fma(p0_prev2, W_jacobi[w_offset0], acc00);
                acc10 = fma(p1_prev2, W_jacobi[w_offset0], acc10);
            }
            if (col1 < D_out) {
                acc01 = fma(p0_prev2, W_jacobi[w_offset1], acc01);
                acc11 = fma(p1_prev2, W_jacobi[w_offset1], acc11);
            }

            if (degree > 1) {
                float p0_prev1 = 0.5f * (alpha - beta + (a_b + 2.0f) * x0);
                float p1_prev1 = 0.5f * (alpha - beta + (a_b + 2.0f) * x1);
                if (col0 < D_out) {
                    acc00 = fma(p0_prev1, W_jacobi[w_offset0 + 1], acc00);
                    acc10 = fma(p1_prev1, W_jacobi[w_offset0 + 1], acc10);
                }
                if (col1 < D_out) {
                    acc01 = fma(p0_prev1, W_jacobi[w_offset1 + 1], acc01);
                    acc11 = fma(p1_prev1, W_jacobi[w_offset1 + 1], acc11);
                }

                for (uint n = 2; n < degree; n++) {
                    float fn = (float)n;
                    float an = 2.0f * fn * (fn + a_b) * (2.0f * fn + a_b - 2.0f);
                    float bn1 = (2.0f * fn + a_b - 1.0f) * (2.0f * fn + a_b) * (2.0f * fn + a_b - 2.0f);
                    float bn2 = (2.0f * fn + a_b - 1.0f) * (alpha * alpha - beta * beta);
                    float cn = 2.0f * (fn + alpha - 1.0f) * (fn + beta - 1.0f) * (2.0f * fn + a_b);

                    float p0_curr = ((bn1 * x0 + bn2) * p0_prev1 - cn * p0_prev2) / an;
                    float p1_curr = ((bn1 * x1 + bn2) * p1_prev1 - cn * p1_prev2) / an;

                    if (col0 < D_out) {
                        acc00 = fma(p0_curr, W_jacobi[w_offset0 + n], acc00);
                        acc10 = fma(p1_curr, W_jacobi[w_offset0 + n], acc10);
                    }
                    if (col1 < D_out) {
                        acc01 = fma(p0_curr, W_jacobi[w_offset1 + n], acc01);
                        acc11 = fma(p1_curr, W_jacobi[w_offset1 + n], acc11);
                    }

                    p0_prev2 = p0_prev1;
                    p0_prev1 = p0_curr;
                    p1_prev2 = p1_prev1;
                    p1_prev1 = p1_curr;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row0 < B && col0 < D_out) Y[row0 * D_out + col0] = acc00;
    if (row0 < B && col1 < D_out) Y[row0 * D_out + col1] = acc01;
    if (row1 < B && col0 < D_out) Y[row1 * D_out + col0] = acc10;
    if (row1 < B && col1 < D_out) Y[row1 * D_out + col1] = acc11;
}

// ==============================================================================
// 7. RATIONALKAN FUSED TILED KERNEL (PADÉ-CHEBYSHEV P(x) / (1 + |Q(x)|))
// ==============================================================================

kernel void kan_rational_tiled(
    device const float*  X        [[buffer(0)]],
    device const float*  W_p      [[buffer(1)]],
    device const float*  W_q      [[buffer(2)]],
    device const float*  W_base   [[buffer(3)]],
    device const float*  bias     [[buffer(4)]],
    device float*        Y        [[buffer(5)]],
    constant uint&       B        [[buffer(6)]],
    constant uint&       D_in     [[buffer(7)]],
    constant uint&       D_out    [[buffer(8)]],
    constant uint&       p_deg    [[buffer(9)]],
    constant uint&       q_deg    [[buffer(10)]],
    constant uint&       has_base [[buffer(11)]],
    constant uint&       has_bias [[buffer(12)]],
    uint2                tg_pos   [[threadgroup_position_in_grid]],
    uint2                t_pos    [[thread_position_in_threadgroup]]
) {
    uint ty = t_pos.y;
    uint tx = t_pos.x;

    uint row0 = tg_pos.y * TILE_M + ty * 2;
    uint row1 = row0 + 1;
    uint col0 = tg_pos.x * TILE_N + tx * 2;
    uint col1 = col0 + 1;

    float acc00 = (has_bias && col0 < D_out) ? bias[col0] : 0.0f;
    float acc01 = (has_bias && col1 < D_out) ? bias[col1] : 0.0f;
    float acc10 = (has_bias && col0 < D_out) ? bias[col0] : 0.0f;
    float acc11 = (has_bias && col1 < D_out) ? bias[col1] : 0.0f;

    threadgroup float s_X[TILE_M][TILE_K];
    uint tid_flat = ty * 16 + tx;
    uint num_k_tiles = (D_in + TILE_K - 1) / TILE_K;
    uint max_deg = max(p_deg, q_deg);

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint load_r = tid_flat / TILE_K;
        uint load_c = tid_flat % TILE_K;
        uint global_r = tg_pos.y * TILE_M + load_r;
        uint global_c = kt * TILE_K + load_c;
        s_X[load_r][load_c] = (global_r < B && global_c < D_in) ? X[global_r * D_in + global_c] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_K; k++) {
            uint curr_din = kt * TILE_K + k;
            if (curr_din >= D_in) break;

            float x0 = clamp(s_X[ty * 2 + 0][k], -1.0f, 1.0f);
            float x1 = clamp(s_X[ty * 2 + 1][k], -1.0f, 1.0f);

            if (has_base) {
                float s0 = x0 / (1.0f + exp(-x0));
                float s1 = x1 / (1.0f + exp(-x1));
                if (col0 < D_out) {
                    float wb0 = W_base[col0 * D_in + curr_din];
                    acc00 = fma(s0, wb0, acc00);
                    acc10 = fma(s1, wb0, acc10);
                }
                if (col1 < D_out) {
                    float wb1 = W_base[col1 * D_in + curr_din];
                    acc01 = fma(s0, wb1, acc01);
                    acc11 = fma(s1, wb1, acc11);
                }
            }

            uint wp_offset0 = col0 * (D_in * p_deg) + curr_din * p_deg;
            uint wp_offset1 = col1 * (D_in * p_deg) + curr_din * p_deg;
            uint wq_offset0 = col0 * (D_in * q_deg) + curr_din * q_deg;
            uint wq_offset1 = col1 * (D_in * q_deg) + curr_din * q_deg;

            float t0_p2 = 1.0f, t0_p1 = x0;
            float t1_p2 = 1.0f, t1_p1 = x1;

            float P0_0 = 0.0f, Q0_0 = 0.0f;
            float P0_1 = 0.0f, Q0_1 = 0.0f;
            float P1_0 = 0.0f, Q1_0 = 0.0f;
            float P1_1 = 0.0f, Q1_1 = 0.0f;

            for (uint d = 0; d < max_deg; d++) {
                float term0, term1;
                if (d == 0) {
                    term0 = 1.0f; term1 = 1.0f;
                } else if (d == 1) {
                    term0 = x0; term1 = x1;
                } else {
                    term0 = 2.0f * x0 * t0_p1 - t0_p2;
                    term1 = 2.0f * x1 * t1_p1 - t1_p2;
                    t0_p2 = t0_p1; t0_p1 = term0;
                    t1_p2 = t1_p1; t1_p1 = term1;
                }

                if (d < p_deg) {
                    if (col0 < D_out) {
                        P0_0 = fma(term0, W_p[wp_offset0 + d], P0_0);
                        P1_0 = fma(term1, W_p[wp_offset0 + d], P1_0);
                    }
                    if (col1 < D_out) {
                        P0_1 = fma(term0, W_p[wp_offset1 + d], P0_1);
                        P1_1 = fma(term1, W_p[wp_offset1 + d], P1_1);
                    }
                }

                if (d < q_deg) {
                    if (col0 < D_out) {
                        Q0_0 = fma(term0, W_q[wq_offset0 + d], Q0_0);
                        Q1_0 = fma(term1, W_q[wq_offset0 + d], Q1_0);
                    }
                    if (col1 < D_out) {
                        Q0_1 = fma(term0, W_q[wq_offset1 + d], Q0_1);
                        Q1_1 = fma(term1, W_q[wq_offset1 + d], Q1_1);
                    }
                }
            }

            if (col0 < D_out) {
                acc00 += P0_0 / (1.0f + fabs(Q0_0));
                acc10 += P1_0 / (1.0f + fabs(Q1_0));
            }
            if (col1 < D_out) {
                acc01 += P0_1 / (1.0f + fabs(Q0_1));
                acc11 += P1_1 / (1.0f + fabs(Q1_1));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row0 < B && col0 < D_out) Y[row0 * D_out + col0] = acc00;
    if (row0 < B && col1 < D_out) Y[row0 * D_out + col1] = acc01;
    if (row1 < B && col0 < D_out) Y[row1 * D_out + col0] = acc10;
    if (row1 < B && col1 < D_out) Y[row1 * D_out + col1] = acc11;
}

// ==============================================================================
// 8. BSPLINE KAN FUSED TILED KERNEL (COX-DE BOOR B-SPLINES)
// ==============================================================================

kernel void kan_bspline_tiled(
    device const float*  X            [[buffer(0)]],
    device const float*  W_spline     [[buffer(1)]],
    device const float*  W_base       [[buffer(2)]],
    device const float*  grid         [[buffer(3)]],
    device const float*  bias         [[buffer(4)]],
    device float*        Y            [[buffer(5)]],
    constant uint&       B            [[buffer(6)]],
    constant uint&       D_in         [[buffer(7)]],
    constant uint&       D_out        [[buffer(8)]],
    constant uint&       grid_size    [[buffer(9)]],
    constant uint&       spline_order [[buffer(10)]],
    constant uint&       has_base     [[buffer(11)]],
    constant uint&       has_bias     [[buffer(12)]],
    uint2                tg_pos       [[threadgroup_position_in_grid]],
    uint2                t_pos        [[thread_position_in_threadgroup]]
) {
    uint ty = t_pos.y;
    uint tx = t_pos.x;

    uint row0 = tg_pos.y * TILE_M + ty * 2;
    uint row1 = row0 + 1;
    uint col0 = tg_pos.x * TILE_N + tx * 2;
    uint col1 = col0 + 1;

    float acc00 = (has_bias && col0 < D_out) ? bias[col0] : 0.0f;
    float acc01 = (has_bias && col1 < D_out) ? bias[col1] : 0.0f;
    float acc10 = (has_bias && col0 < D_out) ? bias[col0] : 0.0f;
    float acc11 = (has_bias && col1 < D_out) ? bias[col1] : 0.0f;

    threadgroup float s_X[TILE_M][TILE_K];
    uint tid_flat = ty * 16 + tx;
    uint num_k_tiles = (D_in + TILE_K - 1) / TILE_K;
    uint num_bases = grid_size + spline_order;
    uint num_knots = grid_size + 2 * spline_order + 1;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint load_r = tid_flat / TILE_K;
        uint load_c = tid_flat % TILE_K;
        uint global_r = tg_pos.y * TILE_M + load_r;
        uint global_c = kt * TILE_K + load_c;
        s_X[load_r][load_c] = (global_r < B && global_c < D_in) ? X[global_r * D_in + global_c] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_K; k++) {
            uint curr_din = kt * TILE_K + k;
            if (curr_din >= D_in) break;

            float x0 = s_X[ty * 2 + 0][k];
            float x1 = s_X[ty * 2 + 1][k];

            if (has_base) {
                float s0 = x0 / (1.0f + exp(-x0));
                float s1 = x1 / (1.0f + exp(-x1));
                if (col0 < D_out) {
                    float wb0 = W_base[col0 * D_in + curr_din];
                    acc00 = fma(s0, wb0, acc00);
                    acc10 = fma(s1, wb0, acc10);
                }
                if (col1 < D_out) {
                    float wb1 = W_base[col1 * D_in + curr_din];
                    acc01 = fma(s0, wb1, acc01);
                    acc11 = fma(s1, wb1, acc11);
                }
            }

            uint w_offset0 = col0 * (D_in * num_bases) + curr_din * num_bases;
            uint w_offset1 = col1 * (D_in * num_bases) + curr_din * num_bases;
            uint knot_offset = curr_din * num_knots;

            float b0[64];
            float b1[64];
            uint total_intervals = num_knots - 1;
            for (uint i = 0; i < total_intervals && i < 64; i++) {
                float g_left = grid[knot_offset + i];
                float g_right = grid[knot_offset + i + 1];
                b0[i] = (x0 >= g_left && x0 < g_right) ? 1.0f : 0.0f;
                b1[i] = (x1 >= g_left && x1 < g_right) ? 1.0f : 0.0f;
            }

            for (uint p = 1; p <= spline_order; p++) {
                uint num_p_bases = num_knots - p - 1;
                for (uint i = 0; i < num_p_bases && i < 64; i++) {
                    float g_i = grid[knot_offset + i];
                    float g_ip = grid[knot_offset + i + p];
                    float g_ip1 = grid[knot_offset + i + p + 1];
                    float g_i1 = grid[knot_offset + i + 1];

                    float d1 = g_ip - g_i;
                    float term1_0 = (d1 > 1e-7f) ? ((x0 - g_i) / d1) * b0[i] : 0.0f;
                    float term1_1 = (d1 > 1e-7f) ? ((x1 - g_i) / d1) * b1[i] : 0.0f;

                    float d2 = g_ip1 - g_i1;
                    float term2_0 = (d2 > 1e-7f) ? ((g_ip1 - x0) / d2) * b0[i + 1] : 0.0f;
                    float term2_1 = (d2 > 1e-7f) ? ((g_ip1 - x1) / d2) * b1[i + 1] : 0.0f;

                    b0[i] = term1_0 + term2_0;
                    b1[i] = term1_1 + term2_1;
                }
            }

            for (uint i = 0; i < num_bases; i++) {
                if (col0 < D_out) {
                    float w = W_spline[w_offset0 + i];
                    acc00 = fma(b0[i], w, acc00);
                    acc10 = fma(b1[i], w, acc10);
                }
                if (col1 < D_out) {
                    float w = W_spline[w_offset1 + i];
                    acc01 = fma(b0[i], w, acc01);
                    acc11 = fma(b1[i], w, acc11);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row0 < B && col0 < D_out) Y[row0 * D_out + col0] = acc00;
    if (row0 < B && col1 < D_out) Y[row0 * D_out + col1] = acc01;
    if (row1 < B && col0 < D_out) Y[row1 * D_out + col0] = acc10;
    if (row1 < B && col1 < D_out) Y[row1 * D_out + col1] = acc11;
}
