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
