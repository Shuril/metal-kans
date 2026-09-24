#include <metal_stdlib>
#include <metal_matrix>
using namespace metal;

// Chebyshev fused tiled kernel with 2x2 register blocking.
// Fuses basis evaluation and matrix multiply-accumulate to avoid intermediate VRAM writes.

#define TILE_M 32
#define TILE_N 32
#define TILE_K 8
#define TILE_K_PAD (TILE_K + 1)

inline float fast_silu(float x) {
    constexpr float inv_ln2 = 1.4426950408889634f;
    return x / (1.0f + metal::fast::exp2(-inv_ln2 * x));
}

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

    threadgroup float  s_X[TILE_M][TILE_K_PAD];
    threadgroup float4 s_W[TILE_N][TILE_K_PAD];
    threadgroup float  s_Wb[TILE_N][TILE_K_PAD];

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
                float s0 = fast_silu(x_val0);
                float s1 = fast_silu(x_val1);
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

// Hardware-accelerated basis and preparation kernels for MPS GEMM dispatch.

kernel void eval_base_and_bias(
    device const float*  X        [[buffer(0)]],
    device const float*  bias     [[buffer(1)]],
    device float*        X_silu   [[buffer(2)]],
    device float*        Y        [[buffer(3)]],
    constant uint&       B        [[buffer(4)]],
    constant uint&       D_in     [[buffer(5)]],
    constant uint&       D_out    [[buffer(6)]],
    constant uint&       has_base [[buffer(7)]],
    constant uint&       has_bias [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint r = gid.y;
    uint c = gid.x;
    float inv_ln2 = 1.4426950408889634f;
    if (r < B && c < D_in && has_base) {
        float x = X[r * D_in + c];
        X_silu[r * D_in + c] = x / (1.0f + exp2(-inv_ln2 * x));
    }
    if (r < B && c < D_out) {
        Y[r * D_out + c] = (has_bias) ? bias[c] : 0.0f;
    }
}

kernel void eval_fastkan_rbf_basis(
    device const float*  X        [[buffer(0)]],
    device const float*  grid     [[buffer(1)]],
    device float*        Phi      [[buffer(2)]],
    constant uint&       B        [[buffer(3)]],
    constant uint&       D_in     [[buffer(4)]],
    constant uint&       K        [[buffer(5)]],
    constant float&      inv_d    [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint b = gid.y;
    uint din = gid.x;
    if (b >= B || din >= D_in) return;
    float x = X[b * D_in + din];
    uint out_offset = b * (D_in * K) + din * K;
    float inv_ln2 = 1.4426950408889634f;
    uint num_vec4 = K / 4;
    for (uint v = 0; v < num_vec4; v++) {
        float4 g = ((device const float4*)grid)[v];
        float4 diff = x - g;
        ((device float4*)(Phi + out_offset))[v] = exp2(-inv_ln2 * (diff * diff) * inv_d);
    }
    for (uint c = num_vec4 * 4; c < K; c++) {
        float diff = x - grid[c];
        Phi[out_offset + c] = exp2(-inv_ln2 * (diff * diff) * inv_d);
    }
}

kernel void eval_wavkan_basis(
    device const float*  X        [[buffer(0)]],
    device const float*  trans    [[buffer(1)]],
    device const float*  inv_sc   [[buffer(2)]],
    device float*        Phi      [[buffer(3)]],
    constant uint&       B        [[buffer(4)]],
    constant uint&       D_in     [[buffer(5)]],
    constant uint&       num_wav  [[buffer(6)]],
    constant uint&       wtype    [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint b = gid.y;
    uint din = gid.x;
    if (b >= B || din >= D_in) return;
    float x = X[b * D_in + din];
    uint param_offset = din * num_wav;
    uint out_offset = b * (D_in * num_wav) + din * num_wav;
    float inv_ln2 = 1.4426950408889634f;
    uint num_vec4 = num_wav / 4;
    for (uint v = 0; v < num_vec4; v++) {
        float4 tr = ((device const float4*)(trans + param_offset))[v];
        float4 isc = ((device const float4*)(inv_sc + param_offset))[v];
        float4 z = (x - tr) * isc;
        float4 exp_z = exp2(-0.5f * inv_ln2 * (z * z));
        float4 psi = (wtype == 1) ? (cos(5.0f * z) * exp_z) : ((wtype == 2) ? (-z * exp_z) : ((1.0f - z * z) * exp_z));
        ((device float4*)(Phi + out_offset))[v] = psi;
    }
    for (uint w = num_vec4 * 4; w < num_wav; w++) {
        float tr = trans[param_offset + w];
        float isc = inv_sc[param_offset + w];
        float z = (x - tr) * isc;
        float exp_z = exp2(-0.5f * inv_ln2 * z * z);
        float psi = (wtype == 1) ? (cos(5.0f * z) * exp_z) : ((wtype == 2) ? (-z * exp_z) : ((1.0f - z * z) * exp_z));
        Phi[out_offset + w] = psi;
    }
}

kernel void eval_cheby_basis(
    device const float*  X     [[buffer(0)]],
    device float*        Phi   [[buffer(1)]],
    constant uint&       B     [[buffer(2)]],
    constant uint&       D_in  [[buffer(3)]],
    constant uint&       deg   [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint b = gid.y;
    uint din = gid.x;
    if (b >= B || din >= D_in) return;
    float x = clamp(X[b * D_in + din], -1.0f, 1.0f);
    uint out_offset = b * (D_in * deg) + din * deg;

    if (deg == 4) {
        float x2 = x * x;
        float t0 = 1.0f;
        float t1 = x;
        float t2 = 2.0f * x2 - 1.0f;
        float t3 = 4.0f * x2 * x - 3.0f * x;
        ((device float4*)(Phi + out_offset))[0] = float4(t0, t1, t2, t3);
    } else {
        float t_prev2 = 1.0f;
        float t_prev1 = x;
        Phi[out_offset + 0] = t_prev2;
        if (deg > 1) Phi[out_offset + 1] = t_prev1;
        for (uint d = 2; d < deg; d++) {
            float t_curr = 2.0f * x * t_prev1 - t_prev2;
            Phi[out_offset + d] = t_curr;
            t_prev2 = t_prev1;
            t_prev1 = t_curr;
        }
    }
}

kernel void eval_bspline_basis(
    device const float*  X            [[buffer(0)]],
    device float*        Phi          [[buffer(1)]],
    constant uint&       B            [[buffer(2)]],
    constant uint&       D_in         [[buffer(3)]],
    constant uint&       grid_size    [[buffer(4)]],
    constant float&      grid_min     [[buffer(5)]],
    constant float&      inv_h        [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint b = gid.y;
    uint din = gid.x;
    if (b >= B || din >= D_in) return;
    float x = X[b * D_in + din];
    uint num_bases = grid_size + 3;
    uint out_offset = b * (D_in * num_bases) + din * num_bases;

    for (uint i = 0; i < num_bases; i++) {
        Phi[out_offset + i] = 0.0f;
    }

    float pos = (x - grid_min) * inv_h;
    int span = clamp((int)floor(pos), 0, (int)grid_size - 1);
    float u = clamp(pos - (float)span, 0.0f, 1.0f);
    float one_sub_u = 1.0f - u;

    constexpr float inv6 = 0.16666667f;
    float b0 = (one_sub_u * one_sub_u * one_sub_u) * inv6;
    float b1 = (3.0f * u * u * u - 6.0f * u * u + 4.0f) * inv6;
    float b2 = (-3.0f * u * u * u + 3.0f * u * u + 3.0f * u + 1.0f) * inv6;
    float b3 = (u * u * u) * inv6;

    Phi[out_offset + span + 0] = b0;
    Phi[out_offset + span + 1] = b1;
    Phi[out_offset + span + 2] = b2;
    Phi[out_offset + span + 3] = b3;
}

kernel void eval_relu_basis(
    device const float*  X     [[buffer(0)]],
    device const float*  grid  [[buffer(1)]],
    device float*        Phi   [[buffer(2)]],
    constant uint&       B     [[buffer(3)]],
    constant uint&       D_in  [[buffer(4)]],
    constant uint&       G     [[buffer(5)]],
    constant float&      inv_h [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint b = gid.y;
    uint din = gid.x;
    if (b >= B || din >= D_in) return;
    float x = X[b * D_in + din];
    uint out_offset = b * (D_in * G) + din * G;
    uint num_vec4 = G / 4;
    for (uint v = 0; v < num_vec4; v++) {
        float4 g = ((device const float4*)grid)[v];
        float4 diff = fabs(x - g);
        float4 t = max(0.0f, 1.0f - diff * inv_h);
        ((device float4*)(Phi + out_offset))[v] = t;
    }
    for (uint g = num_vec4 * 4; g < G; g++) {
        float diff = fabs(x - grid[g]);
        Phi[out_offset + g] = max(0.0f, 1.0f - diff * inv_h);
    }
}

kernel void eval_fourier_basis(
    device const float*  X          [[buffer(0)]],
    device float*        Phi        [[buffer(1)]],
    constant uint&       B          [[buffer(2)]],
    constant uint&       D_in       [[buffer(3)]],
    constant uint&       num_freqs  [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint b = gid.y;
    uint din = gid.x;
    if (b >= B || din >= D_in) return;
    float x = X[b * D_in + din];
    uint num_bases = 2 * num_freqs + 1;
    uint out_offset = b * (D_in * num_bases) + din * num_bases;

    Phi[out_offset + 0] = 1.0f;
    for (uint f = 0; f < num_freqs; f++) {
        float freq = (float)(f + 1) * M_PI_F;
        float c, s;
        s = sincos(freq * x, c);
        Phi[out_offset + 1 + f * 2] = c;
        Phi[out_offset + 2 + f * 2] = s;
    }
}

kernel void eval_jacobi_basis(
    device const float*  X         [[buffer(0)]],
    device float*        Phi       [[buffer(1)]],
    constant uint&       B         [[buffer(2)]],
    constant uint&       D_in      [[buffer(3)]],
    constant uint&       degree    [[buffer(4)]],
    constant float&      alpha     [[buffer(5)]],
    constant float&      beta      [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint b = gid.y;
    uint din = gid.x;
    if (b >= B || din >= D_in) return;
    float x = clamp(X[b * D_in + din], -1.0f, 1.0f);
    uint out_offset = b * (D_in * degree) + din * degree;
    float a_b = alpha + beta;

    float p_prev2 = 1.0f;
    Phi[out_offset + 0] = p_prev2;
    if (degree > 1) {
        float p_prev1 = 0.5f * (alpha - beta + (a_b + 2.0f) * x);
        Phi[out_offset + 1] = p_prev1;
        for (uint n = 2; n < degree; n++) {
            float fn = (float)n;
            float an = 2.0f * fn * (fn + a_b) * (2.0f * fn + a_b - 2.0f);
            float bn1 = (2.0f * fn + a_b - 1.0f) * (2.0f * fn + a_b) * (2.0f * fn + a_b - 2.0f);
            float bn2 = (2.0f * fn + a_b - 1.0f) * (alpha * alpha - beta * beta);
            float cn = 2.0f * (fn + alpha - 1.0f) * (fn + beta - 1.0f) * (2.0f * fn + a_b);

            float p_curr = ((bn1 * x + bn2) * p_prev1 - cn * p_prev2) / an;
            Phi[out_offset + n] = p_curr;
            p_prev2 = p_prev1;
            p_prev1 = p_curr;
        }
    }
}

kernel void combine_mult_nodes(
    device const float* In        [[buffer(0)]],
    device float*       Out       [[buffer(1)]],
    constant uint&      B         [[buffer(2)]],
    constant uint&      num_add   [[buffer(3)]],
    constant uint&      num_mult  [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint b = gid.y;
    uint j = gid.x;
    if (b >= B || j >= (num_add + num_mult)) return;
    uint total_in = num_add + 2 * num_mult;
    uint total_out = num_add + num_mult;
    if (j < num_add) {
        Out[b * total_out + j] = In[b * total_in + j];
    } else {
        uint m = j - num_add;
        float u = In[b * total_in + num_add + m];
        float v = In[b * total_in + num_add + num_mult + m];
        Out[b * total_out + j] = u * v;
    }
}

// FastKAN fused tiled kernel (Gaussian RBF).

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

    threadgroup float s_X[TILE_M][TILE_K_PAD];
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
                float s0 = fast_silu(x0);
                float s1 = fast_silu(x1);
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

            uint num_vec4 = num_centers / 4;
            float inv_ln2 = 1.4426950408889634f;
            float scaled_inv_denom = inv_denom * inv_ln2;

            for (uint v = 0; v < num_vec4; v++) {
                float4 mu = ((device const float4*)grid)[v];
                float4 d0 = x0 - mu;
                float4 d1 = x1 - mu;
                float4 rbf0 = exp2(-(d0 * d0) * scaled_inv_denom);
                float4 rbf1 = exp2(-(d1 * d1) * scaled_inv_denom);

                if (col0 < D_out) {
                    float4 w = ((device const float4*)(W_rbf + w_offset0))[v];
                    acc00 += dot(rbf0, w);
                    acc10 += dot(rbf1, w);
                }
                if (col1 < D_out) {
                    float4 w = ((device const float4*)(W_rbf + w_offset1))[v];
                    acc01 += dot(rbf0, w);
                    acc11 += dot(rbf1, w);
                }
            }

            for (uint c = num_vec4 * 4; c < num_centers; c++) {
                float mu = grid[c];
                float d0 = x0 - mu;
                float d1 = x1 - mu;
                float rbf0 = exp2(-(d0 * d0) * scaled_inv_denom);
                float rbf1 = exp2(-(d1 * d1) * scaled_inv_denom);

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

// ReLUKAN fused tiled kernel (tent basis).

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

    threadgroup float s_X[TILE_M][TILE_K_PAD];
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
                float s0 = fast_silu(x0);
                float s1 = fast_silu(x1);
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

            uint num_vec4 = num_grids / 4;
            for (uint v = 0; v < num_vec4; v++) {
                float4 mu = ((device const float4*)grid)[v];
                float4 t0 = max(float4(0.0f), 1.0f - abs(x0 - mu) * inv_h);
                float4 t1 = max(float4(0.0f), 1.0f - abs(x1 - mu) * inv_h);

                if (col0 < D_out) {
                    float4 w = ((device const float4*)(W_relu + w_offset0))[v];
                    acc00 += dot(t0, w);
                    acc10 += dot(t1, w);
                }
                if (col1 < D_out) {
                    float4 w = ((device const float4*)(W_relu + w_offset1))[v];
                    acc01 += dot(t0, w);
                    acc11 += dot(t1, w);
                }
            }

            for (uint g = num_vec4 * 4; g < num_grids; g++) {
                float mu = grid[g];
                float t0 = max(0.0f, 1.0f - abs(x0 - mu) * inv_h);
                float t1 = max(0.0f, 1.0f - abs(x1 - mu) * inv_h);

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

// WavKAN fused tiled kernel (continuous wavelets).

kernel void kan_wavkan_tiled(
    device const float*  X            [[buffer(0)]],
    device const float*  W_wav        [[buffer(1)]],
    device const float*  W_base       [[buffer(2)]],
    device const float*  translation  [[buffer(3)]],
    device const float*  inv_scale    [[buffer(4)]],
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

    threadgroup float s_X[TILE_M][TILE_K_PAD];
    uint tid_flat = ty * 16 + tx;
    uint num_k_tiles = (D_in + TILE_K - 1) / TILE_K;
    float inv_ln2 = 1.4426950408889634f;

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
                float s0 = x0 / (1.0f + exp2(-inv_ln2 * x0));
                float s1 = x1 / (1.0f + exp2(-inv_ln2 * x1));
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

            uint num_vec4 = num_wavelets / 4;
            for (uint v = 0; v < num_vec4; v++) {
                float4 tr = ((device const float4*)(translation + param_offset))[v];
                float4 isc = ((device const float4*)(inv_scale + param_offset))[v];

                float4 z0 = (x0 - tr) * isc;
                float4 z1 = (x1 - tr) * isc;

                float4 exp0 = exp2(-0.5f * inv_ln2 * (z0 * z0));
                float4 exp1 = exp2(-0.5f * inv_ln2 * (z1 * z1));

                float4 psi0 = (wavelet_type == 1) ? (cos(5.0f * z0) * exp0) :
                              ((wavelet_type == 2) ? (-z0 * exp0) : ((1.0f - z0 * z0) * exp0));
                float4 psi1 = (wavelet_type == 1) ? (cos(5.0f * z1) * exp1) :
                              ((wavelet_type == 2) ? (-z1 * exp1) : ((1.0f - z1 * z1) * exp1));

                if (col0 < D_out) {
                    float4 w0 = ((device const float4*)(W_wav + w_offset0))[v];
                    acc00 += dot(psi0, w0);
                    acc10 += dot(psi1, w0);
                }
                if (col1 < D_out) {
                    float4 w1 = ((device const float4*)(W_wav + w_offset1))[v];
                    acc01 += dot(psi0, w1);
                    acc11 += dot(psi1, w1);
                }
            }

            for (uint w = num_vec4 * 4; w < num_wavelets; w++) {
                float tr = translation[param_offset + w];
                float isc = inv_scale[param_offset + w];
                float z0 = (x0 - tr) * isc;
                float z1 = (x1 - tr) * isc;
                float exp0 = exp2(-0.5f * inv_ln2 * z0 * z0);
                float exp1 = exp2(-0.5f * inv_ln2 * z1 * z1);
                float psi0 = (wavelet_type == 1) ? cos(5.0f * z0) * exp0 : ((wavelet_type == 2) ? -z0 * exp0 : (1.0f - z0 * z0) * exp0);
                float psi1 = (wavelet_type == 1) ? cos(5.0f * z1) * exp1 : ((wavelet_type == 2) ? -z1 * exp1 : (1.0f - z1 * z1) * exp1);
                if (col0 < D_out) {
                    acc00 = fma(psi0, W_wav[w_offset0 + w], acc00);
                    acc10 = fma(psi1, W_wav[w_offset0 + w], acc10);
                }
                if (col1 < D_out) {
                    acc01 = fma(psi0, W_wav[w_offset1 + w], acc01);
                    acc11 = fma(psi1, W_wav[w_offset1 + w], acc11);
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

// FourierKAN fused tiled kernel (trigonometric harmonics).

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

    threadgroup float s_X[TILE_M][TILE_K_PAD];
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
                float s0 = fast_silu(x0);
                float s1 = fast_silu(x1);
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
                float c0, s0, c1, s1;
                s0 = sincos(freq * x0, c0);
                s1 = sincos(freq * x1, c1);

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

// JacobiKAN fused tiled kernel (orthogonal recurrence).

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

    threadgroup float s_X[TILE_M][TILE_K_PAD];
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
                float s0 = fast_silu(x0);
                float s1 = fast_silu(x1);
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

// RationalKAN fused tiled kernel (Padé-Chebyshev rational functions).

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

    threadgroup float s_X[TILE_M][TILE_K_PAD];
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
                float s0 = fast_silu(x0);
                float s1 = fast_silu(x1);
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

// BSplineKAN fused tiled kernel (Cox-de Boor B-splines).

kernel void kan_bspline_tiled(
    device const float*  X            [[buffer(0)]],
    device const float*  W_spline     [[buffer(1)]],
    device const float*  W_base       [[buffer(2)]],
    device const float*  grid         [[buffer(3)]], // [grid_min, inv_h]
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

    threadgroup float s_X[TILE_M][TILE_K_PAD];
    uint tid_flat = ty * 16 + tx;
    uint num_k_tiles = (D_in + TILE_K - 1) / TILE_K;
    uint num_bases = grid_size + 3;

    float grid_min = grid[0];
    float inv_h = grid[1];

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
                float s0 = fast_silu(x0);
                float s1 = fast_silu(x1);
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

            float pos0 = (x0 - grid_min) * inv_h;
            float pos1 = (x1 - grid_min) * inv_h;

            int span0 = clamp((int)floor(pos0), 0, (int)grid_size - 1);
            int span1 = clamp((int)floor(pos1), 0, (int)grid_size - 1);

            float u0 = clamp(pos0 - (float)span0, 0.0f, 1.0f);
            float u1 = clamp(pos1 - (float)span1, 0.0f, 1.0f);

            float one_sub_u0 = 1.0f - u0;
            constexpr float inv6 = 0.16666667f;
            float b0_0 = (one_sub_u0 * one_sub_u0 * one_sub_u0) * inv6;
            float b0_1 = (3.0f * u0 * u0 * u0 - 6.0f * u0 * u0 + 4.0f) * inv6;
            float b0_2 = (-3.0f * u0 * u0 * u0 + 3.0f * u0 * u0 + 3.0f * u0 + 1.0f) * inv6;
            float b0_3 = (u0 * u0 * u0) * inv6;

            float one_sub_u1 = 1.0f - u1;
            float b1_0 = (one_sub_u1 * one_sub_u1 * one_sub_u1) * inv6;
            float b1_1 = (3.0f * u1 * u1 * u1 - 6.0f * u1 * u1 + 4.0f) * inv6;
            float b1_2 = (-3.0f * u1 * u1 * u1 + 3.0f * u1 * u1 + 3.0f * u1 + 1.0f) * inv6;
            float b1_3 = (u1 * u1 * u1) * inv6;

            float4 b0 = float4(b0_0, b0_1, b0_2, b0_3);
            float4 b1 = float4(b1_0, b1_1, b1_2, b1_3);

            uint w_offset0 = col0 * (D_in * num_bases) + curr_din * num_bases;
            uint w_offset1 = col1 * (D_in * num_bases) + curr_din * num_bases;

            if (col0 < D_out) {
                float4 w0_s0 = float4(W_spline[w_offset0 + span0 + 0], W_spline[w_offset0 + span0 + 1], W_spline[w_offset0 + span0 + 2], W_spline[w_offset0 + span0 + 3]);
                float4 w0_s1 = float4(W_spline[w_offset0 + span1 + 0], W_spline[w_offset0 + span1 + 1], W_spline[w_offset0 + span1 + 2], W_spline[w_offset0 + span1 + 3]);
                acc00 += dot(b0, w0_s0);
                acc10 += dot(b1, w0_s1);
            }

            if (col1 < D_out) {
                float4 w1_s0 = float4(W_spline[w_offset1 + span0 + 0], W_spline[w_offset1 + span0 + 1], W_spline[w_offset1 + span0 + 2], W_spline[w_offset1 + span0 + 3]);
                float4 w1_s1 = float4(W_spline[w_offset1 + span1 + 0], W_spline[w_offset1 + span1 + 1], W_spline[w_offset1 + span1 + 2], W_spline[w_offset1 + span1 + 3]);
                acc01 += dot(b0, w1_s0);
                acc11 += dot(b1, w1_s1);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row0 < B && col0 < D_out) Y[row0 * D_out + col0] = acc00;
    if (row0 < B && col1 < D_out) Y[row0 * D_out + col1] = acc01;
    if (row1 < B && col0 < D_out) Y[row1 * D_out + col0] = acc10;
    if (row1 < B && col1 < D_out) Y[row1 * D_out + col1] = acc11;
}

// Direct simdgroup_matrix kernels for Apple Silicon tensor cores.

kernel void gemm_simd_16x16_fp32(
    device const float* A          [[buffer(0)]], // [M, K]
    device const float* B_mat      [[buffer(1)]], // [N, K]
    device const float* bias       [[buffer(2)]], // [N]
    device float*       C          [[buffer(3)]], // [M, N]
    constant uint&      M          [[buffer(4)]],
    constant uint&      N          [[buffer(5)]],
    constant uint&      K          [[buffer(6)]],
    constant float&     alpha      [[buffer(7)]],
    constant float&     beta       [[buffer(8)]],
    constant uint&      has_bias   [[buffer(9)]],
    uint2               tgid       [[threadgroup_position_in_grid]],
    uint                simd_id    [[simdgroup_index_in_threadgroup]],
    uint                s_lane     [[thread_index_in_simdgroup]]
) {
    uint tile_row = tgid.y * 16;
    uint tile_col = tgid.x * 16;

    uint sub_row = (simd_id / 2) * 8;
    uint sub_col = (simd_id % 2) * 8;

    uint r = tile_row + sub_row;
    uint c = tile_col + sub_col;

    if (r >= M || c >= N) return;

    simdgroup_matrix<float, 8, 8> accum(0.0f);
    simdgroup_matrix<float, 8, 8> matA;
    simdgroup_matrix<float, 8, 8> matB;

    for (uint k = 0; k < K; k += 8) {
        simdgroup_load(matA, A + r * K + k, K);
        simdgroup_load(matB, B_mat + c * K + k, K);
        simdgroup_multiply_accumulate(accum, matA, matB, accum);
    }

    threadgroup float temp[4][8][8];
    simdgroup_store(accum, &temp[simd_id][0][0], 8);

    uint idx0 = s_lane * 2;
    uint idx1 = idx0 + 1;

    uint r0 = idx0 / 8, c0 = idx0 % 8;
    uint r1 = idx1 / 8, c1 = idx1 % 8;

    uint gr0 = r + r0, gc0 = c + c0;
    uint gr1 = r + r1, gc1 = c + c1;

    if (gr0 < M && gc0 < N) {
        float b_val = (has_bias && bias) ? bias[gc0] : 0.0f;
        float old_val = (beta != 0.0f) ? C[gr0 * N + gc0] : 0.0f;
        C[gr0 * N + gc0] = temp[simd_id][r0][c0] * alpha + old_val * beta + b_val;
    }
    if (gr1 < M && gc1 < N) {
        float b_val = (has_bias && bias) ? bias[gc1] : 0.0f;
        float old_val = (beta != 0.0f) ? C[gr1 * N + gc1] : 0.0f;
        C[gr1 * N + gc1] = temp[simd_id][r1][c1] * alpha + old_val * beta + b_val;
    }
}

// Single-pass Fused LowRankKAN Kernel for small and medium batches
kernel void fused_lowrank_simd_fp32(
    device const float* X          [[buffer(0)]], // [B, D_in]
    device const float* W_down     [[buffer(1)]], // [rank, D_in * K]
    device const float* W_up       [[buffer(2)]], // [D_out, rank]
    device const float* W_base     [[buffer(3)]], // [D_out, D_in]
    device const float* grid       [[buffer(4)]], // [K]
    device const float* bias       [[buffer(5)]], // [D_out]
    device float*       Y          [[buffer(6)]], // [B, D_out]
    constant uint&      B          [[buffer(7)]],
    constant uint&      D_in       [[buffer(8)]],
    constant uint&      D_out      [[buffer(9)]],
    constant uint&      rank       [[buffer(10)]],
    constant uint&      K          [[buffer(11)]],
    constant float&     inv_den    [[buffer(12)]],
    constant uint&      has_base   [[buffer(13)]],
    constant uint&      has_bias   [[buffer(14)]],
    uint2               tgid       [[threadgroup_position_in_grid]],
    uint                lid        [[thread_index_in_threadgroup]]
) {
    uint b = tgid.y;
    if (b >= B) return;

    threadgroup float s_Z[64];

    // Cooperatively evaluate bottleneck Z
    if (lid < rank && rank <= 64) {
        uint r = lid;
        float sum_r = 0.0f;
        for (uint i = 0; i < D_in; i++) {
            float x_val = X[b * D_in + i];
            for (uint k = 0; k < K; k++) {
                float diff = x_val - grid[k];
                sum_r += exp(-diff * diff * inv_den) * W_down[r * (D_in * K) + i * K + k];
            }
        }
        s_Z[r] = sum_r;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute output features Y[b, j]
    for (uint j = lid; j < D_out; j += 64) {
        float acc = (has_bias && bias) ? bias[j] : 0.0f;
        for (uint r = 0; r < rank; r++) {
            acc += s_Z[r] * W_up[j * rank + r];
        }
        if (has_base && W_base) {
            for (uint i = 0; i < D_in; i++) {
                float x_val = X[b * D_in + i];
                float silu_x = x_val / (1.0f + exp(-x_val));
                acc += silu_x * W_base[j * D_in + i];
            }
        }
        Y[b * D_out + j] = acc;
    }
}

// FP16 half-precision basis generators.

kernel void eval_base_and_bias_fp16(
    device const half* X        [[buffer(0)]], // [B, D_in]
    device const half* bias     [[buffer(1)]], // [D_out]
    device half*       X_silu   [[buffer(2)]], // [B, D_in]
    device half*       Y        [[buffer(3)]], // [B, D_out]
    constant uint&     B        [[buffer(4)]],
    constant uint&     D_in     [[buffer(5)]],
    constant uint&     D_out    [[buffer(6)]],
    constant uint&     has_base [[buffer(7)]],
    constant uint&     has_bias [[buffer(8)]],
    uint2              tid      [[thread_position_in_grid]]
) {
    uint idx = tid.x;
    uint b   = tid.y;
    if (b >= B) return;

    if (has_base && idx < D_in) {
        half x = X[b * D_in + idx];
        X_silu[b * D_in + idx] = x / (1.0h + exp(-x));
    }

    if (idx < D_out) {
        Y[b * D_out + idx] = (has_bias && bias) ? bias[idx] : 0.0h;
    }
}

kernel void eval_cheby_basis_fp16(
    device const half* X       [[buffer(0)]], // [B, D_in]
    device half*       Phi     [[buffer(1)]], // [B, D_in * K]
    constant uint&     B       [[buffer(2)]],
    constant uint&     D_in    [[buffer(3)]],
    constant uint&     K       [[buffer(4)]],
    uint2              tid     [[thread_position_in_grid]]
) {
    uint b = tid.y;
    uint i = tid.x;
    if (b >= B || i >= D_in) return;

    float x = clamp(float(X[b * D_in + i]), -1.0f, 1.0f);

    uint base_idx = b * (D_in * K) + i * K;
    float t0 = 1.0f;
    float t1 = x;

    Phi[base_idx + 0] = half(t0);
    if (K > 1) Phi[base_idx + 1] = half(t1);

    for (uint k = 2; k < K; k++) {
        float t2 = fma(2.0f * x, t1, -t0);
        Phi[base_idx + k] = half(t2);
        t0 = t1;
        t1 = t2;
    }
}

kernel void eval_fastkan_rbf_basis_fp16(
    device const half* X                [[buffer(0)]], // [B, D_in]
    device const half* grid             [[buffer(1)]], // [K]
    device half*       Phi              [[buffer(2)]], // [B, D_in * K]
    constant uint&     B                [[buffer(3)]],
    constant uint&     D_in             [[buffer(4)]],
    constant uint&     K                [[buffer(5)]],
    constant float&    inv_denominator [[buffer(6)]],
    uint2              tid              [[thread_position_in_grid]]
) {
    uint b = tid.y;
    uint i = tid.x;
    if (b >= B || i >= D_in) return;

    half x = X[b * D_in + i];
    uint base_idx = b * (D_in * K) + i * K;
    half inv_den = half(inv_denominator);

    for (uint k = 0; k < K; k++) {
        half diff = x - grid[k];
        Phi[base_idx + k] = exp(-diff * diff * inv_den);
    }
}

kernel void eval_relu_basis_fp16(
    device const half* X                [[buffer(0)]], // [B, D_in]
    device const half* grid             [[buffer(1)]], // [K]
    device half*       Phi              [[buffer(2)]], // [B, D_in * K]
    constant uint&     B                [[buffer(3)]],
    constant uint&     D_in             [[buffer(4)]],
    constant uint&     K                [[buffer(5)]],
    constant float&    inv_denominator [[buffer(6)]],
    uint2              tid              [[thread_position_in_grid]]
) {
    uint b = tid.y;
    uint i = tid.x;
    if (b >= B || i >= D_in) return;

    half x = X[b * D_in + i];
    uint base_idx = b * (D_in * K) + i * K;
    half inv_den = half(inv_denominator);

    for (uint k = 0; k < K; k++) {
        half diff = abs(x - grid[k]) * inv_den;
        Phi[base_idx + k] = max(half(0.0h), 1.0h - diff);
    }
}

kernel void eval_bspline_basis_fp16(
    device const half*  X            [[buffer(0)]],
    device half*        Phi          [[buffer(1)]],
    constant uint&       B            [[buffer(2)]],
    constant uint&       D_in         [[buffer(3)]],
    constant uint&       grid_size    [[buffer(4)]],
    constant float&      grid_min     [[buffer(5)]],
    constant float&      inv_h        [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint b = gid.y;
    uint din = gid.x;
    if (b >= B || din >= D_in) return;
    half x = X[b * D_in + din];
    uint num_bases = grid_size + 3;
    uint out_offset = b * (D_in * num_bases) + din * num_bases;

    for (uint i = 0; i < num_bases; i++) {
        Phi[out_offset + i] = 0.0h;
    }

    float pos = (float(x) - grid_min) * inv_h;
    int span = clamp((int)floor(pos), 0, (int)grid_size - 1);
    float u = clamp(pos - (float)span, 0.0f, 1.0f);
    float one_sub_u = 1.0f - u;

    constexpr float inv6 = 0.16666667f;
    float b0 = (one_sub_u * one_sub_u * one_sub_u) * inv6;
    float b1 = (3.0f * u * u * u - 6.0f * u * u + 4.0f) * inv6;
    float b2 = (-3.0f * u * u * u + 3.0f * u * u + 3.0f * u + 1.0f) * inv6;
    float b3 = (u * u * u) * inv6;

    Phi[out_offset + span + 0] = half(b0);
    Phi[out_offset + span + 1] = half(b1);
    Phi[out_offset + span + 2] = half(b2);
    Phi[out_offset + span + 3] = half(b3);
}

kernel void eval_wavkan_basis_fp16(
    device const half* X            [[buffer(0)]], // [B, D_in]
    device const half* translation  [[buffer(1)]], // [D_in, num_wavelets]
    device const half* inv_scale    [[buffer(2)]], // [D_in, num_wavelets]
    device half*       Phi          [[buffer(3)]], // [B, D_in * num_wavelets]
    constant uint&     B            [[buffer(4)]],
    constant uint&     D_in         [[buffer(5)]],
    constant uint&     num_wavelets [[buffer(6)]],
    constant uint&     wavelet_type [[buffer(7)]],
    uint2              tid          [[thread_position_in_grid]]
) {
    uint b = tid.y;
    uint i = tid.x;
    if (b >= B || i >= D_in) return;

    half x = X[b * D_in + i];
    uint base_idx = b * (D_in * num_wavelets) + i * num_wavelets;

    for (uint w = 0; w < num_wavelets; w++) {
        half tr = translation[i * num_wavelets + w];
        half sc = inv_scale[i * num_wavelets + w];
        half z = (x - tr) * sc;
        half psi = 0.0h;

        if (wavelet_type == 0) { // Mexican Hat
            psi = (1.0h - z * z) * exp(-0.5h * z * z);
        } else if (wavelet_type == 1) { // Morlet
            psi = cos(1.75h * z) * exp(-0.5h * z * z);
        } else { // Derivative of Gaussian (DOG)
            psi = -z * exp(-0.5h * z * z);
        }

        Phi[base_idx + w] = psi;
    }
}

kernel void eval_fourier_basis_fp16(
    device const half* X          [[buffer(0)]], // [B, D_in]
    device half*       Phi        [[buffer(1)]], // [B, D_in * (2 * num_freqs + 1)]
    constant uint&     B          [[buffer(2)]],
    constant uint&     D_in       [[buffer(3)]],
    constant uint&     num_freqs  [[buffer(4)]],
    uint2              tid        [[thread_position_in_grid]]
) {
    uint b = tid.y;
    uint i = tid.x;
    if (b >= B || i >= D_in) return;

    half x = X[b * D_in + i];
    uint num_bases = 2 * num_freqs + 1;
    uint base_idx = b * (D_in * num_bases) + i * num_bases;

    Phi[base_idx + 0] = 1.0h;
    half pi_x = half(3.1415926535h) * x;

    for (uint k = 1; k <= num_freqs; k++) {
        half angle = half(k) * pi_x;
        half c, s;
        s = sincos(angle, c);
        Phi[base_idx + 2 * k - 1] = c;
        Phi[base_idx + 2 * k]     = s;
    }
}

kernel void eval_jacobi_basis_fp16(
    device const half*  X         [[buffer(0)]],
    device half*        Phi       [[buffer(1)]],
    constant uint&      B         [[buffer(2)]],
    constant uint&      D_in      [[buffer(3)]],
    constant uint&      degree    [[buffer(4)]],
    constant float&     alpha     [[buffer(5)]],
    constant float&     beta      [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint b = gid.y;
    uint din = gid.x;
    if (b >= B || din >= D_in) return;
    float x = clamp(float(X[b * D_in + din]), -1.0f, 1.0f);
    uint out_offset = b * (D_in * degree) + din * degree;
    float a_b = alpha + beta;

    float p_prev2 = 1.0f;
    Phi[out_offset + 0] = half(p_prev2);
    if (degree > 1) {
        float p_prev1 = 0.5f * (alpha - beta + (a_b + 2.0f) * x);
        Phi[out_offset + 1] = half(p_prev1);
        for (uint n = 2; n < degree; n++) {
            float fn = (float)n;
            float an = 2.0f * fn * (fn + a_b) * (2.0f * fn + a_b - 2.0f);
            float bn1 = (2.0f * fn + a_b - 1.0f) * (2.0f * fn + a_b) * (2.0f * fn + a_b - 2.0f);
            float bn2 = (2.0f * fn + a_b - 1.0f) * (alpha * alpha - beta * beta);
            float cn = 2.0f * (fn + alpha - 1.0f) * (fn + beta - 1.0f) * (2.0f * fn + a_b);

            float p_curr = (fma(fma(bn1, x, bn2), p_prev1, -cn * p_prev2)) / an;
            Phi[out_offset + n] = half(p_curr);
            p_prev2 = p_prev1;
            p_prev1 = p_curr;
        }
    }
}

kernel void combine_mult_nodes_fp16(
    device const half* internal_out [[buffer(0)]], // [B, num_add + 2 * num_mult]
    device half*       final_out    [[buffer(1)]], // [B, num_add + num_mult]
    constant uint&     B            [[buffer(2)]],
    constant uint&     num_add      [[buffer(3)]],
    constant uint&     num_mult     [[buffer(4)]],
    uint2              tid          [[thread_position_in_grid]]
) {
    uint b = tid.y;
    uint j = tid.x;
    if (b >= B) return;

    uint total_out = num_add + num_mult;
    if (j >= total_out) return;

    uint internal_width = num_add + 2 * num_mult;
    device const half* row_in = internal_out + b * internal_width;
    device half* row_out = final_out + b * total_out;

    if (j < num_add) {
        row_out[j] = row_in[j];
    } else {
        uint m_idx = j - num_add;
        half u = row_in[num_add + m_idx];
        half v = row_in[num_add + num_mult + m_idx];
        row_out[j] = u * v;
    }
}

// Backward gradient and basis derivative compute kernels.

// Evaluates silu(X) into X_silu buffer: silu(x) = x / (1 + exp(-x))
kernel void eval_silu(
    device const float*  X       [[buffer(0)]],
    device float*        X_silu  [[buffer(1)]],
    constant uint&       B       [[buffer(2)]],
    constant uint&       D_in    [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint r = gid.y;
    uint c = gid.x;
    if (r >= B || c >= D_in) return;
    float x = X[r * D_in + c];
    float inv_ln2 = 1.4426950408889634f;
    X_silu[r * D_in + c] = x / (1.0f + exp2(-inv_ln2 * x));
}

// Column-wise reduction: dbias[c] = sum_{b=0}^{B-1} dY[b, c]
kernel void reduce_sum_columns(
    device const float* dY     [[buffer(0)]], // [B, D_out]
    device float*       dbias  [[buffer(1)]], // [D_out]
    constant uint&      B      [[buffer(2)]],
    constant uint&      D_out  [[buffer(3)]],
    uint c [[thread_position_in_grid]]
) {
    if (c >= D_out) return;
    float sum = 0.0f;
    for (uint b = 0; b < B; b++) {
        sum += dY[b * D_out + c];
    }
    dbias[c] = sum;
}

// Backward Chebyshev basis: dX = sum_k (dPhi_k * T'_k(x)) + (has_base ? dX_silu * silu'(x) : 0)
kernel void backward_cheby_basis(
    device const float*  X         [[buffer(0)]], // [B, D_in]
    device const float*  dPhi      [[buffer(1)]], // [B, D_in * deg]
    device const float*  dX_silu   [[buffer(2)]], // [B, D_in] (or unused if !has_base)
    device float*        dX        [[buffer(3)]], // [B, D_in]
    constant uint&       B         [[buffer(4)]],
    constant uint&       D_in      [[buffer(5)]],
    constant uint&       deg       [[buffer(6)]],
    constant uint&       has_base  [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint b = gid.y;
    uint din = gid.x;
    if (b >= B || din >= D_in) return;

    float raw_x = X[b * D_in + din];
    float x = clamp(raw_x, -1.0f, 1.0f);
    uint phi_offset = b * (D_in * deg) + din * deg;

    // Evaluate Chebyshev derivatives T'_k(x)
    // Recurrence: T'_k(x) = 2*T_{k-1}(x) + 2*x*T'_{k-1}(x) - T'_{k-2}(x)
    float grad_basis = 0.0f;
    if (deg > 1) {
        // k = 0: T_0 = 1, T'_0 = 0 -> contribution = 0
        // k = 1: T_1 = x, T'_1 = 1
        float t_prev2 = 1.0f; // T_0
        float t_prev1 = x;    // T_1
        float dt_prev2 = 0.0f; // T'_0
        float dt_prev1 = 1.0f; // T'_1

        grad_basis += dPhi[phi_offset + 1] * dt_prev1;

        for (uint k = 2; k < deg; k++) {
            float t_curr = 2.0f * x * t_prev1 - t_prev2;
            float dt_curr = 2.0f * t_prev1 + 2.0f * x * dt_prev1 - dt_prev2;

            grad_basis += dPhi[phi_offset + k] * dt_curr;

            t_prev2 = t_prev1;
            t_prev1 = t_curr;
            dt_prev2 = dt_prev1;
            dt_prev1 = dt_curr;
        }
    }

    // Derivative of clamp(raw_x, -1.0, 1.0) is 1.0 if inside (-1, 1), 0 otherwise
    if (raw_x < -1.0f || raw_x > 1.0f) {
        grad_basis = 0.0f;
    }

    float total_dx = grad_basis;

    if (has_base) {
        // silu(x) = x * sigmoid(x)
        // d/dx silu(x) = sigmoid(x) * (1.0 + x * (1.0 - sigmoid(x)))
        float sig = 1.0f / (1.0f + exp(-raw_x));
        float d_silu = sig * (1.0f + raw_x * (1.0f - sig));
        total_dx += dX_silu[b * D_in + din] * d_silu;
    }

    dX[b * D_in + din] = total_dx;
}

// Backward FastKAN RBF basis:
// Phi_k(x) = exp(-inv_den * (x - c_k)^2)
// dPhi_k / dx = -2 * (x - c_k) * inv_den * Phi_k(x)
kernel void backward_fastkan_rbf_basis(
    device const float*  X         [[buffer(0)]], // [B, D_in]
    device const float*  grid      [[buffer(1)]], // [K]
    device const float*  dPhi      [[buffer(2)]], // [B, D_in * K]
    device const float*  dX_silu   [[buffer(3)]], // [B, D_in]
    device float*        dX        [[buffer(4)]], // [B, D_in]
    constant uint&       B         [[buffer(5)]],
    constant uint&       D_in      [[buffer(6)]],
    constant uint&       K         [[buffer(7)]],
    constant float&      inv_d     [[buffer(8)]],
    constant uint&       has_base  [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint b = gid.y;
    uint din = gid.x;
    if (b >= B || din >= D_in) return;

    float x = X[b * D_in + din];
    uint phi_offset = b * (D_in * K) + din * K;
    float inv_ln2 = 1.4426950408889634f;

    float grad_basis = 0.0f;
    for (uint k = 0; k < K; k++) {
        float diff = x - grid[k];
        float phi_val = exp2(-inv_ln2 * (diff * diff) * inv_d);
        float dphi_dx = -2.0f * diff * inv_d * phi_val;
        grad_basis += dPhi[phi_offset + k] * dphi_dx;
    }

    float total_dx = grad_basis;
    if (has_base) {
        float sig = 1.0f / (1.0f + exp(-x));
        float d_silu = sig * (1.0f + x * (1.0f - sig));
        total_dx += dX_silu[b * D_in + din] * d_silu;
    }

    dX[b * D_in + din] = total_dx;
}

// Backward ReLUKAN tent basis:
// Phi_k(x) = max(0, 1 - |x - c_k| * inv_h)
// dPhi_k / dx = (1 - |x - c_k| * inv_h > 0) ? (-inv_h * sgn(x - c_k)) : 0
kernel void backward_relu_basis(
    device const float*  X         [[buffer(0)]], // [B, D_in]
    device const float*  grid      [[buffer(1)]], // [G]
    device const float*  dPhi      [[buffer(2)]], // [B, D_in * G]
    device const float*  dX_silu   [[buffer(3)]], // [B, D_in]
    device float*        dX        [[buffer(4)]], // [B, D_in]
    constant uint&       B         [[buffer(5)]],
    constant uint&       D_in      [[buffer(6)]],
    constant uint&       G         [[buffer(7)]],
    constant float&      inv_h     [[buffer(8)]],
    constant uint&       has_base  [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint b = gid.y;
    uint din = gid.x;
    if (b >= B || din >= D_in) return;

    float x = X[b * D_in + din];
    uint phi_offset = b * (D_in * G) + din * G;

    float grad_basis = 0.0f;
    for (uint g = 0; g < G; g++) {
        float diff = x - grid[g];
        float val = 1.0f - fabs(diff) * inv_h;
        if (val > 0.0f) {
            float sgn = (diff > 0.0f) ? 1.0f : ((diff < 0.0f) ? -1.0f : 0.0f);
            float dphi_dx = -sgn * inv_h;
            grad_basis += dPhi[phi_offset + g] * dphi_dx;
        }
    }

    float total_dx = grad_basis;
    if (has_base) {
        float sig = 1.0f / (1.0f + exp(-x));
        float d_silu = sig * (1.0f + x * (1.0f - sig));
        total_dx += dX_silu[b * D_in + din] * d_silu;
    }

    dX[b * D_in + din] = total_dx;
}

// Backward B-Spline basis:
// Closed-form derivative of cubic B-spline polynomials
kernel void backward_bspline_basis(
    device const float*  X         [[buffer(0)]], // [B, D_in]
    device const float*  dPhi      [[buffer(1)]], // [B, D_in * num_bases]
    device const float*  dX_silu   [[buffer(2)]], // [B, D_in]
    device float*        dX        [[buffer(3)]], // [B, D_in]
    constant uint&       B         [[buffer(4)]],
    constant uint&       D_in      [[buffer(5)]],
    constant uint&       grid_size [[buffer(6)]],
    constant float&      grid_min  [[buffer(7)]],
    constant float&      inv_h     [[buffer(8)]],
    constant uint&       has_base  [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint b = gid.y;
    uint din = gid.x;
    if (b >= B || din >= D_in) return;

    float x = X[b * D_in + din];
    uint num_bases = grid_size + 3;
    uint phi_offset = b * (D_in * num_bases) + din * num_bases;

    float pos = (x - grid_min) * inv_h;
    int span = clamp((int)floor(pos), 0, (int)grid_size - 1);
    float u = clamp(pos - (float)span, 0.0f, 1.0f);
    float one_sub_u = 1.0f - u;

    // Cubic B-spline derivatives with respect to u:
    // b0(u) = (1 - u)^3 / 6 -> b0'(u) = -0.5 * (1 - u)^2
    // b1(u) = (3u^3 - 6u^2 + 4) / 6 -> b1'(u) = (9u^2 - 12u) / 6 = 1.5 * u^2 - 2u
    // b2(u) = (-3u^3 + 3u^2 + 3u + 1) / 6 -> b2'(u) = (-9u^2 + 6u + 3) / 6 = -1.5 * u^2 + u + 0.5
    // b3(u) = u^3 / 6 -> b3'(u) = 0.5 * u^2
    // d/dx = d/du * du/dx = d/du * inv_h
    float db0 = -0.5f * one_sub_u * one_sub_u * inv_h;
    float db1 = (1.5f * u * u - 2.0f * u) * inv_h;
    float db2 = (-1.5f * u * u + u + 0.5f) * inv_h;
    float db3 = 0.5f * u * u * inv_h;

    float grad_basis = dPhi[phi_offset + span + 0] * db0 +
                       dPhi[phi_offset + span + 1] * db1 +
                       dPhi[phi_offset + span + 2] * db2 +
                       dPhi[phi_offset + span + 3] * db3;

    float total_dx = grad_basis;
    if (has_base) {
        float sig = 1.0f / (1.0f + exp(-x));
        float d_silu = sig * (1.0f + x * (1.0f - sig));
        total_dx += dX_silu[b * D_in + din] * d_silu;
    }

    dX[b * D_in + din] = total_dx;
}

// Training optimization kernels: MSE loss gradient and native GPU optimizers.

kernel void mse_loss_backward(
    device const float*  pred          [[buffer(0)]],
    device const float*  target        [[buffer(1)]],
    device float*        dY            [[buffer(2)]],
    constant uint&       total_elems   [[buffer(3)]],
    constant float&      scale         [[buffer(4)]],
    uint gid                           [[thread_position_in_grid]]
) {
    if (gid >= total_elems) return;
    dY[gid] = (pred[gid] - target[gid]) * scale;
}

kernel void adamw_step(
    device float*        param         [[buffer(0)]],
    device const float*  grad          [[buffer(1)]],
    device float*        m             [[buffer(2)]],
    device float*        v             [[buffer(3)]],
    constant float&      lr            [[buffer(4)]],
    constant float&      beta1         [[buffer(5)]],
    constant float&      beta2         [[buffer(6)]],
    constant float&      eps           [[buffer(7)]],
    constant float&      weight_decay  [[buffer(8)]],
    constant float&      lr_t          [[buffer(9)]],
    constant uint&       n_elems       [[buffer(10)]],
    uint gid                           [[thread_position_in_grid]]
) {
    if (gid >= n_elems) return;
    float p = param[gid];
    float g = grad[gid];
    if (weight_decay > 0.0f) {
        p -= lr * weight_decay * p;
    }
    float m_val = beta1 * m[gid] + (1.0f - beta1) * g;
    float v_val = beta2 * v[gid] + (1.0f - beta2) * (g * g);
    m[gid] = m_val;
    v[gid] = v_val;
    param[gid] = p - lr_t * (m_val / (sqrt(v_val) + eps));
}

kernel void sgd_step(
    device float*        param         [[buffer(0)]],
    device const float*  grad          [[buffer(1)]],
    device float*        velocity      [[buffer(2)]],
    constant float&      lr            [[buffer(3)]],
    constant float&      momentum      [[buffer(4)]],
    constant float&      weight_decay  [[buffer(5)]],
    constant uint&       nesterov      [[buffer(6)]],
    constant uint&       n_elems       [[buffer(7)]],
    uint gid                           [[thread_position_in_grid]]
) {
    if (gid >= n_elems) return;
    float p = param[gid];
    float g = grad[gid];
    if (weight_decay > 0.0f) {
        g += weight_decay * p;
    }
    float v_val = momentum * velocity[gid] + g;
    velocity[gid] = v_val;
    float update = (nesterov != 0) ? (g + momentum * v_val) : v_val;
    param[gid] = p - lr * update;
}

kernel void lion_step(
    device float*        param         [[buffer(0)]],
    device const float*  grad          [[buffer(1)]],
    device float*        m             [[buffer(2)]],
    constant float&      lr            [[buffer(3)]],
    constant float&      beta1         [[buffer(4)]],
    constant float&      beta2         [[buffer(5)]],
    constant float&      weight_decay  [[buffer(6)]],
    constant uint&       n_elems       [[buffer(7)]],
    uint gid                           [[thread_position_in_grid]]
) {
    if (gid >= n_elems) return;
    float p = param[gid];
    if (weight_decay > 0.0f) {
        p -= lr * weight_decay * p;
    }
    float g = grad[gid];
    float m_val = m[gid];
    float u = beta1 * m_val + (1.0f - beta1) * g;
    float s = (u > 0.0f) ? 1.0f : ((u < 0.0f) ? -1.0f : 0.0f);
    param[gid] = p - lr * s;
    m[gid] = beta2 * m_val + (1.0f - beta2) * g;
}

kernel void rmsprop_step(
    device float*        param         [[buffer(0)]],
    device const float*  grad          [[buffer(1)]],
    device float*        v             [[buffer(2)]],
    device float*        buf           [[buffer(3)]],
    constant float&      lr            [[buffer(4)]],
    constant float&      alpha         [[buffer(5)]],
    constant float&      eps           [[buffer(6)]],
    constant float&      weight_decay  [[buffer(7)]],
    constant float&      momentum      [[buffer(8)]],
    constant uint&       has_momentum  [[buffer(9)]],
    constant uint&       n_elems       [[buffer(10)]],
    uint gid                           [[thread_position_in_grid]]
) {
    if (gid >= n_elems) return;
    float p = param[gid];
    float g = grad[gid];
    if (weight_decay > 0.0f) {
        g += weight_decay * p;
    }
    float v_val = alpha * v[gid] + (1.0f - alpha) * (g * g);
    v[gid] = v_val;
    float avg = g / (sqrt(v_val) + eps);
    if (has_momentum != 0) {
        float b = momentum * buf[gid] + avg;
        buf[gid] = b;
        param[gid] = p - lr * b;
    } else {
        param[gid] = p - lr * avg;
    }
}

// Muon optimizer: GPU Newton-Schulz 5 iteration kernel.

kernel void matrix_transpose_f32(
    device const float* in    [[buffer(0)]],
    device float*       out   [[buffer(1)]],
    constant uint&      rows  [[buffer(2)]],
    constant uint&      cols  [[buffer(3)]],
    uint2               gid   [[thread_position_in_grid]]
) {
    uint r = gid.y;
    uint c = gid.x;
    if (r >= rows || c >= cols) return;
    out[c * rows + r] = in[r * cols + c];
}

kernel void matrix_scale_f32(
    device float*        matrix   [[buffer(0)]],
    constant float&      scale    [[buffer(1)]],
    constant uint&       n_elems  [[buffer(2)]],
    uint                 gid      [[thread_position_in_grid]]
) {
    if (gid >= n_elems) return;
    matrix[gid] *= scale;
}

kernel void ns5_combine_B_f32(
    device const float* A        [[buffer(0)]], // [M, M]
    device const float* A2       [[buffer(1)]], // [M, M]
    device float*       B        [[buffer(2)]], // [M, M]
    constant float&     b_coeff  [[buffer(3)]],
    constant float&     c_coeff  [[buffer(4)]],
    constant uint&      n_elems  [[buffer(5)]],
    uint                gid      [[thread_position_in_grid]]
) {
    if (gid >= n_elems) return;
    B[gid] = fma(c_coeff, A2[gid], b_coeff * A[gid]);
}

kernel void ns5_combine_X_f32(
    device float*       X        [[buffer(0)]], // [M, N]
    device const float* BX       [[buffer(1)]], // [M, N]
    constant float&     a_coeff  [[buffer(2)]],
    constant uint&      n_elems  [[buffer(3)]],
    uint                gid      [[thread_position_in_grid]]
) {
    if (gid >= n_elems) return;
    X[gid] = fma(a_coeff, X[gid], BX[gid]);
}

// Fused backward Chebyshev kernel with streaming batch accumulation.

kernel void fused_backward_cheby_dW(
    device const float* dY       [[buffer(0)]],  // [B, D_out]
    device const float* X        [[buffer(1)]],  // [B, D_in]
    device float*       dW_cheby [[buffer(2)]],  // [D_out, D_in * degree]
    constant uint&      B        [[buffer(3)]],
    constant uint&      D_in     [[buffer(4)]],
    constant uint&      D_out    [[buffer(5)]],
    constant uint&      degree   [[buffer(6)]],
    uint2               gid      [[thread_position_in_grid]]
) {
    uint j = gid.y; // 0..D_out-1
    uint i = gid.x; // 0..D_in-1
    if (j >= D_out || i >= D_in) return;

    float dW_accum[16];
    for (uint k = 0; k < degree && k < 16; k++) {
        dW_accum[k] = 0.0f;
    }

    uint K = D_in * degree;
    uint out_col_base = i * degree;

    for (uint b = 0; b < B; b++) {
        float dy_val = dY[b * D_out + j];
        float x_val = clamp(X[b * D_in + i], -1.0f, 1.0f);

        float t0 = 1.0f;
        float t1 = x_val;

        dW_accum[0] = fma(dy_val, t0, dW_accum[0]);
        if (degree > 1) {
            dW_accum[1] = fma(dy_val, t1, dW_accum[1]);
        }

        for (uint k = 2; k < degree && k < 16; k++) {
            float t2 = fma(2.0f * x_val, t1, -t0);
            dW_accum[k] = fma(dy_val, t2, dW_accum[k]);
            t0 = t1;
            t1 = t2;
        }
    }

    for (uint k = 0; k < degree && k < 16; k++) {
        dW_cheby[j * K + out_col_base + k] = dW_accum[k];
    }
}

kernel void fused_backward_cheby_dX(
    device const float* dY       [[buffer(0)]],  // [B, D_out]
    device const float* X        [[buffer(1)]],  // [B, D_in]
    device const float* W_cheby  [[buffer(2)]],  // [D_out, D_in * degree]
    device const float* W_base   [[buffer(3)]],  // [D_out, D_in] (optional)
    device float*       dX       [[buffer(4)]],  // [B, D_in]
    constant uint&      B        [[buffer(5)]],
    constant uint&      D_in     [[buffer(6)]],
    constant uint&      D_out    [[buffer(7)]],
    constant uint&      degree   [[buffer(8)]],
    constant uint&      has_base [[buffer(9)]],
    uint2               gid      [[thread_position_in_grid]]
) {
    uint b = gid.y;
    uint i = gid.x;
    if (b >= B || i >= D_in) return;

    float raw_x = X[b * D_in + i];
    float x_val = clamp(raw_x, -1.0f, 1.0f);

    uint K = D_in * degree;
    uint col_base = i * degree;

    float dx_accum = 0.0f;

    if (degree > 1 && raw_x >= -1.0f && raw_x <= 1.0f) {
        float T_prev2 = 1.0f, T_prev1 = x_val;
        float dT_prev2 = 0.0f, dT_prev1 = 1.0f;

        float dphi_1 = 0.0f;
        for (uint j = 0; j < D_out; j++) {
            dphi_1 = fma(dY[b * D_out + j], W_cheby[j * K + col_base + 1], dphi_1);
        }
        dx_accum = fma(dphi_1, dT_prev1, dx_accum);

        for (uint k = 2; k < degree; k++) {
            float T_curr = fma(2.0f * x_val, T_prev1, -T_prev2);
            float dT_curr = fma(2.0f * x_val, dT_prev1, fma(2.0f, T_prev1, -dT_prev2));

            float dphi_k = 0.0f;
            for (uint j = 0; j < D_out; j++) {
                dphi_k = fma(dY[b * D_out + j], W_cheby[j * K + col_base + k], dphi_k);
            }
            dx_accum = fma(dphi_k, dT_curr, dx_accum);

            T_prev2 = T_prev1;
            T_prev1 = T_curr;
            dT_prev2 = dT_prev1;
            dT_prev1 = dT_curr;
        }
    }

    if (has_base != 0) {
        float sig = 1.0f / (1.0f + exp(-raw_x));
        float d_silu = sig * (1.0f + raw_x * (1.0f - sig));
        float dbase = 0.0f;
        for (uint j = 0; j < D_out; j++) {
            dbase = fma(dY[b * D_out + j], W_base[j * D_in + i], dbase);
        }
        dx_accum = fma(dbase, d_silu, dx_accum);
    }

    dX[b * D_in + i] = dx_accum;
}

// Sub-4-bit quantization kernels: Ternary (1.58-bit) and INT2.

kernel void kan_forward_ternary_cheby(
    device const float*  X        [[buffer(0)]], // [B, D_in]
    device const uint*   W_packed [[buffer(1)]], // [D_out, (D_in * degree + 15) / 16]
    device const float*  scales   [[buffer(2)]], // [D_out, (D_in * degree + group_size - 1) / group_size]
    device const float*  bias     [[buffer(3)]], // [D_out] (optional)
    device float*        Y        [[buffer(4)]], // [B, D_out]
    constant uint&       B        [[buffer(5)]],
    constant uint&       D_in     [[buffer(6)]],
    constant uint&       D_out    [[buffer(7)]],
    constant uint&       degree   [[buffer(8)]],
    constant uint&       group_sz [[buffer(9)]],
    constant uint&       has_bias [[buffer(10)]],
    uint2                gid      [[thread_position_in_grid]]
) {
    uint b = gid.y;
    uint j = gid.x;
    if (b >= B || j >= D_out) return;

    uint K_dim = D_in * degree;
    uint words_per_row = (K_dim + 15) / 16;
    device const uint* w_row = W_packed + j * words_per_row;
    uint groups_per_row = (K_dim + group_sz - 1) / group_sz;
    device const float* s_row = scales + j * groups_per_row;

    float acc = (has_bias != 0 && bias != nullptr) ? bias[j] : 0.0f;

    for (uint i = 0; i < D_in; i++) {
        float x_val = clamp(X[b * D_in + i], -1.0f, 1.0f);
        float t0 = 1.0f;
        float t1 = x_val;

        for (uint k = 0; k < degree; k++) {
            float phi_k;
            if (k == 0) phi_k = t0;
            else if (k == 1) phi_k = t1;
            else {
                float t2 = fma(2.0f * x_val, t1, -t0);
                phi_k = t2;
                t0 = t1;
                t1 = t2;
            }

            uint flat_idx = i * degree + k;
            uint word_idx = flat_idx / 16;
            uint shift = (flat_idx % 16) * 2;
            uint code = (w_row[word_idx] >> shift) & 0x3;

            float w_val = float(code) - 1.0f;
            if (code < 3) {
                float sc = s_row[flat_idx / group_sz];
                acc = fma(w_val * sc, phi_k, acc);
            }
        }
    }

    Y[b * D_out + j] = acc;
}

kernel void kan_forward_int2_cheby(
    device const float*  X         [[buffer(0)]], // [B, D_in]
    device const uint*   W_packed  [[buffer(1)]], // [D_out, (D_in * degree + 15) / 16]
    device const float*  scales    [[buffer(2)]], // [D_out, (D_in * degree + group_size - 1) / group_size]
    device const float*  codebook  [[buffer(3)]], // [4] Lloyd-Max or uniform centroids
    device const float*  bias      [[buffer(4)]], // [D_out] (optional)
    device float*        Y         [[buffer(5)]], // [B, D_out]
    constant uint&       B         [[buffer(6)]],
    constant uint&       D_in      [[buffer(7)]],
    constant uint&       D_out     [[buffer(8)]],
    constant uint&       degree    [[buffer(9)]],
    constant uint&       group_sz  [[buffer(10)]],
    constant uint&       has_bias  [[buffer(11)]],
    uint2                gid       [[thread_position_in_grid]]
) {
    uint b = gid.y;
    uint j = gid.x;
    if (b >= B || j >= D_out) return;

    uint K_dim = D_in * degree;
    uint words_per_row = (K_dim + 15) / 16;
    device const uint* w_row = W_packed + j * words_per_row;
    uint groups_per_row = (K_dim + group_sz - 1) / group_sz;
    device const float* s_row = scales + j * groups_per_row;

    float acc = (has_bias != 0 && bias != nullptr) ? bias[j] : 0.0f;

    for (uint i = 0; i < D_in; i++) {
        float x_val = clamp(X[b * D_in + i], -1.0f, 1.0f);
        float t0 = 1.0f;
        float t1 = x_val;

        for (uint k = 0; k < degree; k++) {
            float phi_k;
            if (k == 0) phi_k = t0;
            else if (k == 1) phi_k = t1;
            else {
                float t2 = fma(2.0f * x_val, t1, -t0);
                phi_k = t2;
                t0 = t1;
                t1 = t2;
            }

            uint flat_idx = i * degree + k;
            uint word_idx = flat_idx / 16;
            uint shift = (flat_idx % 16) * 2;
            uint code = (w_row[word_idx] >> shift) & 0x3;

            float w_val = codebook[code];
            float sc = s_row[flat_idx / group_sz];
            acc = fma(w_val * sc, phi_k, acc);
        }
    }

    Y[b * D_out + j] = acc;
}

// Element-wise and activation operations.

kernel void swiglu_forward_f32(
    device const float* gate    [[buffer(0)]], // [B, D]
    device const float* up      [[buffer(1)]], // [B, D]
    device float*       out     [[buffer(2)]], // [B, D]
    constant uint&      n_elems [[buffer(3)]],
    uint                gid     [[thread_position_in_grid]]
) {
    if (gid >= n_elems) return;
    float g = gate[gid];
    float u = up[gid];
    float sig = 1.0f / (1.0f + exp(-clamp(g, -20.0f, 20.0f)));
    out[gid] = (g * sig) * u;
}

kernel void swiglu_backward_f32(
    device const float* d_hidden [[buffer(0)]], // [B, D]
    device const float* gate     [[buffer(1)]], // [B, D]
    device const float* up       [[buffer(2)]], // [B, D]
    device float*       d_gate   [[buffer(3)]], // [B, D]
    device float*       d_up     [[buffer(4)]], // [B, D]
    constant uint&      n_elems  [[buffer(5)]],
    uint                gid      [[thread_position_in_grid]]
) {
    if (gid >= n_elems) return;
    float dh = d_hidden[gid];
    float g = gate[gid];
    float u = up[gid];
    float sig = 1.0f / (1.0f + exp(-clamp(g, -20.0f, 20.0f)));
    float silu_g = g * sig;
    float d_silu_g = sig * (1.0f + g * (1.0f - sig));
    d_up[gid] = dh * silu_g;
    d_gate[gid] = dh * u * d_silu_g;
}

kernel void elementwise_add_f32(
    device const float* a       [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint&      n_elems [[buffer(3)]],
    uint                gid     [[thread_position_in_grid]]
) {
    if (gid >= n_elems) return;
    out[gid] = a[gid] + b[gid];
}

kernel void adam_coupled_step(
    device float*        param         [[buffer(0)]],
    device const float*  grad          [[buffer(1)]],
    device float*        m             [[buffer(2)]],
    device float*        v             [[buffer(3)]],
    constant float&      lr            [[buffer(4)]],
    constant float&      beta1         [[buffer(5)]],
    constant float&      beta2         [[buffer(6)]],
    constant float&      eps           [[buffer(7)]],
    constant float&      weight_decay  [[buffer(8)]],
    constant float&      lr_t          [[buffer(9)]],
    constant uint&       n_elems       [[buffer(10)]],
    uint                 gid           [[thread_position_in_grid]]
) {
    if (gid >= n_elems) return;
    float p = param[gid];
    float g = grad[gid];
    if (weight_decay > 0.0f) {
        g += weight_decay * p;
    }
    float m_val = beta1 * m[gid] + (1.0f - beta1) * g;
    float v_val = beta2 * v[gid] + (1.0f - beta2) * (g * g);
    m[gid] = m_val;
    v[gid] = v_val;
    param[gid] = p - lr_t * (m_val / (sqrt(v_val) + eps));
}

kernel void muon_momentum_update(
    device const float* param        [[buffer(0)]],
    device const float* grad         [[buffer(1)]],
    device float*       v            [[buffer(2)]],
    device float*       update       [[buffer(3)]],
    constant float&     momentum     [[buffer(4)]],
    constant float&     weight_decay [[buffer(5)]],
    constant uint&      nesterov     [[buffer(6)]],
    constant uint&      n_elems      [[buffer(7)]],
    uint                gid          [[thread_position_in_grid]]
) {
    if (gid >= n_elems) return;
    float g = grad[gid];
    if (weight_decay > 0.0f) {
        g += weight_decay * param[gid];
    }
    float v_val = momentum * v[gid] + (1.0f - momentum) * g;
    v[gid] = v_val;
    if (nesterov != 0) {
        update[gid] = (1.0f - momentum) * g + momentum * v_val;
    } else {
        update[gid] = v_val;
    }
}

kernel void muon_param_update(
    device float*       param    [[buffer(0)]],
    device const float* update   [[buffer(1)]],
    constant float&     step_lr  [[buffer(2)]],
    constant uint&      n_elems  [[buffer(3)]],
    uint                gid      [[thread_position_in_grid]]
) {
    if (gid >= n_elems) return;
    param[gid] -= step_lr * update[gid];
}

kernel void dequantize_ternary_fast(
    device const uint*   packed      [[buffer(0)]],
    device const float*  scales      [[buffer(1)]],
    device float*        out         [[buffer(2)]],
    constant uint&       D_out       [[buffer(3)]],
    constant uint&       K_dim       [[buffer(4)]],
    constant uint&       group_size  [[buffer(5)]],
    uint2                gid         [[thread_position_in_grid]]
) {
    uint j = gid.y;
    uint k = gid.x;
    if (j >= D_out || k >= K_dim) return;

    uint words_per_row = (K_dim + 15) / 16;
    uint groups_per_row = (K_dim + group_size - 1) / group_size;

    uint word_idx = k / 16;
    uint shift = (k % 16) * 2;
    uint code = (packed[j * words_per_row + word_idx] >> shift) & 0x3;
    float w_val = float(code) - 1.0f;
    if (code == 3) w_val = 0.0f;

    float sc = scales[j * groups_per_row + (k / group_size)];
    out[j * K_dim + k] = w_val * sc;
}

kernel void dequantize_int2_fast(
    device const uint*   packed      [[buffer(0)]],
    device const float*  scales      [[buffer(1)]],
    device const float*  codebook    [[buffer(2)]],
    device float*        out         [[buffer(3)]],
    constant uint&       D_out       [[buffer(4)]],
    constant uint&       K_dim       [[buffer(5)]],
    constant uint&       group_size  [[buffer(6)]],
    uint2                gid         [[thread_position_in_grid]]
) {
    uint j = gid.y;
    uint k = gid.x;
    if (j >= D_out || k >= K_dim) return;

    uint words_per_row = (K_dim + 15) / 16;
    uint groups_per_row = (K_dim + group_size - 1) / group_size;

    uint word_idx = k / 16;
    uint shift = (k % 16) * 2;
    uint code = (packed[j * words_per_row + word_idx] >> shift) & 0x3;

    float sc = scales[j * groups_per_row + (k / group_size)];
    out[j * K_dim + k] = codebook[code] * sc;
}

kernel void calc_mse_loss_backward(
    device const float* pred    [[buffer(0)]],
    device const float* target  [[buffer(1)]],
    device float*       dY      [[buffer(2)]],
    constant uint&      n_elems [[buffer(3)]],
    constant float&     scale   [[buffer(4)]],
    uint                gid     [[thread_position_in_grid]]
) {
    if (gid >= n_elems) return;
    dY[gid] = scale * (pred[gid] - target[gid]);
}





