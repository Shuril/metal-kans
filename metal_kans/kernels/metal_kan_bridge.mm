#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Accelerate/Accelerate.h>
#include <mach/mach_time.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLComputePipelineState> g_pipe_cheby_tiled = nil;
static id<MTLComputePipelineState> g_pipe_fastkan_tiled = nil;
static id<MTLComputePipelineState> g_pipe_relu_tiled = nil;
static id<MTLComputePipelineState> g_pipe_wavkan_tiled = nil;
static id<MTLComputePipelineState> g_pipe_fourier_tiled = nil;
static id<MTLComputePipelineState> g_pipe_jacobi_tiled = nil;
static id<MTLComputePipelineState> g_pipe_rational_tiled = nil;
static id<MTLComputePipelineState> g_pipe_bspline_tiled = nil;

static id<MTLComputePipelineState> g_pipe_prep = nil;
static id<MTLComputePipelineState> g_pipe_rbf_basis = nil;
static id<MTLComputePipelineState> g_pipe_wav_basis = nil;
static id<MTLComputePipelineState> g_pipe_cheby_basis = nil;
static id<MTLComputePipelineState> g_pipe_bspline_basis = nil;
static id<MTLComputePipelineState> g_pipe_relu_basis = nil;
static id<MTLComputePipelineState> g_pipe_fourier_basis = nil;
static id<MTLComputePipelineState> g_pipe_jacobi_basis = nil;
static id<MTLComputePipelineState> g_pipe_combine_mult = nil;
static id<MTLComputePipelineState> g_pipe_reduce_sum_cols = nil;
static id<MTLComputePipelineState> g_pipe_eval_silu = nil;
static id<MTLComputePipelineState> g_pipe_backward_cheby = nil;
static id<MTLComputePipelineState> g_pipe_backward_rbf = nil;
static id<MTLComputePipelineState> g_pipe_backward_relu = nil;
static id<MTLComputePipelineState> g_pipe_backward_bspline = nil;

// Direct SIMDgroup matrix & FP16 pipelines
static id<MTLComputePipelineState> g_pipe_gemm_simd_fp32 = nil;
static id<MTLComputePipelineState> g_pipe_lowrank_simd_fp32 = nil;
static id<MTLComputePipelineState> g_pipe_prep_fp16 = nil;
static id<MTLComputePipelineState> g_pipe_cheby_basis_fp16 = nil;
static id<MTLComputePipelineState> g_pipe_rbf_basis_fp16 = nil;
static id<MTLComputePipelineState> g_pipe_relu_basis_fp16 = nil;
static id<MTLComputePipelineState> g_pipe_bspline_basis_fp16 = nil;
static id<MTLComputePipelineState> g_pipe_wav_basis_fp16 = nil;
static id<MTLComputePipelineState> g_pipe_fourier_basis_fp16 = nil;
static id<MTLComputePipelineState> g_pipe_jacobi_basis_fp16 = nil;
static id<MTLComputePipelineState> g_pipe_combine_mult_fp16 = nil;

// Training & Optimizer pipelines
static id<MTLComputePipelineState> g_pipe_mse_loss_backward = nil;
static id<MTLComputePipelineState> g_pipe_adamw_step = nil;
static id<MTLComputePipelineState> g_pipe_sgd_step = nil;
static id<MTLComputePipelineState> g_pipe_lion_step = nil;
static id<MTLComputePipelineState> g_pipe_rmsprop_step = nil;

// GPU Muon, Fused Backward & Sub-4-bit pipelines
static id<MTLComputePipelineState> g_pipe_matrix_transpose_f32 = nil;
static id<MTLComputePipelineState> g_pipe_matrix_scale_f32 = nil;
static id<MTLComputePipelineState> g_pipe_ns5_combine_B_f32 = nil;
static id<MTLComputePipelineState> g_pipe_ns5_combine_X_f32 = nil;
static id<MTLComputePipelineState> g_pipe_fused_backward_cheby_dW = nil;
static id<MTLComputePipelineState> g_pipe_fused_backward_cheby_dX = nil;
static id<MTLComputePipelineState> g_pipe_kan_forward_ternary_cheby = nil;
static id<MTLComputePipelineState> g_pipe_kan_forward_int2_cheby = nil;

// High-performance element-wise, activation & optimizer pipelines
static id<MTLComputePipelineState> g_pipe_swiglu_forward = nil;
static id<MTLComputePipelineState> g_pipe_swiglu_backward = nil;
static id<MTLComputePipelineState> g_pipe_elementwise_add = nil;
static id<MTLComputePipelineState> g_pipe_adam_coupled_step = nil;
static id<MTLComputePipelineState> g_pipe_muon_momentum_update = nil;
static id<MTLComputePipelineState> g_pipe_muon_param_update = nil;
static id<MTLComputePipelineState> g_pipe_dequantize_ternary_fast = nil;
static id<MTLComputePipelineState> g_pipe_dequantize_int2_fast = nil;
static id<MTLComputePipelineState> g_pipe_calc_mse_loss_backward = nil;



static thread_local int g_async_mode = 0;
static id<MTLCommandBuffer> g_last_cmd = nil;

static void commit_and_sync(id<MTLCommandBuffer> cmd) {
    if (g_async_mode) {
        if (g_last_cmd) [g_last_cmd release];
        [cmd retain];
        g_last_cmd = cmd;
        [cmd commit];
    } else {
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

static double g_timebase_factor = 0.0;

static void init_timebase() {
    if (g_timebase_factor == 0.0) {
        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);
        g_timebase_factor = (double)tb.numer / (double)tb.denom * 1e-9;
    }
}

static id<MTLBuffer> get_scratch_phi(size_t bytes) {
    static id<MTLBuffer> s_phi = nil;
    static size_t s_phi_cap = 0;
    if (bytes > s_phi_cap) {
        s_phi_cap = bytes * 2;
        s_phi = [g_device newBufferWithLength:s_phi_cap options:MTLResourceStorageModePrivate];
    }
    return s_phi;
}

static id<MTLBuffer> get_scratch_silu(size_t bytes) {
    static id<MTLBuffer> s_silu = nil;
    static size_t s_silu_cap = 0;
    if (bytes > s_silu_cap) {
        s_silu_cap = bytes * 2;
        s_silu = [g_device newBufferWithLength:s_silu_cap options:MTLResourceStorageModePrivate];
    }
    return s_silu;
}

static id<MTLBuffer> get_scratch_bottleneck(size_t bytes) {
    static id<MTLBuffer> s_bot = nil;
    static size_t s_bot_cap = 0;
    if (bytes > s_bot_cap) {
        s_bot_cap = bytes * 2;
        s_bot = [g_device newBufferWithLength:s_bot_cap options:MTLResourceStorageModePrivate];
    }
    return s_bot;
}

static id<MTLBuffer> get_scratch_internal(size_t bytes) {
    static id<MTLBuffer> s_int = nil;
    static size_t s_int_cap = 0;
    if (bytes > s_int_cap) {
        s_int_cap = bytes * 2;
        s_int = [g_device newBufferWithLength:s_int_cap options:MTLResourceStorageModePrivate];
    }
    return s_int;
}

static id<MTLBuffer> get_scratch_dphi(size_t bytes) {
    static id<MTLBuffer> s_dphi = nil;
    static size_t s_dphi_cap = 0;
    if (bytes > s_dphi_cap) {
        s_dphi_cap = bytes * 2;
        s_dphi = [g_device newBufferWithLength:s_dphi_cap options:MTLResourceStorageModePrivate];
    }
    return s_dphi;
}

static id<MTLBuffer> get_scratch_dx_silu(size_t bytes) {
    static id<MTLBuffer> s_dx_silu = nil;
    static size_t s_dx_silu_cap = 0;
    if (bytes > s_dx_silu_cap) {
        s_dx_silu_cap = bytes * 2;
        s_dx_silu = [g_device newBufferWithLength:s_dx_silu_cap options:MTLResourceStorageModePrivate];
    }
    return s_dx_silu;
}

static id<MTLBuffer> get_scratch_y(size_t bytes) {
    static id<MTLBuffer> s_y = nil;
    static size_t s_y_cap = 0;
    if (bytes > s_y_cap) {
        s_y_cap = bytes * 2;
        s_y = [g_device newBufferWithLength:s_y_cap options:MTLResourceStorageModePrivate];
    }
    return s_y;
}

static id<MTLBuffer> get_scratch_dy(size_t bytes) {
    static id<MTLBuffer> s_dy = nil;
    static size_t s_dy_cap = 0;
    if (bytes > s_dy_cap) {
        s_dy_cap = bytes * 2;
        s_dy = [g_device newBufferWithLength:s_dy_cap options:MTLResourceStorageModePrivate];
    }
    return s_dy;
}

static id<MTLBuffer> get_scratch_dw(size_t bytes) {
    static id<MTLBuffer> s_dw = nil;
    static size_t s_dw_cap = 0;
    if (bytes > s_dw_cap) {
        s_dw_cap = bytes * 2;
        s_dw = [g_device newBufferWithLength:s_dw_cap options:MTLResourceStorageModePrivate];
    }
    return s_dw;
}

static id<MTLBuffer> get_scratch_dw_base(size_t bytes) {
    static id<MTLBuffer> s_dwb = nil;
    static size_t s_dwb_cap = 0;
    if (bytes > s_dwb_cap) {
        s_dwb_cap = bytes * 2;
        s_dwb = [g_device newBufferWithLength:s_dwb_cap options:MTLResourceStorageModePrivate];
    }
    return s_dwb;
}

static id<MTLBuffer> get_scratch_dbias(size_t bytes) {
    static id<MTLBuffer> s_db = nil;
    static size_t s_db_cap = 0;
    if (bytes > s_db_cap) {
        s_db_cap = bytes * 2;
        s_db = [g_device newBufferWithLength:s_db_cap options:MTLResourceStorageModePrivate];
    }
    return s_db;
}

#include <map>
#include <tuple>

static MPSMatrixDescriptor* get_cached_desc(NSUInteger rows, NSUInteger cols, NSUInteger rowBytes, MPSDataType dataType = MPSDataTypeFloat32) {
    static std::map<std::tuple<NSUInteger, NSUInteger, NSUInteger, int>, MPSMatrixDescriptor*> s_desc_cache;
    auto key = std::make_tuple(rows, cols, rowBytes, (int)dataType);
    auto it = s_desc_cache.find(key);
    if (it != s_desc_cache.end()) {
        return it->second;
    }
    MPSMatrixDescriptor* desc = [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:cols rowBytes:rowBytes dataType:dataType];
    [desc retain];
    s_desc_cache[key] = desc;
    return desc;
}

static MPSMatrixMultiplication* get_cached_matmul(id<MTLDevice> dev, NSUInteger M, NSUInteger N, NSUInteger K, float alpha, float beta) {
    static std::map<std::tuple<NSUInteger, NSUInteger, NSUInteger, int, int>, MPSMatrixMultiplication*> s_mm_cache;
    auto key = std::make_tuple(M, N, K, (int)(alpha * 100), (int)(beta * 100));
    auto it = s_mm_cache.find(key);
    if (it != s_mm_cache.end()) {
        return it->second;
    }
    MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc] initWithDevice:dev
                                                                   transposeLeft:NO
                                                                  transposeRight:YES
                                                                      resultRows:M
                                                                   resultColumns:N
                                                                 interiorColumns:K
                                                                           alpha:alpha
                                                                            beta:beta];
    s_mm_cache[key] = mm;
    return mm;
}

static MPSMatrixMultiplication* get_cached_matmul_general(id<MTLDevice> dev, BOOL transLeft, BOOL transRight, NSUInteger M, NSUInteger N, NSUInteger K, float alpha, float beta) {
    static std::map<std::tuple<int, int, NSUInteger, NSUInteger, NSUInteger, int, int>, MPSMatrixMultiplication*> s_mm_gen_cache;
    auto key = std::make_tuple((int)transLeft, (int)transRight, M, N, K, (int)(alpha * 100), (int)(beta * 100));
    auto it = s_mm_gen_cache.find(key);
    if (it != s_mm_gen_cache.end()) {
        return it->second;
    }
    MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc] initWithDevice:dev
                                                                   transposeLeft:transLeft
                                                                  transposeRight:transRight
                                                                      resultRows:M
                                                                   resultColumns:N
                                                                 interiorColumns:K
                                                                           alpha:alpha
                                                                            beta:beta];
    s_mm_gen_cache[key] = mm;
    return mm;
}

extern "C" {

int metal_kan_set_async(int enabled) {
    g_async_mode = enabled;
    return 0;
}

int metal_kan_sync() {
    if (g_last_cmd) {
        [g_last_cmd waitUntilCompleted];
        [g_last_cmd release];
        g_last_cmd = nil;
    }
    return 0;
}

int metal_kan_init(const char* shader_path) {
    @autoreleasepool {
        init_timebase();
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) return -1;
        g_queue = [g_device newCommandQueue];

        std::ifstream file(shader_path);
        if (!file.is_open()) return -2;
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string source_str = buffer.str();

        NSString* msl_source = [NSString stringWithUTF8String:source_str.c_str()];
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion3_0;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = YES;
#pragma clang diagnostic pop
        
        NSError* error = nil;
        id<MTLLibrary> library = [g_device newLibraryWithSource:msl_source options:options error:&error];
        if (!library) {
            std::cerr << "[MetalKAN] Compile error: " << [[error localizedDescription] UTF8String] << std::endl;
            return -3;
        }

        auto make_pipe = [&](NSString* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> fn = [library newFunctionWithName:name];
            if (!fn) return nil;
            NSError* err = nil;
            return [g_device newComputePipelineStateWithFunction:fn error:&err];
        };

        g_pipe_cheby_tiled    = make_pipe(@"kan_cheby_tiled_deg4");
        g_pipe_fastkan_tiled  = make_pipe(@"kan_fastkan_tiled");
        g_pipe_relu_tiled     = make_pipe(@"kan_relu_tiled");
        g_pipe_wavkan_tiled   = make_pipe(@"kan_wavkan_tiled");
        g_pipe_fourier_tiled  = make_pipe(@"kan_fourier_tiled");
        g_pipe_jacobi_tiled   = make_pipe(@"kan_jacobi_tiled");
        g_pipe_rational_tiled = make_pipe(@"kan_rational_tiled");
        g_pipe_bspline_tiled  = make_pipe(@"kan_bspline_tiled");

        g_pipe_prep           = make_pipe(@"eval_base_and_bias");
        g_pipe_rbf_basis      = make_pipe(@"eval_fastkan_rbf_basis");
        g_pipe_wav_basis      = make_pipe(@"eval_wavkan_basis");
        g_pipe_cheby_basis    = make_pipe(@"eval_cheby_basis");
        g_pipe_bspline_basis  = make_pipe(@"eval_bspline_basis");
        g_pipe_relu_basis     = make_pipe(@"eval_relu_basis");
        g_pipe_fourier_basis  = make_pipe(@"eval_fourier_basis");
        g_pipe_jacobi_basis   = make_pipe(@"eval_jacobi_basis");
        g_pipe_combine_mult   = make_pipe(@"combine_mult_nodes");
        g_pipe_reduce_sum_cols = make_pipe(@"reduce_sum_columns");
        g_pipe_eval_silu       = make_pipe(@"eval_silu");
        g_pipe_backward_cheby = make_pipe(@"backward_cheby_basis");
        g_pipe_backward_rbf   = make_pipe(@"backward_fastkan_rbf_basis");
        g_pipe_backward_relu  = make_pipe(@"backward_relu_basis");
        g_pipe_backward_bspline = make_pipe(@"backward_bspline_basis");

        // Direct SIMDgroup matrix & FP16 pipelines
        g_pipe_gemm_simd_fp32    = make_pipe(@"gemm_simd_16x16_fp32");
        g_pipe_lowrank_simd_fp32 = make_pipe(@"fused_lowrank_simd_fp32");
        g_pipe_prep_fp16         = make_pipe(@"eval_base_and_bias_fp16");
        g_pipe_cheby_basis_fp16  = make_pipe(@"eval_cheby_basis_fp16");
        g_pipe_rbf_basis_fp16    = make_pipe(@"eval_fastkan_rbf_basis_fp16");
        g_pipe_relu_basis_fp16   = make_pipe(@"eval_relu_basis_fp16");
        g_pipe_bspline_basis_fp16= make_pipe(@"eval_bspline_basis_fp16");
        g_pipe_wav_basis_fp16    = make_pipe(@"eval_wavkan_basis_fp16");
        g_pipe_fourier_basis_fp16= make_pipe(@"eval_fourier_basis_fp16");
        g_pipe_jacobi_basis_fp16 = make_pipe(@"eval_jacobi_basis_fp16");
        g_pipe_combine_mult_fp16 = make_pipe(@"combine_mult_nodes_fp16");

        // Training pipelines
        g_pipe_mse_loss_backward = make_pipe(@"mse_loss_backward");
        g_pipe_adamw_step        = make_pipe(@"adamw_step");
        g_pipe_sgd_step          = make_pipe(@"sgd_step");
        g_pipe_lion_step         = make_pipe(@"lion_step");
        g_pipe_rmsprop_step      = make_pipe(@"rmsprop_step");

        // GPU Muon, Fused Backward & Sub-4-bit pipelines
        g_pipe_matrix_transpose_f32 = make_pipe(@"matrix_transpose_f32");
        g_pipe_matrix_scale_f32     = make_pipe(@"matrix_scale_f32");
        g_pipe_ns5_combine_B_f32    = make_pipe(@"ns5_combine_B_f32");
        g_pipe_ns5_combine_X_f32    = make_pipe(@"ns5_combine_X_f32");
        g_pipe_fused_backward_cheby_dW = make_pipe(@"fused_backward_cheby_dW");
        g_pipe_fused_backward_cheby_dX = make_pipe(@"fused_backward_cheby_dX");
        g_pipe_kan_forward_ternary_cheby = make_pipe(@"kan_forward_ternary_cheby");
        g_pipe_kan_forward_int2_cheby    = make_pipe(@"kan_forward_int2_cheby");

        // High-performance element-wise, activation & optimizer pipelines
        g_pipe_swiglu_forward            = make_pipe(@"swiglu_forward_f32");
        g_pipe_swiglu_backward           = make_pipe(@"swiglu_backward_f32");
        g_pipe_elementwise_add           = make_pipe(@"elementwise_add_f32");
        g_pipe_adam_coupled_step         = make_pipe(@"adam_coupled_step");
        g_pipe_muon_momentum_update      = make_pipe(@"muon_momentum_update");
        g_pipe_muon_param_update         = make_pipe(@"muon_param_update");
        g_pipe_dequantize_ternary_fast   = make_pipe(@"dequantize_ternary_fast");
        g_pipe_dequantize_int2_fast      = make_pipe(@"dequantize_int2_fast");
        g_pipe_calc_mse_loss_backward    = make_pipe(@"calc_mse_loss_backward");



        if (!g_pipe_cheby_tiled || !g_pipe_fastkan_tiled || !g_pipe_relu_tiled ||
            !g_pipe_wavkan_tiled || !g_pipe_fourier_tiled || !g_pipe_jacobi_tiled ||
            !g_pipe_rational_tiled || !g_pipe_bspline_tiled ||
            !g_pipe_prep || !g_pipe_rbf_basis || !g_pipe_wav_basis ||
            !g_pipe_cheby_basis || !g_pipe_bspline_basis || !g_pipe_relu_basis ||
            !g_pipe_fourier_basis || !g_pipe_jacobi_basis || !g_pipe_combine_mult) {
            std::cerr << "[MetalKAN] Failed to create one or more compute pipelines!" << std::endl;
            return -4;
        }

        return 0;
    }
}

#include <unordered_map>
#include <mutex>

static id<MTLBuffer> make_no_copy_buffer(void* ptr, size_t bytes) {
    if (!ptr) return nil;
    static std::unordered_map<void*, id<MTLBuffer>> s_buf_cache;
    static std::mutex s_buf_mutex;

    std::lock_guard<std::mutex> lock(s_buf_mutex);
    auto it = s_buf_cache.find(ptr);
    if (it != s_buf_cache.end()) {
        if ([it->second length] >= bytes) {
            return it->second;
        } else {
            [it->second release];
            s_buf_cache.erase(it);
        }
    }

    if (s_buf_cache.size() > 2048) {
        for (auto& kv : s_buf_cache) {
            [kv.second release];
        }
        s_buf_cache.clear();
    }

    id<MTLBuffer> buf = [g_device newBufferWithBytesNoCopy:ptr
                                                   length:bytes
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
    if (buf) {
        [buf retain];
        s_buf_cache[ptr] = buf;
    }
    return buf;
}

int metal_kan_cheby_forward(
    const float* X,
    const float* W_cheby,
    const float* W_base,
    const float* bias,
    float*       Y,
    int B,
    int D_in,
    int D_out,
    int K,
    int has_base,
    int has_bias
) {
    @autoreleasepool {
        int K_dim = D_in * K;
        id<MTLBuffer> buf_X       = make_no_copy_buffer((void*)X, B * D_in * sizeof(float));
        id<MTLBuffer> buf_W_cheby = make_no_copy_buffer((void*)W_cheby, D_out * K_dim * sizeof(float));
        id<MTLBuffer> buf_W_base  = has_base ? make_no_copy_buffer((void*)W_base, D_out * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_bias    = has_bias ? make_no_copy_buffer((void*)bias, D_out * sizeof(float)) : nil;
        id<MTLBuffer> buf_Y       = make_no_copy_buffer((void*)Y, B * D_out * sizeof(float));

        id<MTLBuffer> buf_Phi    = get_scratch_phi(B * K_dim * sizeof(float));
        id<MTLBuffer> buf_X_silu = has_base ? get_scratch_silu(B * D_in * sizeof(float)) : nil;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        uint uB = (uint)B, uDin = (uint)D_in, uDout = (uint)D_out;
        uint uBase = (uint)has_base, uBias = (uint)has_bias;

        // 1. Base SiLU and bias initialization (only when needed)
        if (has_base || has_bias) {
            id<MTLComputeCommandEncoder> enc_prep = [cmd computeCommandEncoder];
            [enc_prep setComputePipelineState:g_pipe_prep];
            [enc_prep setBuffer:buf_X offset:0 atIndex:0];
            [enc_prep setBuffer:(buf_bias ? buf_bias : buf_X) offset:0 atIndex:1];
            [enc_prep setBuffer:(buf_X_silu ? buf_X_silu : buf_X) offset:0 atIndex:2];
            [enc_prep setBuffer:buf_Y offset:0 atIndex:3];
            [enc_prep setBytes:&uB length:sizeof(uint) atIndex:4];
            [enc_prep setBytes:&uDin length:sizeof(uint) atIndex:5];
            [enc_prep setBytes:&uDout length:sizeof(uint) atIndex:6];
            [enc_prep setBytes:&uBase length:sizeof(uint) atIndex:7];
            [enc_prep setBytes:&uBias length:sizeof(uint) atIndex:8];
            uint max_dim = (D_in > D_out) ? D_in : D_out;
            [enc_prep dispatchThreads:MTLSizeMake(max_dim, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc_prep endEncoding];
        }

        // 2. Base linear branch: Y += X_silu @ W_base.T
        if (has_base) {
            MPSMatrixDescriptor* desc_A_base = get_cached_desc(B, D_in, D_in * sizeof(float));
            MPSMatrixDescriptor* desc_B_base = get_cached_desc(D_out, D_in, D_in * sizeof(float));
            MPSMatrixDescriptor* desc_C_base = get_cached_desc(B, D_out, D_out * sizeof(float));
            MPSMatrix* mat_A_base = [[[MPSMatrix alloc] initWithBuffer:buf_X_silu descriptor:desc_A_base] autorelease];
            MPSMatrix* mat_B_base = [[[MPSMatrix alloc] initWithBuffer:buf_W_base descriptor:desc_B_base] autorelease];
            MPSMatrix* mat_C_base = [[[MPSMatrix alloc] initWithBuffer:buf_Y descriptor:desc_C_base] autorelease];
            float beta_base = has_bias ? 1.0f : 0.0f;
            MPSMatrixMultiplication* matmul_base = get_cached_matmul(g_device, B, D_out, D_in, 1.0f, beta_base);
            [matmul_base encodeToCommandBuffer:cmd leftMatrix:mat_A_base rightMatrix:mat_B_base resultMatrix:mat_C_base];
        }

        // 3. Cheby basis evaluation
        id<MTLComputeCommandEncoder> enc_cheby = [cmd computeCommandEncoder];
        [enc_cheby setComputePipelineState:g_pipe_cheby_basis];
        [enc_cheby setBuffer:buf_X offset:0 atIndex:0];
        [enc_cheby setBuffer:buf_Phi offset:0 atIndex:1];
        uint uK = (uint)K;
        [enc_cheby setBytes:&uB length:sizeof(uint) atIndex:2];
        [enc_cheby setBytes:&uDin length:sizeof(uint) atIndex:3];
        [enc_cheby setBytes:&uK length:sizeof(uint) atIndex:4];
        [enc_cheby dispatchThreads:MTLSizeMake(D_in, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc_cheby endEncoding];

        // 4. Matrix multiplication: Y += Phi @ W_cheby.T
        MPSMatrixDescriptor* desc_A_cheby = get_cached_desc(B, K_dim, K_dim * sizeof(float));
        MPSMatrixDescriptor* desc_B_cheby = get_cached_desc(D_out, K_dim, K_dim * sizeof(float));
        MPSMatrixDescriptor* desc_C_cheby = get_cached_desc(B, D_out, D_out * sizeof(float));
        MPSMatrix* mat_A_cheby = [[[MPSMatrix alloc] initWithBuffer:buf_Phi descriptor:desc_A_cheby] autorelease];
        MPSMatrix* mat_B_cheby = [[[MPSMatrix alloc] initWithBuffer:buf_W_cheby descriptor:desc_B_cheby] autorelease];
        MPSMatrix* mat_C_cheby = [[[MPSMatrix alloc] initWithBuffer:buf_Y descriptor:desc_C_cheby] autorelease];
        float beta_spline = (has_base || has_bias) ? 1.0f : 0.0f;
        MPSMatrixMultiplication* matmul_cheby = get_cached_matmul(g_device, B, D_out, K_dim, 1.0f, beta_spline);
        [matmul_cheby encodeToCommandBuffer:cmd leftMatrix:mat_A_cheby rightMatrix:mat_B_cheby resultMatrix:mat_C_cheby];

        [cmd commit];
        [cmd waitUntilCompleted];

        return 0;
    }
}

int metal_kan_fastkan_forward(
    const float* X,
    const float* W_rbf,
    const float* W_base,
    const float* grid,
    const float* bias,
    float*       Y,
    int B,
    int D_in,
    int D_out,
    int num_centers,
    float inv_denominator,
    int has_base,
    int has_bias
) {
    @autoreleasepool {
        int K_dim = D_in * num_centers;
        id<MTLBuffer> buf_X      = make_no_copy_buffer((void*)X, B * D_in * sizeof(float));
        id<MTLBuffer> buf_W_rbf  = make_no_copy_buffer((void*)W_rbf, D_out * K_dim * sizeof(float));
        id<MTLBuffer> buf_W_base = has_base ? make_no_copy_buffer((void*)W_base, D_out * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_grid   = make_no_copy_buffer((void*)grid, num_centers * sizeof(float));
        id<MTLBuffer> buf_bias   = has_bias ? make_no_copy_buffer((void*)bias, D_out * sizeof(float)) : nil;
        id<MTLBuffer> buf_Y      = make_no_copy_buffer((void*)Y, B * D_out * sizeof(float));

        id<MTLBuffer> buf_Phi    = get_scratch_phi(B * K_dim * sizeof(float));
        id<MTLBuffer> buf_X_silu = has_base ? get_scratch_silu(B * D_in * sizeof(float)) : nil;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        uint uB = (uint)B, uDin = (uint)D_in, uDout = (uint)D_out;
        uint uBase = (uint)has_base, uBias = (uint)has_bias;

        // 1. Base SiLU and bias initialization (only when needed)
        if (has_base || has_bias) {
            id<MTLComputeCommandEncoder> enc_prep = [cmd computeCommandEncoder];
            [enc_prep setComputePipelineState:g_pipe_prep];
            [enc_prep setBuffer:buf_X offset:0 atIndex:0];
            [enc_prep setBuffer:(buf_bias ? buf_bias : buf_X) offset:0 atIndex:1];
            [enc_prep setBuffer:(buf_X_silu ? buf_X_silu : buf_X) offset:0 atIndex:2];
            [enc_prep setBuffer:buf_Y offset:0 atIndex:3];
            [enc_prep setBytes:&uB length:sizeof(uint) atIndex:4];
            [enc_prep setBytes:&uDin length:sizeof(uint) atIndex:5];
            [enc_prep setBytes:&uDout length:sizeof(uint) atIndex:6];
            [enc_prep setBytes:&uBase length:sizeof(uint) atIndex:7];
            [enc_prep setBytes:&uBias length:sizeof(uint) atIndex:8];
            uint max_dim = (D_in > D_out) ? D_in : D_out;
            [enc_prep dispatchThreads:MTLSizeMake(max_dim, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc_prep endEncoding];
        }

        // 2. Base linear branch: Y += X_silu @ W_base.T
        if (has_base) {
            MPSMatrixDescriptor* desc_A_base = get_cached_desc(B, D_in, D_in * sizeof(float));
            MPSMatrixDescriptor* desc_B_base = get_cached_desc(D_out, D_in, D_in * sizeof(float));
            MPSMatrixDescriptor* desc_C_base = get_cached_desc(B, D_out, D_out * sizeof(float));
            MPSMatrix* mat_A_base = [[[MPSMatrix alloc] initWithBuffer:buf_X_silu descriptor:desc_A_base] autorelease];
            MPSMatrix* mat_B_base = [[[MPSMatrix alloc] initWithBuffer:buf_W_base descriptor:desc_B_base] autorelease];
            MPSMatrix* mat_C_base = [[[MPSMatrix alloc] initWithBuffer:buf_Y descriptor:desc_C_base] autorelease];
            float beta_base = has_bias ? 1.0f : 0.0f;
            MPSMatrixMultiplication* matmul_base = get_cached_matmul(g_device, B, D_out, D_in, 1.0f, beta_base);
            [matmul_base encodeToCommandBuffer:cmd leftMatrix:mat_A_base rightMatrix:mat_B_base resultMatrix:mat_C_base];
        }

        // 3. FastKAN RBF basis evaluation
        id<MTLComputeCommandEncoder> enc_rbf = [cmd computeCommandEncoder];
        [enc_rbf setComputePipelineState:g_pipe_rbf_basis];
        [enc_rbf setBuffer:buf_X offset:0 atIndex:0];
        [enc_rbf setBuffer:buf_grid offset:0 atIndex:1];
        [enc_rbf setBuffer:buf_Phi offset:0 atIndex:2];
        uint uK = (uint)num_centers;
        float u_inv_d = inv_denominator;
        [enc_rbf setBytes:&uB length:sizeof(uint) atIndex:3];
        [enc_rbf setBytes:&uDin length:sizeof(uint) atIndex:4];
        [enc_rbf setBytes:&uK length:sizeof(uint) atIndex:5];
        [enc_rbf setBytes:&u_inv_d length:sizeof(float) atIndex:6];
        [enc_rbf dispatchThreads:MTLSizeMake(D_in, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc_rbf endEncoding];

        // 4. FastKAN RBF matrix multiplication: Y += Phi @ W_rbf.T
        MPSMatrixDescriptor* desc_A_rbf = get_cached_desc(B, K_dim, K_dim * sizeof(float));
        MPSMatrixDescriptor* desc_B_rbf = get_cached_desc(D_out, K_dim, K_dim * sizeof(float));
        MPSMatrixDescriptor* desc_C_rbf = get_cached_desc(B, D_out, D_out * sizeof(float));
        MPSMatrix* mat_A_rbf = [[[MPSMatrix alloc] initWithBuffer:buf_Phi descriptor:desc_A_rbf] autorelease];
        MPSMatrix* mat_B_rbf = [[[MPSMatrix alloc] initWithBuffer:buf_W_rbf descriptor:desc_B_rbf] autorelease];
        MPSMatrix* mat_C_rbf = [[[MPSMatrix alloc] initWithBuffer:buf_Y descriptor:desc_C_rbf] autorelease];
        float beta_rbf = (has_base || has_bias) ? 1.0f : 0.0f;
        MPSMatrixMultiplication* matmul_rbf = get_cached_matmul(g_device, B, D_out, K_dim, 1.0f, beta_rbf);
        [matmul_rbf encodeToCommandBuffer:cmd leftMatrix:mat_A_rbf rightMatrix:mat_B_rbf resultMatrix:mat_C_rbf];

        [cmd commit];
        [cmd waitUntilCompleted];

        return 0;
    }
}

int metal_kan_relu_forward(
    const float* X,
    const float* W_relu,
    const float* W_base,
    const float* grid,
    const float* bias,
    float*       Y,
    int B,
    int D_in,
    int D_out,
    int num_grids,
    float inv_h,
    int has_base,
    int has_bias
) {
    @autoreleasepool {
        int K_dim = D_in * num_grids;
        id<MTLBuffer> buf_X      = make_no_copy_buffer((void*)X, B * D_in * sizeof(float));
        id<MTLBuffer> buf_W_relu = make_no_copy_buffer((void*)W_relu, D_out * K_dim * sizeof(float));
        id<MTLBuffer> buf_W_base = has_base ? make_no_copy_buffer((void*)W_base, D_out * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_grid   = make_no_copy_buffer((void*)grid, num_grids * sizeof(float));
        id<MTLBuffer> buf_bias   = has_bias ? make_no_copy_buffer((void*)bias, D_out * sizeof(float)) : nil;
        id<MTLBuffer> buf_Y      = make_no_copy_buffer((void*)Y, B * D_out * sizeof(float));

        id<MTLBuffer> buf_Phi    = get_scratch_phi(B * K_dim * sizeof(float));
        id<MTLBuffer> buf_X_silu = has_base ? get_scratch_silu(B * D_in * sizeof(float)) : nil;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        uint uB = (uint)B, uDin = (uint)D_in, uDout = (uint)D_out;
        uint uBase = (uint)has_base, uBias = (uint)has_bias;

        // 1. Base SiLU and bias initialization (only when needed)
        if (has_base || has_bias) {
            id<MTLComputeCommandEncoder> enc_prep = [cmd computeCommandEncoder];
            [enc_prep setComputePipelineState:g_pipe_prep];
            [enc_prep setBuffer:buf_X offset:0 atIndex:0];
            [enc_prep setBuffer:(buf_bias ? buf_bias : buf_X) offset:0 atIndex:1];
            [enc_prep setBuffer:(buf_X_silu ? buf_X_silu : buf_X) offset:0 atIndex:2];
            [enc_prep setBuffer:buf_Y offset:0 atIndex:3];
            [enc_prep setBytes:&uB length:sizeof(uint) atIndex:4];
            [enc_prep setBytes:&uDin length:sizeof(uint) atIndex:5];
            [enc_prep setBytes:&uDout length:sizeof(uint) atIndex:6];
            [enc_prep setBytes:&uBase length:sizeof(uint) atIndex:7];
            [enc_prep setBytes:&uBias length:sizeof(uint) atIndex:8];
            uint max_dim = (D_in > D_out) ? D_in : D_out;
            [enc_prep dispatchThreads:MTLSizeMake(max_dim, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc_prep endEncoding];
        }

        // 2. Base linear branch: Y += X_silu @ W_base.T
        if (has_base) {
            MPSMatrixDescriptor* desc_A_base = get_cached_desc(B, D_in, D_in * sizeof(float));
            MPSMatrixDescriptor* desc_B_base = get_cached_desc(D_out, D_in, D_in * sizeof(float));
            MPSMatrixDescriptor* desc_C_base = get_cached_desc(B, D_out, D_out * sizeof(float));
            MPSMatrix* mat_A_base = [[[MPSMatrix alloc] initWithBuffer:buf_X_silu descriptor:desc_A_base] autorelease];
            MPSMatrix* mat_B_base = [[[MPSMatrix alloc] initWithBuffer:buf_W_base descriptor:desc_B_base] autorelease];
            MPSMatrix* mat_C_base = [[[MPSMatrix alloc] initWithBuffer:buf_Y descriptor:desc_C_base] autorelease];
            float beta_base = has_bias ? 1.0f : 0.0f;
            MPSMatrixMultiplication* matmul_base = get_cached_matmul(g_device, B, D_out, D_in, 1.0f, beta_base);
            [matmul_base encodeToCommandBuffer:cmd leftMatrix:mat_A_base rightMatrix:mat_B_base resultMatrix:mat_C_base];
        }

        // 3. ReLUKAN basis evaluation
        id<MTLComputeCommandEncoder> enc_relu = [cmd computeCommandEncoder];
        [enc_relu setComputePipelineState:g_pipe_relu_basis];
        [enc_relu setBuffer:buf_X offset:0 atIndex:0];
        [enc_relu setBuffer:buf_grid offset:0 atIndex:1];
        [enc_relu setBuffer:buf_Phi offset:0 atIndex:2];
        uint uG = (uint)num_grids;
        float u_inv_h = inv_h;
        [enc_relu setBytes:&uB length:sizeof(uint) atIndex:3];
        [enc_relu setBytes:&uDin length:sizeof(uint) atIndex:4];
        [enc_relu setBytes:&uG length:sizeof(uint) atIndex:5];
        [enc_relu setBytes:&u_inv_h length:sizeof(float) atIndex:6];
        [enc_relu dispatchThreads:MTLSizeMake(D_in, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc_relu endEncoding];

        // 4. Matrix multiplication: Y += Phi @ W_relu.T
        MPSMatrixDescriptor* desc_A_relu = get_cached_desc(B, K_dim, K_dim * sizeof(float));
        MPSMatrixDescriptor* desc_B_relu = get_cached_desc(D_out, K_dim, K_dim * sizeof(float));
        MPSMatrixDescriptor* desc_C_relu = get_cached_desc(B, D_out, D_out * sizeof(float));
        MPSMatrix* mat_A_relu = [[[MPSMatrix alloc] initWithBuffer:buf_Phi descriptor:desc_A_relu] autorelease];
        MPSMatrix* mat_B_relu = [[[MPSMatrix alloc] initWithBuffer:buf_W_relu descriptor:desc_B_relu] autorelease];
        MPSMatrix* mat_C_relu = [[[MPSMatrix alloc] initWithBuffer:buf_Y descriptor:desc_C_relu] autorelease];
        float beta_spline = (has_base || has_bias) ? 1.0f : 0.0f;
        MPSMatrixMultiplication* matmul_relu = get_cached_matmul(g_device, B, D_out, K_dim, 1.0f, beta_spline);
        [matmul_relu encodeToCommandBuffer:cmd leftMatrix:mat_A_relu rightMatrix:mat_B_relu resultMatrix:mat_C_relu];

        [cmd commit];
        [cmd waitUntilCompleted];

        return 0;
    }
}

int metal_kan_wavkan_forward(
    const float* X,
    const float* W_wav,
    const float* W_base,
    const float* translation,
    const float* inv_scale,
    const float* bias,
    float*       Y,
    int B,
    int D_in,
    int D_out,
    int num_wavelets,
    int wavelet_type,
    int has_base,
    int has_bias
) {
    @autoreleasepool {
        int K_dim = D_in * num_wavelets;
        id<MTLBuffer> buf_X       = make_no_copy_buffer((void*)X, B * D_in * sizeof(float));
        id<MTLBuffer> buf_W_wav   = make_no_copy_buffer((void*)W_wav, D_out * K_dim * sizeof(float));
        id<MTLBuffer> buf_W_base  = has_base ? make_no_copy_buffer((void*)W_base, D_out * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_trans   = make_no_copy_buffer((void*)translation, D_in * num_wavelets * sizeof(float));
        id<MTLBuffer> buf_scale   = make_no_copy_buffer((void*)inv_scale, D_in * num_wavelets * sizeof(float));
        id<MTLBuffer> buf_bias    = has_bias ? make_no_copy_buffer((void*)bias, D_out * sizeof(float)) : nil;
        id<MTLBuffer> buf_Y       = make_no_copy_buffer((void*)Y, B * D_out * sizeof(float));

        id<MTLBuffer> buf_Phi     = get_scratch_phi(B * K_dim * sizeof(float));
        id<MTLBuffer> buf_X_silu  = has_base ? get_scratch_silu(B * D_in * sizeof(float)) : nil;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        uint uB = (uint)B, uDin = (uint)D_in, uDout = (uint)D_out;
        uint uBase = (uint)has_base, uBias = (uint)has_bias;

        // 1. Base SiLU and bias initialization (only when needed)
        if (has_base || has_bias) {
            id<MTLComputeCommandEncoder> enc_prep = [cmd computeCommandEncoder];
            [enc_prep setComputePipelineState:g_pipe_prep];
            [enc_prep setBuffer:buf_X offset:0 atIndex:0];
            [enc_prep setBuffer:(buf_bias ? buf_bias : buf_X) offset:0 atIndex:1];
            [enc_prep setBuffer:(buf_X_silu ? buf_X_silu : buf_X) offset:0 atIndex:2];
            [enc_prep setBuffer:buf_Y offset:0 atIndex:3];
            [enc_prep setBytes:&uB length:sizeof(uint) atIndex:4];
            [enc_prep setBytes:&uDin length:sizeof(uint) atIndex:5];
            [enc_prep setBytes:&uDout length:sizeof(uint) atIndex:6];
            [enc_prep setBytes:&uBase length:sizeof(uint) atIndex:7];
            [enc_prep setBytes:&uBias length:sizeof(uint) atIndex:8];
            uint max_dim = (D_in > D_out) ? D_in : D_out;
            [enc_prep dispatchThreads:MTLSizeMake(max_dim, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc_prep endEncoding];
        }

        // 2. Base linear branch: Y += X_silu @ W_base.T
        if (has_base) {
            MPSMatrixDescriptor* desc_A_base = get_cached_desc(B, D_in, D_in * sizeof(float));
            MPSMatrixDescriptor* desc_B_base = get_cached_desc(D_out, D_in, D_in * sizeof(float));
            MPSMatrixDescriptor* desc_C_base = get_cached_desc(B, D_out, D_out * sizeof(float));
            MPSMatrix* mat_A_base = [[[MPSMatrix alloc] initWithBuffer:buf_X_silu descriptor:desc_A_base] autorelease];
            MPSMatrix* mat_B_base = [[[MPSMatrix alloc] initWithBuffer:buf_W_base descriptor:desc_B_base] autorelease];
            MPSMatrix* mat_C_base = [[[MPSMatrix alloc] initWithBuffer:buf_Y descriptor:desc_C_base] autorelease];
            float beta_base = has_bias ? 1.0f : 0.0f;
            MPSMatrixMultiplication* matmul_base = get_cached_matmul(g_device, B, D_out, D_in, 1.0f, beta_base);
            [matmul_base encodeToCommandBuffer:cmd leftMatrix:mat_A_base rightMatrix:mat_B_base resultMatrix:mat_C_base];
        }

        // 3. WavKAN basis evaluation
        id<MTLComputeCommandEncoder> enc_wav = [cmd computeCommandEncoder];
        [enc_wav setComputePipelineState:g_pipe_wav_basis];
        [enc_wav setBuffer:buf_X offset:0 atIndex:0];
        [enc_wav setBuffer:buf_trans offset:0 atIndex:1];
        [enc_wav setBuffer:buf_scale offset:0 atIndex:2];
        [enc_wav setBuffer:buf_Phi offset:0 atIndex:3];
        uint uNwav = (uint)num_wavelets;
        uint uWtype = (uint)wavelet_type;
        [enc_wav setBytes:&uB length:sizeof(uint) atIndex:4];
        [enc_wav setBytes:&uDin length:sizeof(uint) atIndex:5];
        [enc_wav setBytes:&uNwav length:sizeof(uint) atIndex:6];
        [enc_wav setBytes:&uWtype length:sizeof(uint) atIndex:7];
        [enc_wav dispatchThreads:MTLSizeMake(D_in, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc_wav endEncoding];

        // 4. WavKAN matrix multiplication: Y += Phi @ W_wav.T
        MPSMatrixDescriptor* desc_A_wav = get_cached_desc(B, K_dim, K_dim * sizeof(float));
        MPSMatrixDescriptor* desc_B_wav = get_cached_desc(D_out, K_dim, K_dim * sizeof(float));
        MPSMatrixDescriptor* desc_C_wav = get_cached_desc(B, D_out, D_out * sizeof(float));
        MPSMatrix* mat_A_wav = [[[MPSMatrix alloc] initWithBuffer:buf_Phi descriptor:desc_A_wav] autorelease];
        MPSMatrix* mat_B_wav = [[[MPSMatrix alloc] initWithBuffer:buf_W_wav descriptor:desc_B_wav] autorelease];
        MPSMatrix* mat_C_wav = [[[MPSMatrix alloc] initWithBuffer:buf_Y descriptor:desc_C_wav] autorelease];
        float beta_spline = (has_base || has_bias) ? 1.0f : 0.0f;
        MPSMatrixMultiplication* matmul_wav = get_cached_matmul(g_device, B, D_out, K_dim, 1.0f, beta_spline);
        [matmul_wav encodeToCommandBuffer:cmd leftMatrix:mat_A_wav rightMatrix:mat_B_wav resultMatrix:mat_C_wav];

        [cmd commit];
        [cmd waitUntilCompleted];

        return 0;
    }
}

int metal_kan_fourier_forward(
    const float* X,
    const float* W_fourier,
    const float* W_base,
    const float* bias,
    float*       Y,
    int B,
    int D_in,
    int D_out,
    int num_freqs,
    int has_base,
    int has_bias
) {
    @autoreleasepool {
        uint num_bases = 2 * num_freqs + 1;
        int K_dim = D_in * num_bases;
        id<MTLBuffer> buf_X      = make_no_copy_buffer((void*)X, B * D_in * sizeof(float));
        id<MTLBuffer> buf_W_four = make_no_copy_buffer((void*)W_fourier, D_out * K_dim * sizeof(float));
        id<MTLBuffer> buf_W_base = has_base ? make_no_copy_buffer((void*)W_base, D_out * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_bias   = has_bias ? make_no_copy_buffer((void*)bias, D_out * sizeof(float)) : nil;
        id<MTLBuffer> buf_Y      = make_no_copy_buffer((void*)Y, B * D_out * sizeof(float));

        id<MTLBuffer> buf_Phi    = get_scratch_phi(B * K_dim * sizeof(float));
        id<MTLBuffer> buf_X_silu = has_base ? get_scratch_silu(B * D_in * sizeof(float)) : nil;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        uint uB = (uint)B, uDin = (uint)D_in, uDout = (uint)D_out;
        uint uBase = (uint)has_base, uBias = (uint)has_bias;

        // 1. Base SiLU and bias initialization (only when needed)
        if (has_base || has_bias) {
            id<MTLComputeCommandEncoder> enc_prep = [cmd computeCommandEncoder];
            [enc_prep setComputePipelineState:g_pipe_prep];
            [enc_prep setBuffer:buf_X offset:0 atIndex:0];
            [enc_prep setBuffer:(buf_bias ? buf_bias : buf_X) offset:0 atIndex:1];
            [enc_prep setBuffer:(buf_X_silu ? buf_X_silu : buf_X) offset:0 atIndex:2];
            [enc_prep setBuffer:buf_Y offset:0 atIndex:3];
            [enc_prep setBytes:&uB length:sizeof(uint) atIndex:4];
            [enc_prep setBytes:&uDin length:sizeof(uint) atIndex:5];
            [enc_prep setBytes:&uDout length:sizeof(uint) atIndex:6];
            [enc_prep setBytes:&uBase length:sizeof(uint) atIndex:7];
            [enc_prep setBytes:&uBias length:sizeof(uint) atIndex:8];
            uint max_dim = (D_in > D_out) ? D_in : D_out;
            [enc_prep dispatchThreads:MTLSizeMake(max_dim, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc_prep endEncoding];
        }

        // 2. Base linear branch: Y += X_silu @ W_base.T
        if (has_base) {
            MPSMatrixDescriptor* desc_A_base = get_cached_desc(B, D_in, D_in * sizeof(float));
            MPSMatrixDescriptor* desc_B_base = get_cached_desc(D_out, D_in, D_in * sizeof(float));
            MPSMatrixDescriptor* desc_C_base = get_cached_desc(B, D_out, D_out * sizeof(float));
            MPSMatrix* mat_A_base = [[[MPSMatrix alloc] initWithBuffer:buf_X_silu descriptor:desc_A_base] autorelease];
            MPSMatrix* mat_B_base = [[[MPSMatrix alloc] initWithBuffer:buf_W_base descriptor:desc_B_base] autorelease];
            MPSMatrix* mat_C_base = [[[MPSMatrix alloc] initWithBuffer:buf_Y descriptor:desc_C_base] autorelease];
            float beta_base = has_bias ? 1.0f : 0.0f;
            MPSMatrixMultiplication* matmul_base = get_cached_matmul(g_device, B, D_out, D_in, 1.0f, beta_base);
            [matmul_base encodeToCommandBuffer:cmd leftMatrix:mat_A_base rightMatrix:mat_B_base resultMatrix:mat_C_base];
        }

        // 3. Fourier basis evaluation
        id<MTLComputeCommandEncoder> enc_four = [cmd computeCommandEncoder];
        [enc_four setComputePipelineState:g_pipe_fourier_basis];
        [enc_four setBuffer:buf_X offset:0 atIndex:0];
        [enc_four setBuffer:buf_Phi offset:0 atIndex:1];
        uint u_freqs = (uint)num_freqs;
        [enc_four setBytes:&uB length:sizeof(uint) atIndex:2];
        [enc_four setBytes:&uDin length:sizeof(uint) atIndex:3];
        [enc_four setBytes:&u_freqs length:sizeof(uint) atIndex:4];
        [enc_four dispatchThreads:MTLSizeMake(D_in, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc_four endEncoding];

        // 4. Matrix multiplication: Y += Phi @ W_fourier.T
        MPSMatrixDescriptor* desc_A_four = get_cached_desc(B, K_dim, K_dim * sizeof(float));
        MPSMatrixDescriptor* desc_B_four = get_cached_desc(D_out, K_dim, K_dim * sizeof(float));
        MPSMatrixDescriptor* desc_C_four = get_cached_desc(B, D_out, D_out * sizeof(float));
        MPSMatrix* mat_A_four = [[[MPSMatrix alloc] initWithBuffer:buf_Phi descriptor:desc_A_four] autorelease];
        MPSMatrix* mat_B_four = [[[MPSMatrix alloc] initWithBuffer:buf_W_four descriptor:desc_B_four] autorelease];
        MPSMatrix* mat_C_four = [[[MPSMatrix alloc] initWithBuffer:buf_Y descriptor:desc_C_four] autorelease];
        float beta_spline = (has_base || has_bias) ? 1.0f : 0.0f;
        MPSMatrixMultiplication* matmul_four = get_cached_matmul(g_device, B, D_out, K_dim, 1.0f, beta_spline);
        [matmul_four encodeToCommandBuffer:cmd leftMatrix:mat_A_four rightMatrix:mat_B_four resultMatrix:mat_C_four];

        [cmd commit];
        [cmd waitUntilCompleted];

        return 0;
    }
}

int metal_kan_jacobi_forward(
    const float* X,
    const float* W_jacobi,
    const float* W_base,
    const float* bias,
    float*       Y,
    int B,
    int D_in,
    int D_out,
    int degree,
    float alpha,
    float beta,
    int has_base,
    int has_bias
) {
    @autoreleasepool {
        int K_dim = D_in * degree;
        id<MTLBuffer> buf_X      = make_no_copy_buffer((void*)X, B * D_in * sizeof(float));
        id<MTLBuffer> buf_W_jac  = make_no_copy_buffer((void*)W_jacobi, D_out * K_dim * sizeof(float));
        id<MTLBuffer> buf_W_base = has_base ? make_no_copy_buffer((void*)W_base, D_out * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_bias   = has_bias ? make_no_copy_buffer((void*)bias, D_out * sizeof(float)) : nil;
        id<MTLBuffer> buf_Y      = make_no_copy_buffer((void*)Y, B * D_out * sizeof(float));

        id<MTLBuffer> buf_Phi    = get_scratch_phi(B * K_dim * sizeof(float));
        id<MTLBuffer> buf_X_silu = has_base ? get_scratch_silu(B * D_in * sizeof(float)) : nil;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        uint uB = (uint)B, uDin = (uint)D_in, uDout = (uint)D_out;
        uint uBase = (uint)has_base, uBias = (uint)has_bias;

        // 1. Base SiLU and bias initialization (only when needed)
        if (has_base || has_bias) {
            id<MTLComputeCommandEncoder> enc_prep = [cmd computeCommandEncoder];
            [enc_prep setComputePipelineState:g_pipe_prep];
            [enc_prep setBuffer:buf_X offset:0 atIndex:0];
            [enc_prep setBuffer:(buf_bias ? buf_bias : buf_X) offset:0 atIndex:1];
            [enc_prep setBuffer:(buf_X_silu ? buf_X_silu : buf_X) offset:0 atIndex:2];
            [enc_prep setBuffer:buf_Y offset:0 atIndex:3];
            [enc_prep setBytes:&uB length:sizeof(uint) atIndex:4];
            [enc_prep setBytes:&uDin length:sizeof(uint) atIndex:5];
            [enc_prep setBytes:&uDout length:sizeof(uint) atIndex:6];
            [enc_prep setBytes:&uBase length:sizeof(uint) atIndex:7];
            [enc_prep setBytes:&uBias length:sizeof(uint) atIndex:8];
            uint max_dim = (D_in > D_out) ? D_in : D_out;
            [enc_prep dispatchThreads:MTLSizeMake(max_dim, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc_prep endEncoding];
        }

        // 2. Base linear branch: Y += X_silu @ W_base.T
        if (has_base) {
            MPSMatrixDescriptor* desc_A_base = get_cached_desc(B, D_in, D_in * sizeof(float));
            MPSMatrixDescriptor* desc_B_base = get_cached_desc(D_out, D_in, D_in * sizeof(float));
            MPSMatrixDescriptor* desc_C_base = get_cached_desc(B, D_out, D_out * sizeof(float));
            MPSMatrix* mat_A_base = [[[MPSMatrix alloc] initWithBuffer:buf_X_silu descriptor:desc_A_base] autorelease];
            MPSMatrix* mat_B_base = [[[MPSMatrix alloc] initWithBuffer:buf_W_base descriptor:desc_B_base] autorelease];
            MPSMatrix* mat_C_base = [[[MPSMatrix alloc] initWithBuffer:buf_Y descriptor:desc_C_base] autorelease];
            float beta_base = has_bias ? 1.0f : 0.0f;
            MPSMatrixMultiplication* matmul_base = get_cached_matmul(g_device, B, D_out, D_in, 1.0f, beta_base);
            [matmul_base encodeToCommandBuffer:cmd leftMatrix:mat_A_base rightMatrix:mat_B_base resultMatrix:mat_C_base];
        }

        // 3. Jacobi basis evaluation
        id<MTLComputeCommandEncoder> enc_jac = [cmd computeCommandEncoder];
        [enc_jac setComputePipelineState:g_pipe_jacobi_basis];
        [enc_jac setBuffer:buf_X offset:0 atIndex:0];
        [enc_jac setBuffer:buf_Phi offset:0 atIndex:1];
        uint u_deg = (uint)degree;
        float u_alpha = alpha, u_beta = beta;
        [enc_jac setBytes:&uB length:sizeof(uint) atIndex:2];
        [enc_jac setBytes:&uDin length:sizeof(uint) atIndex:3];
        [enc_jac setBytes:&u_deg length:sizeof(uint) atIndex:4];
        [enc_jac setBytes:&u_alpha length:sizeof(float) atIndex:5];
        [enc_jac setBytes:&u_beta length:sizeof(float) atIndex:6];
        [enc_jac dispatchThreads:MTLSizeMake(D_in, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc_jac endEncoding];

        // 4. Matrix multiplication: Y += Phi @ W_jacobi.T
        MPSMatrixDescriptor* desc_A_jac = get_cached_desc(B, K_dim, K_dim * sizeof(float));
        MPSMatrixDescriptor* desc_B_jac = get_cached_desc(D_out, K_dim, K_dim * sizeof(float));
        MPSMatrixDescriptor* desc_C_jac = get_cached_desc(B, D_out, D_out * sizeof(float));
        MPSMatrix* mat_A_jac = [[[MPSMatrix alloc] initWithBuffer:buf_Phi descriptor:desc_A_jac] autorelease];
        MPSMatrix* mat_B_jac = [[[MPSMatrix alloc] initWithBuffer:buf_W_jac descriptor:desc_B_jac] autorelease];
        MPSMatrix* mat_C_jac = [[[MPSMatrix alloc] initWithBuffer:buf_Y descriptor:desc_C_jac] autorelease];
        float beta_spline = (has_base || has_bias) ? 1.0f : 0.0f;
        MPSMatrixMultiplication* matmul_jac = get_cached_matmul(g_device, B, D_out, K_dim, 1.0f, beta_spline);
        [matmul_jac encodeToCommandBuffer:cmd leftMatrix:mat_A_jac rightMatrix:mat_B_jac resultMatrix:mat_C_jac];

        [cmd commit];
        [cmd waitUntilCompleted];

        return 0;
    }
}

int metal_kan_rational_forward(
    const float* X,
    const float* W_p,
    const float* W_q,
    const float* W_base,
    const float* bias,
    float*       Y,
    int B,
    int D_in,
    int D_out,
    int p_deg,
    int q_deg,
    int has_base,
    int has_bias
) {
    @autoreleasepool {
        id<MTLComputePipelineState> pipe = g_pipe_rational_tiled;
        if (!pipe) return -1;

        id<MTLBuffer> buf_X      = make_no_copy_buffer((void*)X, B * D_in * sizeof(float));
        id<MTLBuffer> buf_W_p    = make_no_copy_buffer((void*)W_p, D_out * D_in * p_deg * sizeof(float));
        id<MTLBuffer> buf_W_q    = make_no_copy_buffer((void*)W_q, D_out * D_in * q_deg * sizeof(float));
        id<MTLBuffer> buf_W_base = has_base ? make_no_copy_buffer((void*)W_base, D_out * D_in * sizeof(float)) : buf_X;
        id<MTLBuffer> buf_bias   = has_bias ? make_no_copy_buffer((void*)bias, D_out * sizeof(float)) : buf_X;
        id<MTLBuffer> buf_Y      = make_no_copy_buffer((void*)Y, B * D_out * sizeof(float));

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipe];

        [enc setBuffer:buf_X offset:0 atIndex:0];
        [enc setBuffer:buf_W_p offset:0 atIndex:1];
        [enc setBuffer:buf_W_q offset:0 atIndex:2];
        [enc setBuffer:buf_W_base offset:0 atIndex:3];
        [enc setBuffer:buf_bias offset:0 atIndex:4];
        [enc setBuffer:buf_Y offset:0 atIndex:5];

        uint uB = (uint)B;
        uint uD_in = (uint)D_in;
        uint uD_out = (uint)D_out;
        uint u_p = (uint)p_deg;
        uint u_q = (uint)q_deg;
        uint u_has_base = (uint)has_base;
        uint u_has_bias = (uint)has_bias;

        [enc setBytes:&uB length:sizeof(uint) atIndex:6];
        [enc setBytes:&uD_in length:sizeof(uint) atIndex:7];
        [enc setBytes:&uD_out length:sizeof(uint) atIndex:8];
        [enc setBytes:&u_p length:sizeof(uint) atIndex:9];
        [enc setBytes:&u_q length:sizeof(uint) atIndex:10];
        [enc setBytes:&u_has_base length:sizeof(uint) atIndex:11];
        [enc setBytes:&u_has_bias length:sizeof(uint) atIndex:12];

        MTLSize grid_sz = MTLSizeMake((D_out + 31) / 32, (B + 31) / 32, 1);
        MTLSize tg   = MTLSizeMake(16, 16, 1);
        [enc dispatchThreadgroups:grid_sz threadsPerThreadgroup:tg];

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        return 0;
    }
}

int metal_kan_bspline_forward(
    const float* X,
    const float* W_spline,
    const float* W_base,
    const float* grid,
    const float* bias,
    float*       Y,
    int B,
    int D_in,
    int D_out,
    int grid_size,
    int spline_order,
    int has_base,
    int has_bias
) {
    @autoreleasepool {
        uint num_bases = grid_size + 3;
        int K_dim = D_in * num_bases;

        id<MTLBuffer> buf_X      = make_no_copy_buffer((void*)X, B * D_in * sizeof(float));
        id<MTLBuffer> buf_W_spl  = make_no_copy_buffer((void*)W_spline, D_out * K_dim * sizeof(float));
        id<MTLBuffer> buf_W_base = has_base ? make_no_copy_buffer((void*)W_base, D_out * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_bias   = has_bias ? make_no_copy_buffer((void*)bias, D_out * sizeof(float)) : nil;
        id<MTLBuffer> buf_Y      = make_no_copy_buffer((void*)Y, B * D_out * sizeof(float));

        id<MTLBuffer> buf_Phi    = get_scratch_phi(B * K_dim * sizeof(float));
        id<MTLBuffer> buf_X_silu = has_base ? get_scratch_silu(B * D_in * sizeof(float)) : nil;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        uint uB = (uint)B, uDin = (uint)D_in, uDout = (uint)D_out;
        uint uBase = (uint)has_base, uBias = (uint)has_bias;

        // 1. Base SiLU and bias initialization (only when needed)
        if (has_base || has_bias) {
            id<MTLComputeCommandEncoder> enc_prep = [cmd computeCommandEncoder];
            [enc_prep setComputePipelineState:g_pipe_prep];
            [enc_prep setBuffer:buf_X offset:0 atIndex:0];
            [enc_prep setBuffer:(buf_bias ? buf_bias : buf_X) offset:0 atIndex:1];
            [enc_prep setBuffer:(buf_X_silu ? buf_X_silu : buf_X) offset:0 atIndex:2];
            [enc_prep setBuffer:buf_Y offset:0 atIndex:3];
            [enc_prep setBytes:&uB length:sizeof(uint) atIndex:4];
            [enc_prep setBytes:&uDin length:sizeof(uint) atIndex:5];
            [enc_prep setBytes:&uDout length:sizeof(uint) atIndex:6];
            [enc_prep setBytes:&uBase length:sizeof(uint) atIndex:7];
            [enc_prep setBytes:&uBias length:sizeof(uint) atIndex:8];
            uint max_dim = (D_in > D_out) ? D_in : D_out;
            [enc_prep dispatchThreads:MTLSizeMake(max_dim, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc_prep endEncoding];
        }

        // 2. Base linear branch: Y += X_silu @ W_base.T
        if (has_base) {
            MPSMatrixDescriptor* desc_A_base = get_cached_desc(B, D_in, D_in * sizeof(float));
            MPSMatrixDescriptor* desc_B_base = get_cached_desc(D_out, D_in, D_in * sizeof(float));
            MPSMatrixDescriptor* desc_C_base = get_cached_desc(B, D_out, D_out * sizeof(float));
            MPSMatrix* mat_A_base = [[[MPSMatrix alloc] initWithBuffer:buf_X_silu descriptor:desc_A_base] autorelease];
            MPSMatrix* mat_B_base = [[[MPSMatrix alloc] initWithBuffer:buf_W_base descriptor:desc_B_base] autorelease];
            MPSMatrix* mat_C_base = [[[MPSMatrix alloc] initWithBuffer:buf_Y descriptor:desc_C_base] autorelease];
            float beta_base = has_bias ? 1.0f : 0.0f;
            MPSMatrixMultiplication* matmul_base = get_cached_matmul(g_device, B, D_out, D_in, 1.0f, beta_base);
            [matmul_base encodeToCommandBuffer:cmd leftMatrix:mat_A_base rightMatrix:mat_B_base resultMatrix:mat_C_base];
        }

        // 3. BSpline basis evaluation
        id<MTLComputeCommandEncoder> enc_spl = [cmd computeCommandEncoder];
        [enc_spl setComputePipelineState:g_pipe_bspline_basis];
        [enc_spl setBuffer:buf_X offset:0 atIndex:0];
        [enc_spl setBuffer:buf_Phi offset:0 atIndex:1];
        uint u_gsize = (uint)grid_size;
        float grid_min = grid[0];
        float inv_h = grid[1];
        [enc_spl setBytes:&uB length:sizeof(uint) atIndex:2];
        [enc_spl setBytes:&uDin length:sizeof(uint) atIndex:3];
        [enc_spl setBytes:&u_gsize length:sizeof(uint) atIndex:4];
        [enc_spl setBytes:&grid_min length:sizeof(float) atIndex:5];
        [enc_spl setBytes:&inv_h length:sizeof(float) atIndex:6];
        [enc_spl dispatchThreads:MTLSizeMake(D_in, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc_spl endEncoding];

        // 4. Matrix multiplication: Y += Phi @ W_spline.T
        MPSMatrixDescriptor* desc_A_spl = get_cached_desc(B, K_dim, K_dim * sizeof(float));
        MPSMatrixDescriptor* desc_B_spl = get_cached_desc(D_out, K_dim, K_dim * sizeof(float));
        MPSMatrixDescriptor* desc_C_spl = get_cached_desc(B, D_out, D_out * sizeof(float));
        MPSMatrix* mat_A_spl = [[[MPSMatrix alloc] initWithBuffer:buf_Phi descriptor:desc_A_spl] autorelease];
        MPSMatrix* mat_B_spl = [[[MPSMatrix alloc] initWithBuffer:buf_W_spl descriptor:desc_B_spl] autorelease];
        MPSMatrix* mat_C_spl = [[[MPSMatrix alloc] initWithBuffer:buf_Y descriptor:desc_C_spl] autorelease];
        float beta_spline = (has_base || has_bias) ? 1.0f : 0.0f;
        MPSMatrixMultiplication* matmul_spl = get_cached_matmul(g_device, B, D_out, K_dim, 1.0f, beta_spline);
        [matmul_spl encodeToCommandBuffer:cmd leftMatrix:mat_A_spl rightMatrix:mat_B_spl resultMatrix:mat_C_spl];

        [cmd commit];
        [cmd waitUntilCompleted];

        return 0;
    }
}

// Chained multi-layer GPU execution: dispatches all layers in a single command buffer
// Ping-pong intermediate Metal buffers, eliminating CPU sync roundtrips between layers.
int metal_kan_chain_pipeline_cheby(
    const float* X,
    float*       Y,
    int B,
    int num_layers,
    const int* layer_dims, // length num_layers + 1
    const int* degrees,    // length num_layers
    const float** W_chebys,
    const float** W_bases,
    const float** biases
) {
    @autoreleasepool {
        id<MTLComputePipelineState> pipe = g_pipe_cheby_tiled;
        if (!pipe || num_layers <= 0) return -1;

        // Find max hidden dimension for ping-pong buffer
        int max_dim = 0;
        for (int i = 0; i <= num_layers; i++) {
            if (layer_dims[i] > max_dim) max_dim = layer_dims[i];
        }

        id<MTLBuffer> buf_in  = make_no_copy_buffer((void*)X, B * layer_dims[0] * sizeof(float));
        id<MTLBuffer> buf_out = make_no_copy_buffer((void*)Y, B * layer_dims[num_layers] * sizeof(float));

        id<MTLBuffer> buf_ping = [g_device newBufferWithLength:B * max_dim * sizeof(float) options:MTLResourceStorageModePrivate];
        id<MTLBuffer> buf_pong = [g_device newBufferWithLength:B * max_dim * sizeof(float) options:MTLResourceStorageModePrivate];

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];

        id<MTLBuffer> current_src = buf_in;
        for (int l = 0; l < num_layers; l++) {
            int d_in = layer_dims[l];
            int d_out = layer_dims[l + 1];
            int deg = degrees[l];

            id<MTLBuffer> current_dst;
            if (l == num_layers - 1) {
                current_dst = buf_out;
            } else {
                current_dst = (l % 2 == 0) ? buf_pong : buf_ping;
            }

            id<MTLBuffer> buf_w = make_no_copy_buffer((void*)W_chebys[l], d_out * d_in * 4 * sizeof(float));
            id<MTLBuffer> buf_wb = (W_bases && W_bases[l]) ? make_no_copy_buffer((void*)W_bases[l], d_out * d_in * sizeof(float)) : current_src;
            id<MTLBuffer> buf_b  = (biases && biases[l]) ? make_no_copy_buffer((void*)biases[l], d_out * sizeof(float)) : current_src;

            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipe];

            [enc setBuffer:current_src offset:0 atIndex:0];
            [enc setBuffer:buf_w offset:0 atIndex:1];
            [enc setBuffer:buf_wb offset:0 atIndex:2];
            [enc setBuffer:buf_b offset:0 atIndex:3];
            [enc setBuffer:current_dst offset:0 atIndex:4];

            uint uB = (uint)B;
            uint uD_in = (uint)d_in;
            uint uD_out = (uint)d_out;
            uint u_has_base = (W_bases && W_bases[l]) ? 1 : 0;
            uint u_has_bias = (biases && biases[l]) ? 1 : 0;

            [enc setBytes:&uB length:sizeof(uint) atIndex:5];
            [enc setBytes:&uD_in length:sizeof(uint) atIndex:6];
            [enc setBytes:&uD_out length:sizeof(uint) atIndex:7];
            [enc setBytes:&u_has_base length:sizeof(uint) atIndex:8];
            [enc setBytes:&u_has_bias length:sizeof(uint) atIndex:9];

            MTLSize grid = MTLSizeMake((d_out + 31) / 32, (B + 31) / 32, 1);
            MTLSize tg   = MTLSizeMake(16, 16, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];

            current_src = current_dst;
        }

        [cmd commit];
        [cmd waitUntilCompleted];

        return 0;
    }
}

double benchmark_metal_cheby(
    const float* X,
    const float* W_cheby,
    const float* W_base,
    const float* bias,
    float*       Y,
    int B,
    int D_in,
    int D_out,
    int K,
    int has_base,
    int has_bias,
    int warmup,
    int iters
) {
    for (int i = 0; i < warmup; i++) {
        metal_kan_cheby_forward(X, W_cheby, W_base, bias, Y, B, D_in, D_out, K, has_base, has_bias);
    }

    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++) {
        metal_kan_cheby_forward(X, W_cheby, W_base, bias, Y, B, D_in, D_out, K, has_base, has_bias);
    }
    uint64_t t1 = mach_absolute_time();

    double total_sec = (double)(t1 - t0) * g_timebase_factor;
    return (total_sec / (double)iters) * 1000.0;
}

double benchmark_metal_wavkan(
    const float* X,
    const float* W_wav,
    const float* W_base,
    const float* translation,
    const float* inv_scale,
    const float* bias,
    float*       Y,
    int B,
    int D_in,
    int D_out,
    int num_wavelets,
    int wavelet_type,
    int has_base,
    int has_bias,
    int warmup,
    int iters
) {
    for (int i = 0; i < warmup; i++) {
        metal_kan_wavkan_forward(X, W_wav, W_base, translation, inv_scale, bias, Y, B, D_in, D_out, num_wavelets, wavelet_type, has_base, has_bias);
    }

    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++) {
        metal_kan_wavkan_forward(X, W_wav, W_base, translation, inv_scale, bias, Y, B, D_in, D_out, num_wavelets, wavelet_type, has_base, has_bias);
    }
    uint64_t t1 = mach_absolute_time();

    double total_sec = (double)(t1 - t0) * g_timebase_factor;
    return (total_sec / (double)iters) * 1000.0;
}

double benchmark_metal_bspline(
    const float* X,
    const float* W_spline,
    const float* W_base,
    const float* grid,
    const float* bias,
    float*       Y,
    int B,
    int D_in,
    int D_out,
    int grid_size,
    int spline_order,
    int has_base,
    int has_bias,
    int warmup,
    int iters
) {
    for (int i = 0; i < warmup; i++) {
        metal_kan_bspline_forward(X, W_spline, W_base, grid, bias, Y, B, D_in, D_out, grid_size, spline_order, has_base, has_bias);
    }

    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++) {
        metal_kan_bspline_forward(X, W_spline, W_base, grid, bias, Y, B, D_in, D_out, grid_size, spline_order, has_base, has_bias);
    }
    uint64_t t1 = mach_absolute_time();

    double total_sec = (double)(t1 - t0) * g_timebase_factor;
    return (total_sec / (double)iters) * 1000.0;
}

double benchmark_metal_fastkan(
    const float* X,
    const float* W_rbf,
    const float* W_base,
    const float* grid,
    const float* bias,
    float*       Y,
    int B,
    int D_in,
    int D_out,
    int num_centers,
    float inv_denominator,
    int has_base,
    int has_bias,
    int warmup,
    int iters
) {
    for (int i = 0; i < warmup; i++) {
        metal_kan_fastkan_forward(X, W_rbf, W_base, grid, bias, Y, B, D_in, D_out, num_centers, inv_denominator, has_base, has_bias);
    }

    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++) {
        metal_kan_fastkan_forward(X, W_rbf, W_base, grid, bias, Y, B, D_in, D_out, num_centers, inv_denominator, has_base, has_bias);
    }
    uint64_t t1 = mach_absolute_time();

    double total_sec = (double)(t1 - t0) * g_timebase_factor;
    return (total_sec / (double)iters) * 1000.0;
}

int metal_kan_combine_mult_nodes(
    const float* In,
    float*       Out,
    int B,
    int num_add,
    int num_mult
) {
    @autoreleasepool {
        id<MTLComputePipelineState> pipe = g_pipe_combine_mult;
        if (!pipe) return -1;

        int total_in = num_add + 2 * num_mult;
        int total_out = num_add + num_mult;

        id<MTLBuffer> buf_In  = make_no_copy_buffer((void*)In, B * total_in * sizeof(float));
        id<MTLBuffer> buf_Out = make_no_copy_buffer((void*)Out, B * total_out * sizeof(float));

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipe];
        [enc setBuffer:buf_In offset:0 atIndex:0];
        [enc setBuffer:buf_Out offset:0 atIndex:1];

        uint uB = (uint)B;
        uint uAdd = (uint)num_add;
        uint uMult = (uint)num_mult;
        [enc setBytes:&uB length:sizeof(uint) atIndex:2];
        [enc setBytes:&uAdd length:sizeof(uint) atIndex:3];
        [enc setBytes:&uMult length:sizeof(uint) atIndex:4];

        [enc dispatchThreads:MTLSizeMake(total_out, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];

        return 0;
    }
}

double benchmark_metal_relu(
    const float* X,
    const float* W_relu,
    const float* W_base,
    const float* grid,
    const float* bias,
    float*       Y,
    int B,
    int D_in,
    int D_out,
    int num_grids,
    float inv_h,
    int has_base,
    int has_bias,
    int warmup,
    int iters
) {
    for (int i = 0; i < warmup; i++) {
        metal_kan_relu_forward(X, W_relu, W_base, grid, bias, Y, B, D_in, D_out, num_grids, inv_h, has_base, has_bias);
    }

    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++) {
        metal_kan_relu_forward(X, W_relu, W_base, grid, bias, Y, B, D_in, D_out, num_grids, inv_h, has_base, has_bias);
    }
    uint64_t t1 = mach_absolute_time();

    double total_sec = (double)(t1 - t0) * g_timebase_factor;
    return (total_sec / (double)iters) * 1000.0;
}

double benchmark_metal_fourier(
    const float* X,
    const float* W_fourier,
    const float* W_base,
    const float* bias,
    float*       Y,
    int B,
    int D_in,
    int D_out,
    int num_freqs,
    int has_base,
    int has_bias,
    int warmup,
    int iters
) {
    for (int i = 0; i < warmup; i++) {
        metal_kan_fourier_forward(X, W_fourier, W_base, bias, Y, B, D_in, D_out, num_freqs, has_base, has_bias);
    }

    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++) {
        metal_kan_fourier_forward(X, W_fourier, W_base, bias, Y, B, D_in, D_out, num_freqs, has_base, has_bias);
    }
    uint64_t t1 = mach_absolute_time();

    double total_sec = (double)(t1 - t0) * g_timebase_factor;
    return (total_sec / (double)iters) * 1000.0;
}

double benchmark_metal_jacobi(
    const float* X,
    const float* W_jacobi,
    const float* W_base,
    const float* bias,
    float*       Y,
    int B,
    int D_in,
    int D_out,
    int degree,
    float alpha,
    float beta,
    int has_base,
    int has_bias,
    int warmup,
    int iters
) {
    for (int i = 0; i < warmup; i++) {
        metal_kan_jacobi_forward(X, W_jacobi, W_base, bias, Y, B, D_in, D_out, degree, alpha, beta, has_base, has_bias);
    }

    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++) {
        metal_kan_jacobi_forward(X, W_jacobi, W_base, bias, Y, B, D_in, D_out, degree, alpha, beta, has_base, has_bias);
    }
    uint64_t t1 = mach_absolute_time();

    double total_sec = (double)(t1 - t0) * g_timebase_factor;
    return (total_sec / (double)iters) * 1000.0;
}

double benchmark_metal_rational(
    const float* X,
    const float* W_p,
    const float* W_q,
    const float* W_base,
    const float* bias,
    float*       Y,
    int B,
    int D_in,
    int D_out,
    int p_deg,
    int q_deg,
    int has_base,
    int has_bias,
    int warmup,
    int iters
) {
    for (int i = 0; i < warmup; i++) {
        metal_kan_rational_forward(X, W_p, W_q, W_base, bias, Y, B, D_in, D_out, p_deg, q_deg, has_base, has_bias);
    }

    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++) {
        metal_kan_rational_forward(X, W_p, W_q, W_base, bias, Y, B, D_in, D_out, p_deg, q_deg, has_base, has_bias);
    }
    uint64_t t1 = mach_absolute_time();

    double total_sec = (double)(t1 - t0) * g_timebase_factor;
    return (total_sec / (double)iters) * 1000.0;
}

int metal_kan_lowrank_forward(
    const float* X,
    const float* W_down,
    const float* W_up,
    const float* W_base,
    const float* grid,
    const float* bias,
    float*       Y,
    int B,
    int D_in,
    int D_out,
    int rank,
    int num_centers,
    float inv_denominator,
    int has_base,
    int has_bias
) {
    @autoreleasepool {
        int K_dim = D_in * num_centers;
        id<MTLBuffer> buf_X      = make_no_copy_buffer((void*)X, B * D_in * sizeof(float));
        id<MTLBuffer> buf_W_down = make_no_copy_buffer((void*)W_down, rank * K_dim * sizeof(float));
        id<MTLBuffer> buf_W_up   = make_no_copy_buffer((void*)W_up, D_out * rank * sizeof(float));
        id<MTLBuffer> buf_W_base = has_base ? make_no_copy_buffer((void*)W_base, D_out * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_grid   = make_no_copy_buffer((void*)grid, num_centers * sizeof(float));
        id<MTLBuffer> buf_bias   = has_bias ? make_no_copy_buffer((void*)bias, D_out * sizeof(float)) : nil;
        id<MTLBuffer> buf_Y      = make_no_copy_buffer((void*)Y, B * D_out * sizeof(float));

        id<MTLBuffer> buf_Phi    = get_scratch_phi(B * K_dim * sizeof(float));
        id<MTLBuffer> buf_Z      = get_scratch_bottleneck(B * rank * sizeof(float));
        id<MTLBuffer> buf_X_silu = has_base ? get_scratch_silu(B * D_in * sizeof(float)) : nil;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];

        // Fast-path: Direct fused SIMD kernel for small/medium batches (eliminates 3 MPS calls)
        if (B <= 256 && D_in <= 128 && rank <= 64 && g_pipe_lowrank_simd_fp32) {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:g_pipe_lowrank_simd_fp32];
            [enc setBuffer:buf_X offset:0 atIndex:0];
            [enc setBuffer:buf_W_down offset:0 atIndex:1];
            [enc setBuffer:buf_W_up offset:0 atIndex:2];
            [enc setBuffer:(buf_W_base ? buf_W_base : buf_X) offset:0 atIndex:3];
            [enc setBuffer:buf_grid offset:0 atIndex:4];
            [enc setBuffer:(buf_bias ? buf_bias : buf_X) offset:0 atIndex:5];
            [enc setBuffer:buf_Y offset:0 atIndex:6];
            uint uB = (uint)B, uDin = (uint)D_in, uDout = (uint)D_out;
            uint urank = (uint)rank, uK = (uint)num_centers;
            float u_inv_den = inv_denominator;
            uint uBase = (uint)has_base, uBias = (uint)has_bias;
            [enc setBytes:&uB length:sizeof(uint) atIndex:7];
            [enc setBytes:&uDin length:sizeof(uint) atIndex:8];
            [enc setBytes:&uDout length:sizeof(uint) atIndex:9];
            [enc setBytes:&urank length:sizeof(uint) atIndex:10];
            [enc setBytes:&uK length:sizeof(uint) atIndex:11];
            [enc setBytes:&u_inv_den length:sizeof(float) atIndex:12];
            [enc setBytes:&uBase length:sizeof(uint) atIndex:13];
            [enc setBytes:&uBias length:sizeof(uint) atIndex:14];
            [enc dispatchThreadgroups:MTLSizeMake(1, B, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [enc endEncoding];
            commit_and_sync(cmd);
            return 0;
        }

        // 1. Base SiLU and bias initialization (only when needed)
        if (has_base || has_bias) {
            id<MTLComputeCommandEncoder> enc_prep = [cmd computeCommandEncoder];
            [enc_prep setComputePipelineState:g_pipe_prep];
            [enc_prep setBuffer:buf_X offset:0 atIndex:0];
            [enc_prep setBuffer:(buf_bias ? buf_bias : buf_X) offset:0 atIndex:1];
            [enc_prep setBuffer:(buf_X_silu ? buf_X_silu : buf_X) offset:0 atIndex:2];
            [enc_prep setBuffer:buf_Y offset:0 atIndex:3];
            uint uB = (uint)B, uDin = (uint)D_in, uDout = (uint)D_out;
            uint uBase = (uint)has_base, uBias = (uint)has_bias;
            [enc_prep setBytes:&uB length:sizeof(uint) atIndex:4];
            [enc_prep setBytes:&uDin length:sizeof(uint) atIndex:5];
            [enc_prep setBytes:&uDout length:sizeof(uint) atIndex:6];
            [enc_prep setBytes:&uBase length:sizeof(uint) atIndex:7];
            [enc_prep setBytes:&uBias length:sizeof(uint) atIndex:8];
            uint max_dim = (D_in > D_out) ? D_in : D_out;
            [enc_prep dispatchThreads:MTLSizeMake(max_dim, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc_prep endEncoding];
        }

        // 2. FastKAN RBF basis evaluation: Phi from X
        id<MTLComputeCommandEncoder> enc_rbf = [cmd computeCommandEncoder];
        [enc_rbf setComputePipelineState:g_pipe_rbf_basis];
        [enc_rbf setBuffer:buf_X offset:0 atIndex:0];
        [enc_rbf setBuffer:buf_grid offset:0 atIndex:1];
        [enc_rbf setBuffer:buf_Phi offset:0 atIndex:2];
        uint uB = (uint)B, uDin = (uint)D_in, uK = (uint)num_centers;
        float u_inv_den = inv_denominator;
        [enc_rbf setBytes:&uB length:sizeof(uint) atIndex:3];
        [enc_rbf setBytes:&uDin length:sizeof(uint) atIndex:4];
        [enc_rbf setBytes:&uK length:sizeof(uint) atIndex:5];
        [enc_rbf setBytes:&u_inv_den length:sizeof(float) atIndex:6];
        [enc_rbf dispatchThreads:MTLSizeMake(D_in, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc_rbf endEncoding];

        // 3. Down-projection GEMM: Z = Phi @ W_down.T [B, rank]
        MPSMatrixDescriptor* desc_Phi = get_cached_desc(B, K_dim, K_dim * sizeof(float));
        MPSMatrixDescriptor* desc_Wdown = get_cached_desc(rank, K_dim, K_dim * sizeof(float));
        MPSMatrixDescriptor* desc_Z = get_cached_desc(B, rank, rank * sizeof(float));
        MPSMatrix* mat_Phi = [[[MPSMatrix alloc] initWithBuffer:buf_Phi descriptor:desc_Phi] autorelease];
        MPSMatrix* mat_Wdown = [[[MPSMatrix alloc] initWithBuffer:buf_W_down descriptor:desc_Wdown] autorelease];
        MPSMatrix* mat_Z = [[[MPSMatrix alloc] initWithBuffer:buf_Z descriptor:desc_Z] autorelease];
        MPSMatrixMultiplication* matmul_down = get_cached_matmul(g_device, B, rank, K_dim, 1.0f, 0.0f);
        [matmul_down encodeToCommandBuffer:cmd leftMatrix:mat_Phi rightMatrix:mat_Wdown resultMatrix:mat_Z];

        // 4. Up-projection GEMM: Y = Z @ W_up.T (+ bias if has_bias)
        MPSMatrixDescriptor* desc_Wup = get_cached_desc(D_out, rank, rank * sizeof(float));
        MPSMatrixDescriptor* desc_Y = get_cached_desc(B, D_out, D_out * sizeof(float));
        MPSMatrix* mat_Wup = [[[MPSMatrix alloc] initWithBuffer:buf_W_up descriptor:desc_Wup] autorelease];
        MPSMatrix* mat_Y = [[[MPSMatrix alloc] initWithBuffer:buf_Y descriptor:desc_Y] autorelease];
        float beta_up = has_bias ? 1.0f : 0.0f;
        MPSMatrixMultiplication* matmul_up = get_cached_matmul(g_device, B, D_out, rank, 1.0f, beta_up);
        [matmul_up encodeToCommandBuffer:cmd leftMatrix:mat_Z rightMatrix:mat_Wup resultMatrix:mat_Y];

        // 5. Base linear branch: Y += X_silu @ W_base.T
        if (has_base) {
            MPSMatrixDescriptor* desc_A_base = get_cached_desc(B, D_in, D_in * sizeof(float));
            MPSMatrixDescriptor* desc_B_base = get_cached_desc(D_out, D_in, D_in * sizeof(float));
            MPSMatrix* mat_A_base = [[[MPSMatrix alloc] initWithBuffer:buf_X_silu descriptor:desc_A_base] autorelease];
            MPSMatrix* mat_B_base = [[[MPSMatrix alloc] initWithBuffer:buf_W_base descriptor:desc_B_base] autorelease];
            MPSMatrixMultiplication* matmul_base = get_cached_matmul(g_device, B, D_out, D_in, 1.0f, 1.0f);
            [matmul_base encodeToCommandBuffer:cmd leftMatrix:mat_A_base rightMatrix:mat_B_base resultMatrix:mat_Y];
        }

        commit_and_sync(cmd);

        return 0;
    }
}

double benchmark_metal_lowrank(
    const float* X,
    const float* W_down,
    const float* W_up,
    const float* W_base,
    const float* grid,
    const float* bias,
    float*       Y,
    int B,
    int D_in,
    int D_out,
    int rank,
    int num_centers,
    float inv_denominator,
    int has_base,
    int has_bias,
    int warmup,
    int iters
) {
    for (int i = 0; i < warmup; i++) {
        metal_kan_lowrank_forward(X, W_down, W_up, W_base, grid, bias, Y, B, D_in, D_out, rank, num_centers, inv_denominator, has_base, has_bias);
    }

    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++) {
        metal_kan_lowrank_forward(X, W_down, W_up, W_base, grid, bias, Y, B, D_in, D_out, rank, num_centers, inv_denominator, has_base, has_bias);
    }
    uint64_t t1 = mach_absolute_time();

    double total_sec = (double)(t1 - t0) * g_timebase_factor;
    return (total_sec / (double)iters) * 1000.0;
}

int metal_kan_mult_forward(
    const float* X,
    const float* W_rbf,
    const float* W_base,
    const float* grid,
    const float* bias,
    float*       Y,
    int B,
    int D_in,
    int D_out,
    int num_add,
    int num_mult,
    int num_centers,
    float inv_denominator,
    int has_base,
    int has_bias
) {
    @autoreleasepool {
        int internal_out = num_add + 2 * num_mult;
        int K_dim = D_in * num_centers;
        id<MTLBuffer> buf_X      = make_no_copy_buffer((void*)X, B * D_in * sizeof(float));
        id<MTLBuffer> buf_W_rbf  = make_no_copy_buffer((void*)W_rbf, internal_out * K_dim * sizeof(float));
        id<MTLBuffer> buf_W_base = has_base ? make_no_copy_buffer((void*)W_base, internal_out * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_grid   = make_no_copy_buffer((void*)grid, num_centers * sizeof(float));
        id<MTLBuffer> buf_bias   = has_bias ? make_no_copy_buffer((void*)bias, internal_out * sizeof(float)) : nil;
        id<MTLBuffer> buf_Y      = make_no_copy_buffer((void*)Y, B * D_out * sizeof(float));

        id<MTLBuffer> buf_Phi      = get_scratch_phi(B * K_dim * sizeof(float));
        id<MTLBuffer> buf_internal = (num_mult > 0) ? get_scratch_internal(B * internal_out * sizeof(float)) : buf_Y;
        id<MTLBuffer> buf_X_silu   = has_base ? get_scratch_silu(B * D_in * sizeof(float)) : nil;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];

        // 1. Base SiLU and bias initialization into buf_internal
        if (has_base || has_bias) {
            id<MTLComputeCommandEncoder> enc_prep = [cmd computeCommandEncoder];
            [enc_prep setComputePipelineState:g_pipe_prep];
            [enc_prep setBuffer:buf_X offset:0 atIndex:0];
            [enc_prep setBuffer:(buf_bias ? buf_bias : buf_X) offset:0 atIndex:1];
            [enc_prep setBuffer:(buf_X_silu ? buf_X_silu : buf_X) offset:0 atIndex:2];
            [enc_prep setBuffer:buf_internal offset:0 atIndex:3];
            uint uB = (uint)B, uDin = (uint)D_in, uDout = (uint)internal_out;
            uint uBase = (uint)has_base, uBias = (uint)has_bias;
            [enc_prep setBytes:&uB length:sizeof(uint) atIndex:4];
            [enc_prep setBytes:&uDin length:sizeof(uint) atIndex:5];
            [enc_prep setBytes:&uDout length:sizeof(uint) atIndex:6];
            [enc_prep setBytes:&uBase length:sizeof(uint) atIndex:7];
            [enc_prep setBytes:&uBias length:sizeof(uint) atIndex:8];
            uint max_dim = (D_in > internal_out) ? D_in : internal_out;
            [enc_prep dispatchThreads:MTLSizeMake(max_dim, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc_prep endEncoding];
        }

        // 2. Base linear branch: internal += X_silu @ W_base.T
        if (has_base) {
            MPSMatrixDescriptor* desc_A_base = get_cached_desc(B, D_in, D_in * sizeof(float));
            MPSMatrixDescriptor* desc_B_base = get_cached_desc(internal_out, D_in, D_in * sizeof(float));
            MPSMatrixDescriptor* desc_C_base = get_cached_desc(B, internal_out, internal_out * sizeof(float));
            MPSMatrix* mat_A_base = [[[MPSMatrix alloc] initWithBuffer:buf_X_silu descriptor:desc_A_base] autorelease];
            MPSMatrix* mat_B_base = [[[MPSMatrix alloc] initWithBuffer:buf_W_base descriptor:desc_B_base] autorelease];
            MPSMatrix* mat_C_base = [[[MPSMatrix alloc] initWithBuffer:buf_internal descriptor:desc_C_base] autorelease];
            float beta_base = has_bias ? 1.0f : 0.0f;
            MPSMatrixMultiplication* matmul_base = get_cached_matmul(g_device, B, internal_out, D_in, 1.0f, beta_base);
            [matmul_base encodeToCommandBuffer:cmd leftMatrix:mat_A_base rightMatrix:mat_B_base resultMatrix:mat_C_base];
        }

        // 3. FastKAN RBF basis evaluation
        id<MTLComputeCommandEncoder> enc_rbf = [cmd computeCommandEncoder];
        [enc_rbf setComputePipelineState:g_pipe_rbf_basis];
        [enc_rbf setBuffer:buf_X offset:0 atIndex:0];
        [enc_rbf setBuffer:buf_grid offset:0 atIndex:1];
        [enc_rbf setBuffer:buf_Phi offset:0 atIndex:2];
        uint uB = (uint)B, uDin = (uint)D_in, uK = (uint)num_centers;
        float u_inv_den = inv_denominator;
        [enc_rbf setBytes:&uB length:sizeof(uint) atIndex:3];
        [enc_rbf setBytes:&uDin length:sizeof(uint) atIndex:4];
        [enc_rbf setBytes:&uK length:sizeof(uint) atIndex:5];
        [enc_rbf setBytes:&u_inv_den length:sizeof(float) atIndex:6];
        [enc_rbf dispatchThreads:MTLSizeMake(D_in, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc_rbf endEncoding];

        // 4. Matrix multiplication: internal += Phi @ W_rbf.T
        MPSMatrixDescriptor* desc_A_rbf = get_cached_desc(B, K_dim, K_dim * sizeof(float));
        MPSMatrixDescriptor* desc_B_rbf = get_cached_desc(internal_out, K_dim, K_dim * sizeof(float));
        MPSMatrixDescriptor* desc_C_rbf = get_cached_desc(B, internal_out, internal_out * sizeof(float));
        MPSMatrix* mat_A_rbf = [[[MPSMatrix alloc] initWithBuffer:buf_Phi descriptor:desc_A_rbf] autorelease];
        MPSMatrix* mat_B_rbf = [[[MPSMatrix alloc] initWithBuffer:buf_W_rbf descriptor:desc_B_rbf] autorelease];
        MPSMatrix* mat_C_rbf = [[[MPSMatrix alloc] initWithBuffer:buf_internal descriptor:desc_C_rbf] autorelease];
        float beta_rbf = (has_base || has_bias) ? 1.0f : 0.0f;
        MPSMatrixMultiplication* matmul_rbf = get_cached_matmul(g_device, B, internal_out, K_dim, 1.0f, beta_rbf);
        [matmul_rbf encodeToCommandBuffer:cmd leftMatrix:mat_A_rbf rightMatrix:mat_B_rbf resultMatrix:mat_C_rbf];

        // 5. Combine multiplicative nodes: internal -> Y
        if (num_mult > 0) {
            id<MTLComputeCommandEncoder> enc_comb = [cmd computeCommandEncoder];
            [enc_comb setComputePipelineState:g_pipe_combine_mult];
            [enc_comb setBuffer:buf_internal offset:0 atIndex:0];
            [enc_comb setBuffer:buf_Y offset:0 atIndex:1];
            uint uB = (uint)B;
            uint uAdd = (uint)num_add;
            uint uMult = (uint)num_mult;
            [enc_comb setBytes:&uB length:sizeof(uint) atIndex:2];
            [enc_comb setBytes:&uAdd length:sizeof(uint) atIndex:3];
            [enc_comb setBytes:&uMult length:sizeof(uint) atIndex:4];
            [enc_comb dispatchThreads:MTLSizeMake(D_out, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc_comb endEncoding];
        }

        commit_and_sync(cmd);

        return 0;
    }
}

double benchmark_metal_mult(
    const float* X,
    const float* W_rbf,
    const float* W_base,
    const float* grid,
    const float* bias,
    float*       Y,
    int B,
    int D_in,
    int D_out,
    int num_add,
    int num_mult,
    int num_centers,
    float inv_denominator,
    int has_base,
    int has_bias,
    int warmup,
    int iters
) {
    for (int i = 0; i < warmup; i++) {
        metal_kan_mult_forward(X, W_rbf, W_base, grid, bias, Y, B, D_in, D_out, num_add, num_mult, num_centers, inv_denominator, has_base, has_bias);
    }

    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++) {
        metal_kan_mult_forward(X, W_rbf, W_base, grid, bias, Y, B, D_in, D_out, num_add, num_mult, num_centers, inv_denominator, has_base, has_bias);
    }
    uint64_t t1 = mach_absolute_time();

    double total_sec = (double)(t1 - t0) * g_timebase_factor;
    return (total_sec / (double)iters) * 1000.0;
}

// Direct simdgroup_matrix FP32 GEMM export.

int metal_kan_gemm_simd_fp32(
    const float* A,
    const float* B_mat,
    const float* bias,
    float*       C,
    int M,
    int N,
    int K,
    float alpha,
    float beta,
    int has_bias
) {
    @autoreleasepool {
        id<MTLBuffer> buf_A = make_no_copy_buffer((void*)A, M * K * sizeof(float));
        id<MTLBuffer> buf_B = make_no_copy_buffer((void*)B_mat, N * K * sizeof(float));
        id<MTLBuffer> buf_bias = has_bias ? make_no_copy_buffer((void*)bias, N * sizeof(float)) : nil;
        id<MTLBuffer> buf_C = make_no_copy_buffer((void*)C, M * N * sizeof(float));

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_pipe_gemm_simd_fp32];
        [enc setBuffer:buf_A offset:0 atIndex:0];
        [enc setBuffer:buf_B offset:0 atIndex:1];
        [enc setBuffer:(buf_bias ? buf_bias : buf_A) offset:0 atIndex:2];
        [enc setBuffer:buf_C offset:0 atIndex:3];
        uint uM = (uint)M, uN = (uint)N, uK = (uint)K, uHb = (uint)has_bias;
        [enc setBytes:&uM length:sizeof(uint) atIndex:4];
        [enc setBytes:&uN length:sizeof(uint) atIndex:5];
        [enc setBytes:&uK length:sizeof(uint) atIndex:6];
        [enc setBytes:&alpha length:sizeof(float) atIndex:7];
        [enc setBytes:&beta length:sizeof(float) atIndex:8];
        [enc setBytes:&uHb length:sizeof(uint) atIndex:9];
        [enc dispatchThreadgroups:MTLSizeMake((N + 15) / 16, (M + 15) / 16, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc endEncoding];
        commit_and_sync(cmd);
        return 0;
    }
}

// FP16 half-precision forward kernels.

int metal_kan_cheby_forward_fp16(
    const uint16_t* X,
    const uint16_t* W_cheby,
    const uint16_t* W_base,
    const uint16_t* bias,
    uint16_t*       Y,
    int B,
    int D_in,
    int D_out,
    int K,
    int has_base,
    int has_bias
) {
    @autoreleasepool {
        int K_dim = D_in * K;
        id<MTLBuffer> buf_X       = make_no_copy_buffer((void*)X, B * D_in * sizeof(uint16_t));
        id<MTLBuffer> buf_W_cheby = make_no_copy_buffer((void*)W_cheby, D_out * K_dim * sizeof(uint16_t));
        id<MTLBuffer> buf_W_base  = has_base ? make_no_copy_buffer((void*)W_base, D_out * D_in * sizeof(uint16_t)) : nil;
        id<MTLBuffer> buf_bias    = has_bias ? make_no_copy_buffer((void*)bias, D_out * sizeof(uint16_t)) : nil;
        id<MTLBuffer> buf_Y       = make_no_copy_buffer((void*)Y, B * D_out * sizeof(uint16_t));

        id<MTLBuffer> buf_Phi    = get_scratch_phi(B * K_dim * sizeof(uint16_t));
        id<MTLBuffer> buf_X_silu = has_base ? get_scratch_silu(B * D_in * sizeof(uint16_t)) : nil;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        uint uB = (uint)B, uDin = (uint)D_in, uDout = (uint)D_out;
        uint uBase = (uint)has_base, uBias = (uint)has_bias;

        if (has_base || has_bias) {
            id<MTLComputeCommandEncoder> enc_prep = [cmd computeCommandEncoder];
            [enc_prep setComputePipelineState:g_pipe_prep_fp16];
            [enc_prep setBuffer:buf_X offset:0 atIndex:0];
            [enc_prep setBuffer:(buf_bias ? buf_bias : buf_X) offset:0 atIndex:1];
            [enc_prep setBuffer:(buf_X_silu ? buf_X_silu : buf_X) offset:0 atIndex:2];
            [enc_prep setBuffer:buf_Y offset:0 atIndex:3];
            [enc_prep setBytes:&uB length:sizeof(uint) atIndex:4];
            [enc_prep setBytes:&uDin length:sizeof(uint) atIndex:5];
            [enc_prep setBytes:&uDout length:sizeof(uint) atIndex:6];
            [enc_prep setBytes:&uBase length:sizeof(uint) atIndex:7];
            [enc_prep setBytes:&uBias length:sizeof(uint) atIndex:8];
            uint max_dim = (D_in > D_out) ? D_in : D_out;
            [enc_prep dispatchThreads:MTLSizeMake(max_dim, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc_prep endEncoding];
        }

        if (has_base) {
            MPSMatrixDescriptor* desc_A_base = get_cached_desc(B, D_in, D_in * sizeof(uint16_t), MPSDataTypeFloat16);
            MPSMatrixDescriptor* desc_B_base = get_cached_desc(D_out, D_in, D_in * sizeof(uint16_t), MPSDataTypeFloat16);
            MPSMatrixDescriptor* desc_C_base = get_cached_desc(B, D_out, D_out * sizeof(uint16_t), MPSDataTypeFloat16);
            MPSMatrix* mat_A_base = [[[MPSMatrix alloc] initWithBuffer:buf_X_silu descriptor:desc_A_base] autorelease];
            MPSMatrix* mat_B_base = [[[MPSMatrix alloc] initWithBuffer:buf_W_base descriptor:desc_B_base] autorelease];
            MPSMatrix* mat_C_base = [[[MPSMatrix alloc] initWithBuffer:buf_Y descriptor:desc_C_base] autorelease];
            float beta_base = has_bias ? 1.0f : 0.0f;
            MPSMatrixMultiplication* matmul_base = get_cached_matmul(g_device, B, D_out, D_in, 1.0f, beta_base);
            [matmul_base encodeToCommandBuffer:cmd leftMatrix:mat_A_base rightMatrix:mat_B_base resultMatrix:mat_C_base];
        }

        id<MTLComputeCommandEncoder> enc_cheby = [cmd computeCommandEncoder];
        [enc_cheby setComputePipelineState:g_pipe_cheby_basis_fp16];
        [enc_cheby setBuffer:buf_X offset:0 atIndex:0];
        [enc_cheby setBuffer:buf_Phi offset:0 atIndex:1];
        uint uK = (uint)K;
        [enc_cheby setBytes:&uB length:sizeof(uint) atIndex:2];
        [enc_cheby setBytes:&uDin length:sizeof(uint) atIndex:3];
        [enc_cheby setBytes:&uK length:sizeof(uint) atIndex:4];
        [enc_cheby dispatchThreads:MTLSizeMake(D_in, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc_cheby endEncoding];

        MPSMatrixDescriptor* desc_A_cheby = get_cached_desc(B, K_dim, K_dim * sizeof(uint16_t), MPSDataTypeFloat16);
        MPSMatrixDescriptor* desc_B_cheby = get_cached_desc(D_out, K_dim, K_dim * sizeof(uint16_t), MPSDataTypeFloat16);
        MPSMatrixDescriptor* desc_C_cheby = get_cached_desc(B, D_out, D_out * sizeof(uint16_t), MPSDataTypeFloat16);
        MPSMatrix* mat_A_cheby = [[[MPSMatrix alloc] initWithBuffer:buf_Phi descriptor:desc_A_cheby] autorelease];
        MPSMatrix* mat_B_cheby = [[[MPSMatrix alloc] initWithBuffer:buf_W_cheby descriptor:desc_B_cheby] autorelease];
        MPSMatrix* mat_C_cheby = [[[MPSMatrix alloc] initWithBuffer:buf_Y descriptor:desc_C_cheby] autorelease];
        float beta_spline = (has_base || has_bias) ? 1.0f : 0.0f;
        MPSMatrixMultiplication* matmul_cheby = get_cached_matmul(g_device, B, D_out, K_dim, 1.0f, beta_spline);
        [matmul_cheby encodeToCommandBuffer:cmd leftMatrix:mat_A_cheby rightMatrix:mat_B_cheby resultMatrix:mat_C_cheby];

        commit_and_sync(cmd);
        return 0;
    }
}

int metal_kan_fastkan_forward_fp16(
    const uint16_t* X,
    const uint16_t* W_rbf,
    const uint16_t* W_base,
    const uint16_t* grid,
    const uint16_t* bias,
    uint16_t*       Y,
    int B,
    int D_in,
    int D_out,
    int num_centers,
    float inv_denominator,
    int has_base,
    int has_bias
) {
    @autoreleasepool {
        int K_dim = D_in * num_centers;
        id<MTLBuffer> buf_X      = make_no_copy_buffer((void*)X, B * D_in * sizeof(uint16_t));
        id<MTLBuffer> buf_W_rbf  = make_no_copy_buffer((void*)W_rbf, D_out * K_dim * sizeof(uint16_t));
        id<MTLBuffer> buf_W_base = has_base ? make_no_copy_buffer((void*)W_base, D_out * D_in * sizeof(uint16_t)) : nil;
        id<MTLBuffer> buf_grid   = make_no_copy_buffer((void*)grid, num_centers * sizeof(uint16_t));
        id<MTLBuffer> buf_bias   = has_bias ? make_no_copy_buffer((void*)bias, D_out * sizeof(uint16_t)) : nil;
        id<MTLBuffer> buf_Y      = make_no_copy_buffer((void*)Y, B * D_out * sizeof(uint16_t));

        id<MTLBuffer> buf_Phi    = get_scratch_phi(B * K_dim * sizeof(uint16_t));
        id<MTLBuffer> buf_X_silu = has_base ? get_scratch_silu(B * D_in * sizeof(uint16_t)) : nil;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        uint uB = (uint)B, uDin = (uint)D_in, uDout = (uint)D_out;
        uint uBase = (uint)has_base, uBias = (uint)has_bias;

        if (has_base || has_bias) {
            id<MTLComputeCommandEncoder> enc_prep = [cmd computeCommandEncoder];
            [enc_prep setComputePipelineState:g_pipe_prep_fp16];
            [enc_prep setBuffer:buf_X offset:0 atIndex:0];
            [enc_prep setBuffer:(buf_bias ? buf_bias : buf_X) offset:0 atIndex:1];
            [enc_prep setBuffer:(buf_X_silu ? buf_X_silu : buf_X) offset:0 atIndex:2];
            [enc_prep setBuffer:buf_Y offset:0 atIndex:3];
            [enc_prep setBytes:&uB length:sizeof(uint) atIndex:4];
            [enc_prep setBytes:&uDin length:sizeof(uint) atIndex:5];
            [enc_prep setBytes:&uDout length:sizeof(uint) atIndex:6];
            [enc_prep setBytes:&uBase length:sizeof(uint) atIndex:7];
            [enc_prep setBytes:&uBias length:sizeof(uint) atIndex:8];
            uint max_dim = (D_in > D_out) ? D_in : D_out;
            [enc_prep dispatchThreads:MTLSizeMake(max_dim, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc_prep endEncoding];
        }

        if (has_base) {
            MPSMatrixDescriptor* desc_A_base = get_cached_desc(B, D_in, D_in * sizeof(uint16_t), MPSDataTypeFloat16);
            MPSMatrixDescriptor* desc_B_base = get_cached_desc(D_out, D_in, D_in * sizeof(uint16_t), MPSDataTypeFloat16);
            MPSMatrixDescriptor* desc_C_base = get_cached_desc(B, D_out, D_out * sizeof(uint16_t), MPSDataTypeFloat16);
            MPSMatrix* mat_A_base = [[[MPSMatrix alloc] initWithBuffer:buf_X_silu descriptor:desc_A_base] autorelease];
            MPSMatrix* mat_B_base = [[[MPSMatrix alloc] initWithBuffer:buf_W_base descriptor:desc_B_base] autorelease];
            MPSMatrix* mat_C_base = [[[MPSMatrix alloc] initWithBuffer:buf_Y descriptor:desc_C_base] autorelease];
            float beta_base = has_bias ? 1.0f : 0.0f;
            MPSMatrixMultiplication* matmul_base = get_cached_matmul(g_device, B, D_out, D_in, 1.0f, beta_base);
            [matmul_base encodeToCommandBuffer:cmd leftMatrix:mat_A_base rightMatrix:mat_B_base resultMatrix:mat_C_base];
        }

        id<MTLComputeCommandEncoder> enc_rbf = [cmd computeCommandEncoder];
        [enc_rbf setComputePipelineState:g_pipe_rbf_basis_fp16];
        [enc_rbf setBuffer:buf_X offset:0 atIndex:0];
        [enc_rbf setBuffer:buf_grid offset:0 atIndex:1];
        [enc_rbf setBuffer:buf_Phi offset:0 atIndex:2];
        uint uK = (uint)num_centers;
        float u_inv_den = inv_denominator;
        [enc_rbf setBytes:&uB length:sizeof(uint) atIndex:3];
        [enc_rbf setBytes:&uDin length:sizeof(uint) atIndex:4];
        [enc_rbf setBytes:&uK length:sizeof(uint) atIndex:5];
        [enc_rbf setBytes:&u_inv_den length:sizeof(float) atIndex:6];
        [enc_rbf dispatchThreads:MTLSizeMake(D_in, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc_rbf endEncoding];

        MPSMatrixDescriptor* desc_A_rbf = get_cached_desc(B, K_dim, K_dim * sizeof(uint16_t), MPSDataTypeFloat16);
        MPSMatrixDescriptor* desc_B_rbf = get_cached_desc(D_out, K_dim, K_dim * sizeof(uint16_t), MPSDataTypeFloat16);
        MPSMatrixDescriptor* desc_C_rbf = get_cached_desc(B, D_out, D_out * sizeof(uint16_t), MPSDataTypeFloat16);
        MPSMatrix* mat_A_rbf = [[[MPSMatrix alloc] initWithBuffer:buf_Phi descriptor:desc_A_rbf] autorelease];
        MPSMatrix* mat_B_rbf = [[[MPSMatrix alloc] initWithBuffer:buf_W_rbf descriptor:desc_B_rbf] autorelease];
        MPSMatrix* mat_C_rbf = [[[MPSMatrix alloc] initWithBuffer:buf_Y descriptor:desc_C_rbf] autorelease];
        float beta_rbf = (has_base || has_bias) ? 1.0f : 0.0f;
        MPSMatrixMultiplication* matmul_rbf = get_cached_matmul(g_device, B, D_out, K_dim, 1.0f, beta_rbf);
        [matmul_rbf encodeToCommandBuffer:cmd leftMatrix:mat_A_rbf rightMatrix:mat_B_rbf resultMatrix:mat_C_rbf];

        commit_and_sync(cmd);
        return 0;
    }
}

int metal_kan_bspline_forward_fp16(
    const uint16_t* X,
    const uint16_t* W_spline,
    const uint16_t* W_base,
    const float*    grid,
    const uint16_t* bias,
    uint16_t*       Y,
    int B,
    int D_in,
    int D_out,
    int grid_size,
    int spline_order,
    int has_base,
    int has_bias
) {
    @autoreleasepool {
        uint num_bases = grid_size + 3;
        int K_dim = D_in * num_bases;

        id<MTLBuffer> buf_X       = make_no_copy_buffer((void*)X, B * D_in * sizeof(uint16_t));
        id<MTLBuffer> buf_W_spl   = make_no_copy_buffer((void*)W_spline, D_out * K_dim * sizeof(uint16_t));
        id<MTLBuffer> buf_W_base  = has_base ? make_no_copy_buffer((void*)W_base, D_out * D_in * sizeof(uint16_t)) : nil;
        id<MTLBuffer> buf_bias    = has_bias ? make_no_copy_buffer((void*)bias, D_out * sizeof(uint16_t)) : nil;
        id<MTLBuffer> buf_Y       = make_no_copy_buffer((void*)Y, B * D_out * sizeof(uint16_t));

        id<MTLBuffer> buf_Phi    = get_scratch_phi(B * K_dim * sizeof(uint16_t));
        id<MTLBuffer> buf_X_silu = has_base ? get_scratch_silu(B * D_in * sizeof(uint16_t)) : nil;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        uint uB = (uint)B, uDin = (uint)D_in, uDout = (uint)D_out;
        uint uBase = (uint)has_base, uBias = (uint)has_bias;

        if (has_base || has_bias) {
            id<MTLComputeCommandEncoder> enc_prep = [cmd computeCommandEncoder];
            [enc_prep setComputePipelineState:g_pipe_prep_fp16];
            [enc_prep setBuffer:buf_X offset:0 atIndex:0];
            [enc_prep setBuffer:(buf_bias ? buf_bias : buf_X) offset:0 atIndex:1];
            [enc_prep setBuffer:(buf_X_silu ? buf_X_silu : buf_X) offset:0 atIndex:2];
            [enc_prep setBuffer:buf_Y offset:0 atIndex:3];
            [enc_prep setBytes:&uB length:sizeof(uint) atIndex:4];
            [enc_prep setBytes:&uDin length:sizeof(uint) atIndex:5];
            [enc_prep setBytes:&uDout length:sizeof(uint) atIndex:6];
            [enc_prep setBytes:&uBase length:sizeof(uint) atIndex:7];
            [enc_prep setBytes:&uBias length:sizeof(uint) atIndex:8];
            uint max_dim = (D_in > D_out) ? D_in : D_out;
            [enc_prep dispatchThreads:MTLSizeMake(max_dim, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc_prep endEncoding];
        }

        if (has_base) {
            MPSMatrixDescriptor* desc_A_base = get_cached_desc(B, D_in, D_in * sizeof(uint16_t), MPSDataTypeFloat16);
            MPSMatrixDescriptor* desc_B_base = get_cached_desc(D_out, D_in, D_in * sizeof(uint16_t), MPSDataTypeFloat16);
            MPSMatrixDescriptor* desc_C_base = get_cached_desc(B, D_out, D_out * sizeof(uint16_t), MPSDataTypeFloat16);
            MPSMatrix* mat_A_base = [[[MPSMatrix alloc] initWithBuffer:buf_X_silu descriptor:desc_A_base] autorelease];
            MPSMatrix* mat_B_base = [[[MPSMatrix alloc] initWithBuffer:buf_W_base descriptor:desc_B_base] autorelease];
            MPSMatrix* mat_C_base = [[[MPSMatrix alloc] initWithBuffer:buf_Y descriptor:desc_C_base] autorelease];
            float beta_base = has_bias ? 1.0f : 0.0f;
            MPSMatrixMultiplication* matmul_base = get_cached_matmul(g_device, B, D_out, D_in, 1.0f, beta_base);
            [matmul_base encodeToCommandBuffer:cmd leftMatrix:mat_A_base rightMatrix:mat_B_base resultMatrix:mat_C_base];
        }

        id<MTLComputeCommandEncoder> enc_spl = [cmd computeCommandEncoder];
        [enc_spl setComputePipelineState:g_pipe_bspline_basis_fp16];
        [enc_spl setBuffer:buf_X offset:0 atIndex:0];
        [enc_spl setBuffer:buf_Phi offset:0 atIndex:1];
        uint u_gsize = (uint)grid_size;
        float grid_min = grid[0];
        float inv_h = grid[1];
        [enc_spl setBytes:&uB length:sizeof(uint) atIndex:2];
        [enc_spl setBytes:&uDin length:sizeof(uint) atIndex:3];
        [enc_spl setBytes:&u_gsize length:sizeof(uint) atIndex:4];
        [enc_spl setBytes:&grid_min length:sizeof(float) atIndex:5];
        [enc_spl setBytes:&inv_h length:sizeof(float) atIndex:6];
        [enc_spl dispatchThreads:MTLSizeMake(D_in, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc_spl endEncoding];

        MPSMatrixDescriptor* desc_A_spl = get_cached_desc(B, K_dim, K_dim * sizeof(uint16_t), MPSDataTypeFloat16);
        MPSMatrixDescriptor* desc_B_spl = get_cached_desc(D_out, K_dim, K_dim * sizeof(uint16_t), MPSDataTypeFloat16);
        MPSMatrixDescriptor* desc_C_spl = get_cached_desc(B, D_out, D_out * sizeof(uint16_t), MPSDataTypeFloat16);
        MPSMatrix* mat_A_spl = [[[MPSMatrix alloc] initWithBuffer:buf_Phi descriptor:desc_A_spl] autorelease];
        MPSMatrix* mat_B_spl = [[[MPSMatrix alloc] initWithBuffer:buf_W_spl descriptor:desc_B_spl] autorelease];
        MPSMatrix* mat_C_spl = [[[MPSMatrix alloc] initWithBuffer:buf_Y descriptor:desc_C_spl] autorelease];
        float beta_spline = (has_base || has_bias) ? 1.0f : 0.0f;
        MPSMatrixMultiplication* matmul_spl = get_cached_matmul(g_device, B, D_out, K_dim, 1.0f, beta_spline);
        [matmul_spl encodeToCommandBuffer:cmd leftMatrix:mat_A_spl rightMatrix:mat_B_spl resultMatrix:mat_C_spl];

        commit_and_sync(cmd);
        return 0;
    }
}

int metal_kan_lowrank_forward_fp16(
    const uint16_t* X,
    const uint16_t* W_down,
    const uint16_t* W_up,
    const uint16_t* W_base,
    const uint16_t* grid,
    const uint16_t* bias,
    uint16_t*       Y,
    int B,
    int D_in,
    int D_out,
    int rank,
    int num_centers,
    float inv_denominator,
    int has_base,
    int has_bias
) {
    @autoreleasepool {
        int K_dim = D_in * num_centers;
        id<MTLBuffer> buf_X      = make_no_copy_buffer((void*)X, B * D_in * sizeof(uint16_t));
        id<MTLBuffer> buf_W_down = make_no_copy_buffer((void*)W_down, rank * K_dim * sizeof(uint16_t));
        id<MTLBuffer> buf_W_up   = make_no_copy_buffer((void*)W_up, D_out * rank * sizeof(uint16_t));
        id<MTLBuffer> buf_W_base = has_base ? make_no_copy_buffer((void*)W_base, D_out * D_in * sizeof(uint16_t)) : nil;
        id<MTLBuffer> buf_grid   = make_no_copy_buffer((void*)grid, num_centers * sizeof(uint16_t));
        id<MTLBuffer> buf_bias   = has_bias ? make_no_copy_buffer((void*)bias, D_out * sizeof(uint16_t)) : nil;
        id<MTLBuffer> buf_Y      = make_no_copy_buffer((void*)Y, B * D_out * sizeof(uint16_t));

        id<MTLBuffer> buf_Phi    = get_scratch_phi(B * K_dim * sizeof(uint16_t));
        id<MTLBuffer> buf_Z      = get_scratch_bottleneck(B * rank * sizeof(uint16_t));
        id<MTLBuffer> buf_X_silu = has_base ? get_scratch_silu(B * D_in * sizeof(uint16_t)) : nil;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];

        if (has_base || has_bias) {
            id<MTLComputeCommandEncoder> enc_prep = [cmd computeCommandEncoder];
            [enc_prep setComputePipelineState:g_pipe_prep_fp16];
            [enc_prep setBuffer:buf_X offset:0 atIndex:0];
            [enc_prep setBuffer:(buf_bias ? buf_bias : buf_X) offset:0 atIndex:1];
            [enc_prep setBuffer:(buf_X_silu ? buf_X_silu : buf_X) offset:0 atIndex:2];
            [enc_prep setBuffer:buf_Y offset:0 atIndex:3];
            uint uB = (uint)B, uDin = (uint)D_in, uDout = (uint)D_out;
            uint uBase = (uint)has_base, uBias = (uint)has_bias;
            [enc_prep setBytes:&uB length:sizeof(uint) atIndex:4];
            [enc_prep setBytes:&uDin length:sizeof(uint) atIndex:5];
            [enc_prep setBytes:&uDout length:sizeof(uint) atIndex:6];
            [enc_prep setBytes:&uBase length:sizeof(uint) atIndex:7];
            [enc_prep setBytes:&uBias length:sizeof(uint) atIndex:8];
            uint max_dim = (D_in > D_out) ? D_in : D_out;
            [enc_prep dispatchThreads:MTLSizeMake(max_dim, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc_prep endEncoding];
        }

        id<MTLComputeCommandEncoder> enc_rbf = [cmd computeCommandEncoder];
        [enc_rbf setComputePipelineState:g_pipe_rbf_basis_fp16];
        [enc_rbf setBuffer:buf_X offset:0 atIndex:0];
        [enc_rbf setBuffer:buf_grid offset:0 atIndex:1];
        [enc_rbf setBuffer:buf_Phi offset:0 atIndex:2];
        uint uB = (uint)B, uDin = (uint)D_in, uK = (uint)num_centers;
        float u_inv_den = inv_denominator;
        [enc_rbf setBytes:&uB length:sizeof(uint) atIndex:3];
        [enc_rbf setBytes:&uDin length:sizeof(uint) atIndex:4];
        [enc_rbf setBytes:&uK length:sizeof(uint) atIndex:5];
        [enc_rbf setBytes:&u_inv_den length:sizeof(float) atIndex:6];
        [enc_rbf dispatchThreads:MTLSizeMake(D_in, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc_rbf endEncoding];

        MPSMatrixDescriptor* desc_Phi = get_cached_desc(B, K_dim, K_dim * sizeof(uint16_t), MPSDataTypeFloat16);
        MPSMatrixDescriptor* desc_Wdown = get_cached_desc(rank, K_dim, K_dim * sizeof(uint16_t), MPSDataTypeFloat16);
        MPSMatrixDescriptor* desc_Z = get_cached_desc(B, rank, rank * sizeof(uint16_t), MPSDataTypeFloat16);
        MPSMatrix* mat_Phi = [[[MPSMatrix alloc] initWithBuffer:buf_Phi descriptor:desc_Phi] autorelease];
        MPSMatrix* mat_Wdown = [[[MPSMatrix alloc] initWithBuffer:buf_W_down descriptor:desc_Wdown] autorelease];
        MPSMatrix* mat_Z = [[[MPSMatrix alloc] initWithBuffer:buf_Z descriptor:desc_Z] autorelease];
        MPSMatrixMultiplication* matmul_down = get_cached_matmul(g_device, B, rank, K_dim, 1.0f, 0.0f);
        [matmul_down encodeToCommandBuffer:cmd leftMatrix:mat_Phi rightMatrix:mat_Wdown resultMatrix:mat_Z];

        MPSMatrixDescriptor* desc_Wup = get_cached_desc(D_out, rank, rank * sizeof(uint16_t), MPSDataTypeFloat16);
        MPSMatrixDescriptor* desc_Y = get_cached_desc(B, D_out, D_out * sizeof(uint16_t), MPSDataTypeFloat16);
        MPSMatrix* mat_Wup = [[[MPSMatrix alloc] initWithBuffer:buf_W_up descriptor:desc_Wup] autorelease];
        MPSMatrix* mat_Y = [[[MPSMatrix alloc] initWithBuffer:buf_Y descriptor:desc_Y] autorelease];
        float beta_up = has_bias ? 1.0f : 0.0f;
        MPSMatrixMultiplication* matmul_up = get_cached_matmul(g_device, B, D_out, rank, 1.0f, beta_up);
        [matmul_up encodeToCommandBuffer:cmd leftMatrix:mat_Z rightMatrix:mat_Wup resultMatrix:mat_Y];

        if (has_base) {
            MPSMatrixDescriptor* desc_A_base = get_cached_desc(B, D_in, D_in * sizeof(uint16_t), MPSDataTypeFloat16);
            MPSMatrixDescriptor* desc_B_base = get_cached_desc(D_out, D_in, D_in * sizeof(uint16_t), MPSDataTypeFloat16);
            MPSMatrix* mat_A_base = [[[MPSMatrix alloc] initWithBuffer:buf_X_silu descriptor:desc_A_base] autorelease];
            MPSMatrix* mat_B_base = [[[MPSMatrix alloc] initWithBuffer:buf_W_base descriptor:desc_B_base] autorelease];
            MPSMatrixMultiplication* matmul_base = get_cached_matmul(g_device, B, D_out, D_in, 1.0f, 1.0f);
            [matmul_base encodeToCommandBuffer:cmd leftMatrix:mat_A_base rightMatrix:mat_B_base resultMatrix:mat_Y];
        }

        commit_and_sync(cmd);
        return 0;
    }
}

int metal_kan_mult_forward_fp16(
    const uint16_t* X,
    const uint16_t* W_rbf,
    const uint16_t* W_base,
    const uint16_t* grid,
    const uint16_t* bias,
    uint16_t*       Y,
    int B,
    int D_in,
    int D_out,
    int num_add,
    int num_mult,
    int num_centers,
    float inv_denominator,
    int has_base,
    int has_bias
) {
    @autoreleasepool {
        int internal_out = num_add + 2 * num_mult;
        int K_dim = D_in * num_centers;

        id<MTLBuffer> buf_X      = make_no_copy_buffer((void*)X, B * D_in * sizeof(uint16_t));
        id<MTLBuffer> buf_W_rbf  = make_no_copy_buffer((void*)W_rbf, internal_out * K_dim * sizeof(uint16_t));
        id<MTLBuffer> buf_W_base = has_base ? make_no_copy_buffer((void*)W_base, internal_out * D_in * sizeof(uint16_t)) : nil;
        id<MTLBuffer> buf_grid   = make_no_copy_buffer((void*)grid, num_centers * sizeof(uint16_t));
        id<MTLBuffer> buf_bias   = has_bias ? make_no_copy_buffer((void*)bias, internal_out * sizeof(uint16_t)) : nil;
        id<MTLBuffer> buf_Y      = make_no_copy_buffer((void*)Y, B * D_out * sizeof(uint16_t));

        id<MTLBuffer> buf_Phi      = get_scratch_phi(B * K_dim * sizeof(uint16_t));
        id<MTLBuffer> buf_internal = get_scratch_internal(B * internal_out * sizeof(uint16_t));
        id<MTLBuffer> buf_X_silu   = has_base ? get_scratch_silu(B * D_in * sizeof(uint16_t)) : nil;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        uint uB = (uint)B, uDin = (uint)D_in, uIntOut = (uint)internal_out;
        uint uBase = (uint)has_base, uBias = (uint)has_bias;

        if (has_base || has_bias) {
            id<MTLComputeCommandEncoder> enc_prep = [cmd computeCommandEncoder];
            [enc_prep setComputePipelineState:g_pipe_prep_fp16];
            [enc_prep setBuffer:buf_X offset:0 atIndex:0];
            [enc_prep setBuffer:(buf_bias ? buf_bias : buf_X) offset:0 atIndex:1];
            [enc_prep setBuffer:(buf_X_silu ? buf_X_silu : buf_X) offset:0 atIndex:2];
            [enc_prep setBuffer:buf_internal offset:0 atIndex:3];
            [enc_prep setBytes:&uB length:sizeof(uint) atIndex:4];
            [enc_prep setBytes:&uDin length:sizeof(uint) atIndex:5];
            [enc_prep setBytes:&uIntOut length:sizeof(uint) atIndex:6];
            [enc_prep setBytes:&uBase length:sizeof(uint) atIndex:7];
            [enc_prep setBytes:&uBias length:sizeof(uint) atIndex:8];
            uint max_dim = (D_in > internal_out) ? D_in : internal_out;
            [enc_prep dispatchThreads:MTLSizeMake(max_dim, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc_prep endEncoding];
        }

        if (has_base) {
            MPSMatrixDescriptor* desc_A_base = get_cached_desc(B, D_in, D_in * sizeof(uint16_t), MPSDataTypeFloat16);
            MPSMatrixDescriptor* desc_B_base = get_cached_desc(internal_out, D_in, D_in * sizeof(uint16_t), MPSDataTypeFloat16);
            MPSMatrixDescriptor* desc_C_base = get_cached_desc(B, internal_out, internal_out * sizeof(uint16_t), MPSDataTypeFloat16);
            MPSMatrix* mat_A_base = [[[MPSMatrix alloc] initWithBuffer:buf_X_silu descriptor:desc_A_base] autorelease];
            MPSMatrix* mat_B_base = [[[MPSMatrix alloc] initWithBuffer:buf_W_base descriptor:desc_B_base] autorelease];
            MPSMatrix* mat_C_base = [[[MPSMatrix alloc] initWithBuffer:buf_internal descriptor:desc_C_base] autorelease];
            float beta_base = has_bias ? 1.0f : 0.0f;
            MPSMatrixMultiplication* matmul_base = get_cached_matmul(g_device, B, internal_out, D_in, 1.0f, beta_base);
            [matmul_base encodeToCommandBuffer:cmd leftMatrix:mat_A_base rightMatrix:mat_B_base resultMatrix:mat_C_base];
        }

        id<MTLComputeCommandEncoder> enc_rbf = [cmd computeCommandEncoder];
        [enc_rbf setComputePipelineState:g_pipe_rbf_basis_fp16];
        [enc_rbf setBuffer:buf_X offset:0 atIndex:0];
        [enc_rbf setBuffer:buf_grid offset:0 atIndex:1];
        [enc_rbf setBuffer:buf_Phi offset:0 atIndex:2];
        uint uK = (uint)num_centers;
        float u_inv_den = inv_denominator;
        [enc_rbf setBytes:&uB length:sizeof(uint) atIndex:3];
        [enc_rbf setBytes:&uDin length:sizeof(uint) atIndex:4];
        [enc_rbf setBytes:&uK length:sizeof(uint) atIndex:5];
        [enc_rbf setBytes:&u_inv_den length:sizeof(float) atIndex:6];
        [enc_rbf dispatchThreads:MTLSizeMake(D_in, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc_rbf endEncoding];

        MPSMatrixDescriptor* desc_A_rbf = get_cached_desc(B, K_dim, K_dim * sizeof(uint16_t), MPSDataTypeFloat16);
        MPSMatrixDescriptor* desc_B_rbf = get_cached_desc(internal_out, K_dim, K_dim * sizeof(uint16_t), MPSDataTypeFloat16);
        MPSMatrixDescriptor* desc_C_rbf = get_cached_desc(B, internal_out, internal_out * sizeof(uint16_t), MPSDataTypeFloat16);
        MPSMatrix* mat_A_rbf = [[[MPSMatrix alloc] initWithBuffer:buf_Phi descriptor:desc_A_rbf] autorelease];
        MPSMatrix* mat_B_rbf = [[[MPSMatrix alloc] initWithBuffer:buf_W_rbf descriptor:desc_B_rbf] autorelease];
        MPSMatrix* mat_C_rbf = [[[MPSMatrix alloc] initWithBuffer:buf_internal descriptor:desc_C_rbf] autorelease];
        float beta_rbf = (has_base || has_bias) ? 1.0f : 0.0f;
        MPSMatrixMultiplication* matmul_rbf = get_cached_matmul(g_device, B, internal_out, K_dim, 1.0f, beta_rbf);
        [matmul_rbf encodeToCommandBuffer:cmd leftMatrix:mat_A_rbf rightMatrix:mat_B_rbf resultMatrix:mat_C_rbf];

        if (num_mult > 0) {
            id<MTLComputeCommandEncoder> enc_comb = [cmd computeCommandEncoder];
            [enc_comb setComputePipelineState:g_pipe_combine_mult_fp16];
            [enc_comb setBuffer:buf_internal offset:0 atIndex:0];
            [enc_comb setBuffer:buf_Y offset:0 atIndex:1];
            uint uB = (uint)B;
            uint uAdd = (uint)num_add;
            uint uMult = (uint)num_mult;
            [enc_comb setBytes:&uB length:sizeof(uint) atIndex:2];
            [enc_comb setBytes:&uAdd length:sizeof(uint) atIndex:3];
            [enc_comb setBytes:&uMult length:sizeof(uint) atIndex:4];
            [enc_comb dispatchThreads:MTLSizeMake(D_out, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc_comb endEncoding];
        }

        commit_and_sync(cmd);
        return 0;
    }
}

int metal_kan_relu_forward_fp16(
    const uint16_t* X,
    const uint16_t* W_relu,
    const uint16_t* W_base,
    const uint16_t* grid,
    const uint16_t* bias,
    uint16_t*       Y,
    int B,
    int D_in,
    int D_out,
    int num_grids,
    float inv_h,
    int has_base,
    int has_bias
) {
    @autoreleasepool {
        int K_dim = D_in * num_grids;
        id<MTLBuffer> buf_X      = make_no_copy_buffer((void*)X, B * D_in * sizeof(uint16_t));
        id<MTLBuffer> buf_W_relu = make_no_copy_buffer((void*)W_relu, D_out * K_dim * sizeof(uint16_t));
        id<MTLBuffer> buf_W_base = has_base ? make_no_copy_buffer((void*)W_base, D_out * D_in * sizeof(uint16_t)) : nil;
        id<MTLBuffer> buf_grid   = make_no_copy_buffer((void*)grid, num_grids * sizeof(uint16_t));
        id<MTLBuffer> buf_bias   = has_bias ? make_no_copy_buffer((void*)bias, D_out * sizeof(uint16_t)) : nil;
        id<MTLBuffer> buf_Y      = make_no_copy_buffer((void*)Y, B * D_out * sizeof(uint16_t));

        id<MTLBuffer> buf_Phi    = get_scratch_phi(B * K_dim * sizeof(uint16_t));
        id<MTLBuffer> buf_X_silu = has_base ? get_scratch_silu(B * D_in * sizeof(uint16_t)) : nil;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        uint uB = (uint)B, uDin = (uint)D_in, uDout = (uint)D_out;
        uint uBase = (uint)has_base, uBias = (uint)has_bias;

        if (has_base || has_bias) {
            id<MTLComputeCommandEncoder> enc_prep = [cmd computeCommandEncoder];
            [enc_prep setComputePipelineState:g_pipe_prep_fp16];
            [enc_prep setBuffer:buf_X offset:0 atIndex:0];
            [enc_prep setBuffer:(buf_bias ? buf_bias : buf_X) offset:0 atIndex:1];
            [enc_prep setBuffer:(buf_X_silu ? buf_X_silu : buf_X) offset:0 atIndex:2];
            [enc_prep setBuffer:buf_Y offset:0 atIndex:3];
            [enc_prep setBytes:&uB length:sizeof(uint) atIndex:4];
            [enc_prep setBytes:&uDin length:sizeof(uint) atIndex:5];
            [enc_prep setBytes:&uDout length:sizeof(uint) atIndex:6];
            [enc_prep setBytes:&uBase length:sizeof(uint) atIndex:7];
            [enc_prep setBytes:&uBias length:sizeof(uint) atIndex:8];
            uint max_dim = (D_in > D_out) ? D_in : D_out;
            [enc_prep dispatchThreads:MTLSizeMake(max_dim, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc_prep endEncoding];
        }

        if (has_base) {
            MPSMatrixDescriptor* desc_A_base = get_cached_desc(B, D_in, D_in * sizeof(uint16_t), MPSDataTypeFloat16);
            MPSMatrixDescriptor* desc_B_base = get_cached_desc(D_out, D_in, D_in * sizeof(uint16_t), MPSDataTypeFloat16);
            MPSMatrixDescriptor* desc_C_base = get_cached_desc(B, D_out, D_out * sizeof(uint16_t), MPSDataTypeFloat16);
            MPSMatrix* mat_A_base = [[[MPSMatrix alloc] initWithBuffer:buf_X_silu descriptor:desc_A_base] autorelease];
            MPSMatrix* mat_B_base = [[[MPSMatrix alloc] initWithBuffer:buf_W_base descriptor:desc_B_base] autorelease];
            MPSMatrix* mat_C_base = [[[MPSMatrix alloc] initWithBuffer:buf_Y descriptor:desc_C_base] autorelease];
            float beta_base = has_bias ? 1.0f : 0.0f;
            MPSMatrixMultiplication* matmul_base = get_cached_matmul(g_device, B, D_out, D_in, 1.0f, beta_base);
            [matmul_base encodeToCommandBuffer:cmd leftMatrix:mat_A_base rightMatrix:mat_B_base resultMatrix:mat_C_base];
        }

        id<MTLComputeCommandEncoder> enc_relu = [cmd computeCommandEncoder];
        [enc_relu setComputePipelineState:g_pipe_relu_basis_fp16];
        [enc_relu setBuffer:buf_X offset:0 atIndex:0];
        [enc_relu setBuffer:buf_grid offset:0 atIndex:1];
        [enc_relu setBuffer:buf_Phi offset:0 atIndex:2];
        uint uG = (uint)num_grids;
        float u_inv_h = inv_h;
        [enc_relu setBytes:&uB length:sizeof(uint) atIndex:3];
        [enc_relu setBytes:&uDin length:sizeof(uint) atIndex:4];
        [enc_relu setBytes:&uG length:sizeof(uint) atIndex:5];
        [enc_relu setBytes:&u_inv_h length:sizeof(float) atIndex:6];
        [enc_relu dispatchThreads:MTLSizeMake(D_in, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc_relu endEncoding];

        MPSMatrixDescriptor* desc_A_relu = get_cached_desc(B, K_dim, K_dim * sizeof(uint16_t), MPSDataTypeFloat16);
        MPSMatrixDescriptor* desc_B_relu = get_cached_desc(D_out, K_dim, K_dim * sizeof(uint16_t), MPSDataTypeFloat16);
        MPSMatrixDescriptor* desc_C_relu = get_cached_desc(B, D_out, D_out * sizeof(uint16_t), MPSDataTypeFloat16);
        MPSMatrix* mat_A_relu = [[[MPSMatrix alloc] initWithBuffer:buf_Phi descriptor:desc_A_relu] autorelease];
        MPSMatrix* mat_B_relu = [[[MPSMatrix alloc] initWithBuffer:buf_W_relu descriptor:desc_B_relu] autorelease];
        MPSMatrix* mat_C_relu = [[[MPSMatrix alloc] initWithBuffer:buf_Y descriptor:desc_C_relu] autorelease];
        float beta_spline = (has_base || has_bias) ? 1.0f : 0.0f;
        MPSMatrixMultiplication* matmul_relu = get_cached_matmul(g_device, B, D_out, K_dim, 1.0f, beta_spline);
        [matmul_relu encodeToCommandBuffer:cmd leftMatrix:mat_A_relu rightMatrix:mat_B_relu resultMatrix:mat_C_relu];

        commit_and_sync(cmd);
        return 0;
    }
}

int metal_kan_wavkan_forward_fp16(
    const uint16_t* X,
    const uint16_t* W_wav,
    const uint16_t* W_base,
    const uint16_t* translation,
    const uint16_t* inv_scale,
    const uint16_t* bias,
    uint16_t*       Y,
    int B,
    int D_in,
    int D_out,
    int num_wavelets,
    int wavelet_type,
    int has_base,
    int has_bias
) {
    @autoreleasepool {
        int K_dim = D_in * num_wavelets;
        id<MTLBuffer> buf_X       = make_no_copy_buffer((void*)X, B * D_in * sizeof(uint16_t));
        id<MTLBuffer> buf_W_wav   = make_no_copy_buffer((void*)W_wav, D_out * K_dim * sizeof(uint16_t));
        id<MTLBuffer> buf_W_base  = has_base ? make_no_copy_buffer((void*)W_base, D_out * D_in * sizeof(uint16_t)) : nil;
        id<MTLBuffer> buf_trans   = make_no_copy_buffer((void*)translation, D_in * num_wavelets * sizeof(uint16_t));
        id<MTLBuffer> buf_scale   = make_no_copy_buffer((void*)inv_scale, D_in * num_wavelets * sizeof(uint16_t));
        id<MTLBuffer> buf_bias    = has_bias ? make_no_copy_buffer((void*)bias, D_out * sizeof(uint16_t)) : nil;
        id<MTLBuffer> buf_Y       = make_no_copy_buffer((void*)Y, B * D_out * sizeof(uint16_t));

        id<MTLBuffer> buf_Phi     = get_scratch_phi(B * K_dim * sizeof(uint16_t));
        id<MTLBuffer> buf_X_silu  = has_base ? get_scratch_silu(B * D_in * sizeof(uint16_t)) : nil;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        uint uB = (uint)B, uDin = (uint)D_in, uDout = (uint)D_out;
        uint uBase = (uint)has_base, uBias = (uint)has_bias;

        if (has_base || has_bias) {
            id<MTLComputeCommandEncoder> enc_prep = [cmd computeCommandEncoder];
            [enc_prep setComputePipelineState:g_pipe_prep_fp16];
            [enc_prep setBuffer:buf_X offset:0 atIndex:0];
            [enc_prep setBuffer:(buf_bias ? buf_bias : buf_X) offset:0 atIndex:1];
            [enc_prep setBuffer:(buf_X_silu ? buf_X_silu : buf_X) offset:0 atIndex:2];
            [enc_prep setBuffer:buf_Y offset:0 atIndex:3];
            [enc_prep setBytes:&uB length:sizeof(uint) atIndex:4];
            [enc_prep setBytes:&uDin length:sizeof(uint) atIndex:5];
            [enc_prep setBytes:&uDout length:sizeof(uint) atIndex:6];
            [enc_prep setBytes:&uBase length:sizeof(uint) atIndex:7];
            [enc_prep setBytes:&uBias length:sizeof(uint) atIndex:8];
            uint max_dim = (D_in > D_out) ? D_in : D_out;
            [enc_prep dispatchThreads:MTLSizeMake(max_dim, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc_prep endEncoding];
        }

        if (has_base) {
            MPSMatrixDescriptor* desc_A_base = get_cached_desc(B, D_in, D_in * sizeof(uint16_t), MPSDataTypeFloat16);
            MPSMatrixDescriptor* desc_B_base = get_cached_desc(D_out, D_in, D_in * sizeof(uint16_t), MPSDataTypeFloat16);
            MPSMatrixDescriptor* desc_C_base = get_cached_desc(B, D_out, D_out * sizeof(uint16_t), MPSDataTypeFloat16);
            MPSMatrix* mat_A_base = [[[MPSMatrix alloc] initWithBuffer:buf_X_silu descriptor:desc_A_base] autorelease];
            MPSMatrix* mat_B_base = [[[MPSMatrix alloc] initWithBuffer:buf_W_base descriptor:desc_B_base] autorelease];
            MPSMatrix* mat_C_base = [[[MPSMatrix alloc] initWithBuffer:buf_Y descriptor:desc_C_base] autorelease];
            float beta_base = has_bias ? 1.0f : 0.0f;
            MPSMatrixMultiplication* matmul_base = get_cached_matmul(g_device, B, D_out, D_in, 1.0f, beta_base);
            [matmul_base encodeToCommandBuffer:cmd leftMatrix:mat_A_base rightMatrix:mat_B_base resultMatrix:mat_C_base];
        }

        id<MTLComputeCommandEncoder> enc_wav = [cmd computeCommandEncoder];
        [enc_wav setComputePipelineState:g_pipe_wav_basis_fp16];
        [enc_wav setBuffer:buf_X offset:0 atIndex:0];
        [enc_wav setBuffer:buf_trans offset:0 atIndex:1];
        [enc_wav setBuffer:buf_scale offset:0 atIndex:2];
        [enc_wav setBuffer:buf_Phi offset:0 atIndex:3];
        uint uK = (uint)num_wavelets, uType = (uint)wavelet_type;
        [enc_wav setBytes:&uB length:sizeof(uint) atIndex:4];
        [enc_wav setBytes:&uDin length:sizeof(uint) atIndex:5];
        [enc_wav setBytes:&uK length:sizeof(uint) atIndex:6];
        [enc_wav setBytes:&uType length:sizeof(uint) atIndex:7];
        [enc_wav dispatchThreads:MTLSizeMake(D_in, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc_wav endEncoding];

        MPSMatrixDescriptor* desc_A_wav = get_cached_desc(B, K_dim, K_dim * sizeof(uint16_t), MPSDataTypeFloat16);
        MPSMatrixDescriptor* desc_B_wav = get_cached_desc(D_out, K_dim, K_dim * sizeof(uint16_t), MPSDataTypeFloat16);
        MPSMatrixDescriptor* desc_C_wav = get_cached_desc(B, D_out, D_out * sizeof(uint16_t), MPSDataTypeFloat16);
        MPSMatrix* mat_A_wav = [[[MPSMatrix alloc] initWithBuffer:buf_Phi descriptor:desc_A_wav] autorelease];
        MPSMatrix* mat_B_wav = [[[MPSMatrix alloc] initWithBuffer:buf_W_wav descriptor:desc_B_wav] autorelease];
        MPSMatrix* mat_C_wav = [[[MPSMatrix alloc] initWithBuffer:buf_Y descriptor:desc_C_wav] autorelease];
        float beta_spline = (has_base || has_bias) ? 1.0f : 0.0f;
        MPSMatrixMultiplication* matmul_wav = get_cached_matmul(g_device, B, D_out, K_dim, 1.0f, beta_spline);
        [matmul_wav encodeToCommandBuffer:cmd leftMatrix:mat_A_wav rightMatrix:mat_B_wav resultMatrix:mat_C_wav];

        commit_and_sync(cmd);
        return 0;
    }
}

int metal_kan_fourier_forward_fp16(
    const uint16_t* X,
    const uint16_t* W_fourier,
    const uint16_t* W_base,
    const uint16_t* bias,
    uint16_t*       Y,
    int B,
    int D_in,
    int D_out,
    int num_frequencies,
    int has_base,
    int has_bias
) {
    @autoreleasepool {
        int num_bases = 2 * num_frequencies + 1;
        int K_dim = D_in * num_bases;

        id<MTLBuffer> buf_X       = make_no_copy_buffer((void*)X, B * D_in * sizeof(uint16_t));
        id<MTLBuffer> buf_W_four  = make_no_copy_buffer((void*)W_fourier, D_out * K_dim * sizeof(uint16_t));
        id<MTLBuffer> buf_W_base  = has_base ? make_no_copy_buffer((void*)W_base, D_out * D_in * sizeof(uint16_t)) : nil;
        id<MTLBuffer> buf_bias    = has_bias ? make_no_copy_buffer((void*)bias, D_out * sizeof(uint16_t)) : nil;
        id<MTLBuffer> buf_Y       = make_no_copy_buffer((void*)Y, B * D_out * sizeof(uint16_t));

        id<MTLBuffer> buf_Phi     = get_scratch_phi(B * K_dim * sizeof(uint16_t));
        id<MTLBuffer> buf_X_silu  = has_base ? get_scratch_silu(B * D_in * sizeof(uint16_t)) : nil;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        uint uB = (uint)B, uDin = (uint)D_in, uDout = (uint)D_out;
        uint uBase = (uint)has_base, uBias = (uint)has_bias;

        if (has_base || has_bias) {
            id<MTLComputeCommandEncoder> enc_prep = [cmd computeCommandEncoder];
            [enc_prep setComputePipelineState:g_pipe_prep_fp16];
            [enc_prep setBuffer:buf_X offset:0 atIndex:0];
            [enc_prep setBuffer:(buf_bias ? buf_bias : buf_X) offset:0 atIndex:1];
            [enc_prep setBuffer:(buf_X_silu ? buf_X_silu : buf_X) offset:0 atIndex:2];
            [enc_prep setBuffer:buf_Y offset:0 atIndex:3];
            [enc_prep setBytes:&uB length:sizeof(uint) atIndex:4];
            [enc_prep setBytes:&uDin length:sizeof(uint) atIndex:5];
            [enc_prep setBytes:&uDout length:sizeof(uint) atIndex:6];
            [enc_prep setBytes:&uBase length:sizeof(uint) atIndex:7];
            [enc_prep setBytes:&uBias length:sizeof(uint) atIndex:8];
            uint max_dim = (D_in > D_out) ? D_in : D_out;
            [enc_prep dispatchThreads:MTLSizeMake(max_dim, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc_prep endEncoding];
        }

        if (has_base) {
            MPSMatrixDescriptor* desc_A_base = get_cached_desc(B, D_in, D_in * sizeof(uint16_t), MPSDataTypeFloat16);
            MPSMatrixDescriptor* desc_B_base = get_cached_desc(D_out, D_in, D_in * sizeof(uint16_t), MPSDataTypeFloat16);
            MPSMatrixDescriptor* desc_C_base = get_cached_desc(B, D_out, D_out * sizeof(uint16_t), MPSDataTypeFloat16);
            MPSMatrix* mat_A_base = [[[MPSMatrix alloc] initWithBuffer:buf_X_silu descriptor:desc_A_base] autorelease];
            MPSMatrix* mat_B_base = [[[MPSMatrix alloc] initWithBuffer:buf_W_base descriptor:desc_B_base] autorelease];
            MPSMatrix* mat_C_base = [[[MPSMatrix alloc] initWithBuffer:buf_Y descriptor:desc_C_base] autorelease];
            float beta_base = has_bias ? 1.0f : 0.0f;
            MPSMatrixMultiplication* matmul_base = get_cached_matmul(g_device, B, D_out, D_in, 1.0f, beta_base);
            [matmul_base encodeToCommandBuffer:cmd leftMatrix:mat_A_base rightMatrix:mat_B_base resultMatrix:mat_C_base];
        }

        id<MTLComputeCommandEncoder> enc_four = [cmd computeCommandEncoder];
        [enc_four setComputePipelineState:g_pipe_fourier_basis_fp16];
        [enc_four setBuffer:buf_X offset:0 atIndex:0];
        [enc_four setBuffer:buf_Phi offset:0 atIndex:1];
        uint uNumFreq = (uint)num_frequencies;
        [enc_four setBytes:&uB length:sizeof(uint) atIndex:2];
        [enc_four setBytes:&uDin length:sizeof(uint) atIndex:3];
        [enc_four setBytes:&uNumFreq length:sizeof(uint) atIndex:4];
        [enc_four dispatchThreads:MTLSizeMake(D_in, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc_four endEncoding];

        MPSMatrixDescriptor* desc_A_four = get_cached_desc(B, K_dim, K_dim * sizeof(uint16_t), MPSDataTypeFloat16);
        MPSMatrixDescriptor* desc_B_four = get_cached_desc(D_out, K_dim, K_dim * sizeof(uint16_t), MPSDataTypeFloat16);
        MPSMatrixDescriptor* desc_C_four = get_cached_desc(B, D_out, D_out * sizeof(uint16_t), MPSDataTypeFloat16);
        MPSMatrix* mat_A_four = [[[MPSMatrix alloc] initWithBuffer:buf_Phi descriptor:desc_A_four] autorelease];
        MPSMatrix* mat_B_four = [[[MPSMatrix alloc] initWithBuffer:buf_W_four descriptor:desc_B_four] autorelease];
        MPSMatrix* mat_C_four = [[[MPSMatrix alloc] initWithBuffer:buf_Y descriptor:desc_C_four] autorelease];
        float beta_four = (has_base || has_bias) ? 1.0f : 0.0f;
        MPSMatrixMultiplication* matmul_four = get_cached_matmul(g_device, B, D_out, K_dim, 1.0f, beta_four);
        [matmul_four encodeToCommandBuffer:cmd leftMatrix:mat_A_four rightMatrix:mat_B_four resultMatrix:mat_C_four];

        commit_and_sync(cmd);
        return 0;
    }
}

int metal_kan_jacobi_forward_fp16(
    const uint16_t* X,
    const uint16_t* W_jacobi,
    const uint16_t* W_base,
    const uint16_t* bias,
    uint16_t*       Y,
    int B,
    int D_in,
    int D_out,
    int degree,
    float alpha,
    float beta,
    int has_base,
    int has_bias
) {
    @autoreleasepool {
        int K_dim = D_in * degree;

        id<MTLBuffer> buf_X       = make_no_copy_buffer((void*)X, B * D_in * sizeof(uint16_t));
        id<MTLBuffer> buf_W_jac   = make_no_copy_buffer((void*)W_jacobi, D_out * K_dim * sizeof(uint16_t));
        id<MTLBuffer> buf_W_base  = has_base ? make_no_copy_buffer((void*)W_base, D_out * D_in * sizeof(uint16_t)) : nil;
        id<MTLBuffer> buf_bias    = has_bias ? make_no_copy_buffer((void*)bias, D_out * sizeof(uint16_t)) : nil;
        id<MTLBuffer> buf_Y       = make_no_copy_buffer((void*)Y, B * D_out * sizeof(uint16_t));

        id<MTLBuffer> buf_Phi     = get_scratch_phi(B * K_dim * sizeof(uint16_t));
        id<MTLBuffer> buf_X_silu  = has_base ? get_scratch_silu(B * D_in * sizeof(uint16_t)) : nil;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        uint uB = (uint)B, uDin = (uint)D_in, uDout = (uint)D_out;
        uint uBase = (uint)has_base, uBias = (uint)has_bias;

        if (has_base || has_bias) {
            id<MTLComputeCommandEncoder> enc_prep = [cmd computeCommandEncoder];
            [enc_prep setComputePipelineState:g_pipe_prep_fp16];
            [enc_prep setBuffer:buf_X offset:0 atIndex:0];
            [enc_prep setBuffer:(buf_bias ? buf_bias : buf_X) offset:0 atIndex:1];
            [enc_prep setBuffer:(buf_X_silu ? buf_X_silu : buf_X) offset:0 atIndex:2];
            [enc_prep setBuffer:buf_Y offset:0 atIndex:3];
            [enc_prep setBytes:&uB length:sizeof(uint) atIndex:4];
            [enc_prep setBytes:&uDin length:sizeof(uint) atIndex:5];
            [enc_prep setBytes:&uDout length:sizeof(uint) atIndex:6];
            [enc_prep setBytes:&uBase length:sizeof(uint) atIndex:7];
            [enc_prep setBytes:&uBias length:sizeof(uint) atIndex:8];
            uint max_dim = (D_in > D_out) ? D_in : D_out;
            [enc_prep dispatchThreads:MTLSizeMake(max_dim, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc_prep endEncoding];
        }

        if (has_base) {
            MPSMatrixDescriptor* desc_A_base = get_cached_desc(B, D_in, D_in * sizeof(uint16_t), MPSDataTypeFloat16);
            MPSMatrixDescriptor* desc_B_base = get_cached_desc(D_out, D_in, D_in * sizeof(uint16_t), MPSDataTypeFloat16);
            MPSMatrixDescriptor* desc_C_base = get_cached_desc(B, D_out, D_out * sizeof(uint16_t), MPSDataTypeFloat16);
            MPSMatrix* mat_A_base = [[[MPSMatrix alloc] initWithBuffer:buf_X_silu descriptor:desc_A_base] autorelease];
            MPSMatrix* mat_B_base = [[[MPSMatrix alloc] initWithBuffer:buf_W_base descriptor:desc_B_base] autorelease];
            MPSMatrix* mat_C_base = [[[MPSMatrix alloc] initWithBuffer:buf_Y descriptor:desc_C_base] autorelease];
            float beta_base = has_bias ? 1.0f : 0.0f;
            MPSMatrixMultiplication* matmul_base = get_cached_matmul(g_device, B, D_out, D_in, 1.0f, beta_base);
            [matmul_base encodeToCommandBuffer:cmd leftMatrix:mat_A_base rightMatrix:mat_B_base resultMatrix:mat_C_base];
        }

        id<MTLComputeCommandEncoder> enc_jac = [cmd computeCommandEncoder];
        [enc_jac setComputePipelineState:g_pipe_jacobi_basis_fp16];
        [enc_jac setBuffer:buf_X offset:0 atIndex:0];
        [enc_jac setBuffer:buf_Phi offset:0 atIndex:1];
        uint u_deg = (uint)degree;
        float u_alpha = alpha, u_beta = beta;
        [enc_jac setBytes:&uB length:sizeof(uint) atIndex:2];
        [enc_jac setBytes:&uDin length:sizeof(uint) atIndex:3];
        [enc_jac setBytes:&u_deg length:sizeof(uint) atIndex:4];
        [enc_jac setBytes:&u_alpha length:sizeof(float) atIndex:5];
        [enc_jac setBytes:&u_beta length:sizeof(float) atIndex:6];
        [enc_jac dispatchThreads:MTLSizeMake(D_in, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc_jac endEncoding];

        MPSMatrixDescriptor* desc_A_jac = get_cached_desc(B, K_dim, K_dim * sizeof(uint16_t), MPSDataTypeFloat16);
        MPSMatrixDescriptor* desc_B_jac = get_cached_desc(D_out, K_dim, K_dim * sizeof(uint16_t), MPSDataTypeFloat16);
        MPSMatrixDescriptor* desc_C_jac = get_cached_desc(B, D_out, D_out * sizeof(uint16_t), MPSDataTypeFloat16);
        MPSMatrix* mat_A_jac = [[[MPSMatrix alloc] initWithBuffer:buf_Phi descriptor:desc_A_jac] autorelease];
        MPSMatrix* mat_B_jac = [[[MPSMatrix alloc] initWithBuffer:buf_W_jac descriptor:desc_B_jac] autorelease];
        MPSMatrix* mat_C_jac = [[[MPSMatrix alloc] initWithBuffer:buf_Y descriptor:desc_C_jac] autorelease];
        float beta_jac = (has_base || has_bias) ? 1.0f : 0.0f;
        MPSMatrixMultiplication* matmul_jac = get_cached_matmul(g_device, B, D_out, K_dim, 1.0f, beta_jac);
        [matmul_jac encodeToCommandBuffer:cmd leftMatrix:mat_A_jac rightMatrix:mat_B_jac resultMatrix:mat_C_jac];

        commit_and_sync(cmd);
        return 0;
    }
}

// Backward bridge functions for gradient and basis derivatives.

int metal_kan_cheby_backward(
    const float* dY,        // [B, D_out]
    const float* X,         // [B, D_in]
    const float* W_cheby,   // [D_out, D_in * degree]
    const float* W_base,    // [D_out, D_in]
    float*       dW_cheby,  // [D_out, D_in * degree]
    float*       dW_base,   // [D_out, D_in]
    float*       dbias,     // [D_out]
    float*       dX,        // [B, D_in]
    int B,
    int D_in,
    int D_out,
    int degree,
    int has_base,
    int has_bias
) {
    @autoreleasepool {
        int K_dim = D_in * degree;
        id<MTLBuffer> buf_dY       = make_no_copy_buffer((void*)dY, B * D_out * sizeof(float));
        id<MTLBuffer> buf_X        = make_no_copy_buffer((void*)X, B * D_in * sizeof(float));
        id<MTLBuffer> buf_W_cheby  = make_no_copy_buffer((void*)W_cheby, D_out * K_dim * sizeof(float));
        id<MTLBuffer> buf_W_base   = has_base ? make_no_copy_buffer((void*)W_base, D_out * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_dW_cheby = make_no_copy_buffer((void*)dW_cheby, D_out * K_dim * sizeof(float));
        id<MTLBuffer> buf_dW_base  = has_base ? make_no_copy_buffer((void*)dW_base, D_out * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_dbias    = has_bias ? make_no_copy_buffer((void*)dbias, D_out * sizeof(float)) : nil;
        id<MTLBuffer> buf_dX       = make_no_copy_buffer((void*)dX, B * D_in * sizeof(float));

        id<MTLBuffer> buf_Phi      = get_scratch_phi(B * K_dim * sizeof(float));
        id<MTLBuffer> buf_X_silu   = has_base ? get_scratch_silu(B * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_dPhi     = get_scratch_dphi(B * K_dim * sizeof(float));
        id<MTLBuffer> buf_dX_silu  = has_base ? get_scratch_dx_silu(B * D_in * sizeof(float)) : nil;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        uint uB = (uint)B, uDin = (uint)D_in, uDout = (uint)D_out;
        uint uDeg = (uint)degree;
        uint uBase = (uint)has_base, uBias = (uint)has_bias;

        // 1. Bias gradient: dbias = sum_b dY[b, :]
        if (has_bias) {
            id<MTLComputeCommandEncoder> enc_bias = [cmd computeCommandEncoder];
            [enc_bias setComputePipelineState:g_pipe_reduce_sum_cols];
            [enc_bias setBuffer:buf_dY offset:0 atIndex:0];
            [enc_bias setBuffer:buf_dbias offset:0 atIndex:1];
            [enc_bias setBytes:&uB length:sizeof(uint) atIndex:2];
            [enc_bias setBytes:&uDout length:sizeof(uint) atIndex:3];
            [enc_bias dispatchThreads:MTLSizeMake(D_out, 1, 1) threadsPerThreadgroup:MTLSizeMake(std::min(256, D_out), 1, 1)];
            [enc_bias endEncoding];
        }

        // 2. Forward basis re-evaluation: Phi(X) and optional X_silu
        id<MTLComputeCommandEncoder> enc_cheby = [cmd computeCommandEncoder];
        [enc_cheby setComputePipelineState:g_pipe_cheby_basis];
        [enc_cheby setBuffer:buf_X offset:0 atIndex:0];
        [enc_cheby setBuffer:buf_Phi offset:0 atIndex:1];
        [enc_cheby setBytes:&uB length:sizeof(uint) atIndex:2];
        [enc_cheby setBytes:&uDin length:sizeof(uint) atIndex:3];
        [enc_cheby setBytes:&uDeg length:sizeof(uint) atIndex:4];
        [enc_cheby dispatchThreads:MTLSizeMake(D_in, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc_cheby endEncoding];

        if (has_base) {
            id<MTLComputeCommandEncoder> enc_silu = [cmd computeCommandEncoder];
            [enc_silu setComputePipelineState:g_pipe_eval_silu];
            [enc_silu setBuffer:buf_X offset:0 atIndex:0];
            [enc_silu setBuffer:buf_X_silu offset:0 atIndex:1];
            [enc_silu setBytes:&uB length:sizeof(uint) atIndex:2];
            [enc_silu setBytes:&uDin length:sizeof(uint) atIndex:3];
            [enc_silu dispatchThreads:MTLSizeMake(D_in, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc_silu endEncoding];
        }

        // 3. Weight gradients:
        // dW_cheby = dY^T @ Phi  [D_out x B] x [B x K_dim] -> [D_out x K_dim]
        MPSMatrixDescriptor* desc_dY_for_dW = get_cached_desc(B, D_out, D_out * sizeof(float));
        MPSMatrixDescriptor* desc_Phi_for_dW = get_cached_desc(B, K_dim, K_dim * sizeof(float));
        MPSMatrixDescriptor* desc_dW = get_cached_desc(D_out, K_dim, K_dim * sizeof(float));
        MPSMatrix* mat_dY_dW = [[[MPSMatrix alloc] initWithBuffer:buf_dY descriptor:desc_dY_for_dW] autorelease];
        MPSMatrix* mat_Phi_dW = [[[MPSMatrix alloc] initWithBuffer:buf_Phi descriptor:desc_Phi_for_dW] autorelease];
        MPSMatrix* mat_dW = [[[MPSMatrix alloc] initWithBuffer:buf_dW_cheby descriptor:desc_dW] autorelease];
        MPSMatrixMultiplication* mm_dW = get_cached_matmul_general(g_device, YES, NO, D_out, K_dim, B, 1.0f, 0.0f);
        [mm_dW encodeToCommandBuffer:cmd leftMatrix:mat_dY_dW rightMatrix:mat_Phi_dW resultMatrix:mat_dW];

        // If has_base: dW_base = dY^T @ X_silu  [D_out x B] x [B x D_in] -> [D_out x D_in]
        if (has_base) {
            MPSMatrixDescriptor* desc_X_silu = get_cached_desc(B, D_in, D_in * sizeof(float));
            MPSMatrixDescriptor* desc_dW_b = get_cached_desc(D_out, D_in, D_in * sizeof(float));
            MPSMatrix* mat_X_silu = [[[MPSMatrix alloc] initWithBuffer:buf_X_silu descriptor:desc_X_silu] autorelease];
            MPSMatrix* mat_dW_b = [[[MPSMatrix alloc] initWithBuffer:buf_dW_base descriptor:desc_dW_b] autorelease];
            MPSMatrixMultiplication* mm_dWb = get_cached_matmul_general(g_device, YES, NO, D_out, D_in, B, 1.0f, 0.0f);
            [mm_dWb encodeToCommandBuffer:cmd leftMatrix:mat_dY_dW rightMatrix:mat_X_silu resultMatrix:mat_dW_b];
        }

        // 4. Backprop to basis activations:
        // dPhi = dY @ W_cheby  [B x D_out] x [D_out x K_dim] -> [B x K_dim]
        MPSMatrixDescriptor* desc_dY = get_cached_desc(B, D_out, D_out * sizeof(float));
        MPSMatrixDescriptor* desc_W = get_cached_desc(D_out, K_dim, K_dim * sizeof(float));
        MPSMatrixDescriptor* desc_dPhi = get_cached_desc(B, K_dim, K_dim * sizeof(float));
        MPSMatrix* mat_dY = [[[MPSMatrix alloc] initWithBuffer:buf_dY descriptor:desc_dY] autorelease];
        MPSMatrix* mat_W = [[[MPSMatrix alloc] initWithBuffer:buf_W_cheby descriptor:desc_W] autorelease];
        MPSMatrix* mat_dPhi = [[[MPSMatrix alloc] initWithBuffer:buf_dPhi descriptor:desc_dPhi] autorelease];
        MPSMatrixMultiplication* mm_dPhi = get_cached_matmul_general(g_device, NO, NO, B, K_dim, D_out, 1.0f, 0.0f);
        [mm_dPhi encodeToCommandBuffer:cmd leftMatrix:mat_dY rightMatrix:mat_W resultMatrix:mat_dPhi];

        // If has_base: dX_silu = dY @ W_base  [B x D_out] x [D_out x D_in] -> [B x D_in]
        if (has_base) {
            MPSMatrixDescriptor* desc_W_b = get_cached_desc(D_out, D_in, D_in * sizeof(float));
            MPSMatrixDescriptor* desc_dX_s = get_cached_desc(B, D_in, D_in * sizeof(float));
            MPSMatrix* mat_W_b = [[[MPSMatrix alloc] initWithBuffer:buf_W_base descriptor:desc_W_b] autorelease];
            MPSMatrix* mat_dX_s = [[[MPSMatrix alloc] initWithBuffer:buf_dX_silu descriptor:desc_dX_s] autorelease];
            MPSMatrixMultiplication* mm_dxs = get_cached_matmul_general(g_device, NO, NO, B, D_in, D_out, 1.0f, 0.0f);
            [mm_dxs encodeToCommandBuffer:cmd leftMatrix:mat_dY rightMatrix:mat_W_b resultMatrix:mat_dX_s];
        }

        // 5. Compute dX via analytical basis derivatives:
        id<MTLComputeCommandEncoder> enc_back = [cmd computeCommandEncoder];
        [enc_back setComputePipelineState:g_pipe_backward_cheby];
        [enc_back setBuffer:buf_X offset:0 atIndex:0];
        [enc_back setBuffer:buf_dPhi offset:0 atIndex:1];
        [enc_back setBuffer:(buf_dX_silu ? buf_dX_silu : buf_X) offset:0 atIndex:2];
        [enc_back setBuffer:buf_dX offset:0 atIndex:3];
        [enc_back setBytes:&uB length:sizeof(uint) atIndex:4];
        [enc_back setBytes:&uDin length:sizeof(uint) atIndex:5];
        [enc_back setBytes:&uDeg length:sizeof(uint) atIndex:6];
        [enc_back setBytes:&uBase length:sizeof(uint) atIndex:7];
        [enc_back dispatchThreads:MTLSizeMake(D_in, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc_back endEncoding];

        commit_and_sync(cmd);
        return 0;
    }
}

int metal_kan_fastkan_backward(
    const float* dY,        // [B, D_out]
    const float* X,         // [B, D_in]
    const float* W_rbf,     // [D_out, D_in * num_centers]
    const float* W_base,    // [D_out, D_in]
    const float* grid,      // [num_centers]
    float*       dW_rbf,    // [D_out, D_in * num_centers]
    float*       dW_base,   // [D_out, D_in]
    float*       dbias,     // [D_out]
    float*       dX,        // [B, D_in]
    int B,
    int D_in,
    int D_out,
    int num_centers,
    float inv_denominator,
    int has_base,
    int has_bias
) {
    @autoreleasepool {
        int K_dim = D_in * num_centers;
        id<MTLBuffer> buf_dY       = make_no_copy_buffer((void*)dY, B * D_out * sizeof(float));
        id<MTLBuffer> buf_X        = make_no_copy_buffer((void*)X, B * D_in * sizeof(float));
        id<MTLBuffer> buf_W_rbf    = make_no_copy_buffer((void*)W_rbf, D_out * K_dim * sizeof(float));
        id<MTLBuffer> buf_W_base   = has_base ? make_no_copy_buffer((void*)W_base, D_out * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_grid     = make_no_copy_buffer((void*)grid, num_centers * sizeof(float));
        id<MTLBuffer> buf_dW_rbf   = make_no_copy_buffer((void*)dW_rbf, D_out * K_dim * sizeof(float));
        id<MTLBuffer> buf_dW_base  = has_base ? make_no_copy_buffer((void*)dW_base, D_out * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_dbias    = has_bias ? make_no_copy_buffer((void*)dbias, D_out * sizeof(float)) : nil;
        id<MTLBuffer> buf_dX       = make_no_copy_buffer((void*)dX, B * D_in * sizeof(float));

        id<MTLBuffer> buf_Phi      = get_scratch_phi(B * K_dim * sizeof(float));
        id<MTLBuffer> buf_X_silu   = has_base ? get_scratch_silu(B * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_dPhi     = get_scratch_dphi(B * K_dim * sizeof(float));
        id<MTLBuffer> buf_dX_silu  = has_base ? get_scratch_dx_silu(B * D_in * sizeof(float)) : nil;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        uint uB = (uint)B, uDin = (uint)D_in, uDout = (uint)D_out;
        uint uK = (uint)num_centers;
        float u_inv_d = inv_denominator;
        uint uBase = (uint)has_base, uBias = (uint)has_bias;

        // 1. Bias gradient: dbias = sum_b dY[b, :]
        if (has_bias) {
            id<MTLComputeCommandEncoder> enc_bias = [cmd computeCommandEncoder];
            [enc_bias setComputePipelineState:g_pipe_reduce_sum_cols];
            [enc_bias setBuffer:buf_dY offset:0 atIndex:0];
            [enc_bias setBuffer:buf_dbias offset:0 atIndex:1];
            [enc_bias setBytes:&uB length:sizeof(uint) atIndex:2];
            [enc_bias setBytes:&uDout length:sizeof(uint) atIndex:3];
            [enc_bias dispatchThreads:MTLSizeMake(D_out, 1, 1) threadsPerThreadgroup:MTLSizeMake(std::min(256, D_out), 1, 1)];
            [enc_bias endEncoding];
        }

        // 2. Forward basis re-evaluation: Phi(X) and optional X_silu
        id<MTLComputeCommandEncoder> enc_rbf = [cmd computeCommandEncoder];
        [enc_rbf setComputePipelineState:g_pipe_rbf_basis];
        [enc_rbf setBuffer:buf_X offset:0 atIndex:0];
        [enc_rbf setBuffer:buf_grid offset:0 atIndex:1];
        [enc_rbf setBuffer:buf_Phi offset:0 atIndex:2];
        [enc_rbf setBytes:&uB length:sizeof(uint) atIndex:3];
        [enc_rbf setBytes:&uDin length:sizeof(uint) atIndex:4];
        [enc_rbf setBytes:&uK length:sizeof(uint) atIndex:5];
        [enc_rbf setBytes:&u_inv_d length:sizeof(float) atIndex:6];
        [enc_rbf dispatchThreads:MTLSizeMake(D_in, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc_rbf endEncoding];

        if (has_base) {
            id<MTLComputeCommandEncoder> enc_silu = [cmd computeCommandEncoder];
            [enc_silu setComputePipelineState:g_pipe_eval_silu];
            [enc_silu setBuffer:buf_X offset:0 atIndex:0];
            [enc_silu setBuffer:buf_X_silu offset:0 atIndex:1];
            [enc_silu setBytes:&uB length:sizeof(uint) atIndex:2];
            [enc_silu setBytes:&uDin length:sizeof(uint) atIndex:3];
            [enc_silu dispatchThreads:MTLSizeMake(D_in, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc_silu endEncoding];
        }

        // 3. Weight gradients:
        // dW_rbf = dY^T @ Phi  [D_out x B] x [B x K_dim] -> [D_out x K_dim]
        MPSMatrixDescriptor* desc_dY_for_dW = get_cached_desc(B, D_out, D_out * sizeof(float));
        MPSMatrixDescriptor* desc_Phi_for_dW = get_cached_desc(B, K_dim, K_dim * sizeof(float));
        MPSMatrixDescriptor* desc_dW = get_cached_desc(D_out, K_dim, K_dim * sizeof(float));
        MPSMatrix* mat_dY_dW = [[[MPSMatrix alloc] initWithBuffer:buf_dY descriptor:desc_dY_for_dW] autorelease];
        MPSMatrix* mat_Phi_dW = [[[MPSMatrix alloc] initWithBuffer:buf_Phi descriptor:desc_Phi_for_dW] autorelease];
        MPSMatrix* mat_dW = [[[MPSMatrix alloc] initWithBuffer:buf_dW_rbf descriptor:desc_dW] autorelease];
        MPSMatrixMultiplication* mm_dW = get_cached_matmul_general(g_device, YES, NO, D_out, K_dim, B, 1.0f, 0.0f);
        [mm_dW encodeToCommandBuffer:cmd leftMatrix:mat_dY_dW rightMatrix:mat_Phi_dW resultMatrix:mat_dW];

        if (has_base) {
            MPSMatrixDescriptor* desc_X_silu = get_cached_desc(B, D_in, D_in * sizeof(float));
            MPSMatrixDescriptor* desc_dW_b = get_cached_desc(D_out, D_in, D_in * sizeof(float));
            MPSMatrix* mat_X_silu = [[[MPSMatrix alloc] initWithBuffer:buf_X_silu descriptor:desc_X_silu] autorelease];
            MPSMatrix* mat_dW_b = [[[MPSMatrix alloc] initWithBuffer:buf_dW_base descriptor:desc_dW_b] autorelease];
            MPSMatrixMultiplication* mm_dWb = get_cached_matmul_general(g_device, YES, NO, D_out, D_in, B, 1.0f, 0.0f);
            [mm_dWb encodeToCommandBuffer:cmd leftMatrix:mat_dY_dW rightMatrix:mat_X_silu resultMatrix:mat_dW_b];
        }

        // 4. Backprop to basis activations:
        // dPhi = dY @ W_rbf  [B x D_out] x [D_out x K_dim] -> [B x K_dim]
        MPSMatrixDescriptor* desc_dY = get_cached_desc(B, D_out, D_out * sizeof(float));
        MPSMatrixDescriptor* desc_W = get_cached_desc(D_out, K_dim, K_dim * sizeof(float));
        MPSMatrixDescriptor* desc_dPhi = get_cached_desc(B, K_dim, K_dim * sizeof(float));
        MPSMatrix* mat_dY = [[[MPSMatrix alloc] initWithBuffer:buf_dY descriptor:desc_dY] autorelease];
        MPSMatrix* mat_W = [[[MPSMatrix alloc] initWithBuffer:buf_W_rbf descriptor:desc_W] autorelease];
        MPSMatrix* mat_dPhi = [[[MPSMatrix alloc] initWithBuffer:buf_dPhi descriptor:desc_dPhi] autorelease];
        MPSMatrixMultiplication* mm_dPhi = get_cached_matmul_general(g_device, NO, NO, B, K_dim, D_out, 1.0f, 0.0f);
        [mm_dPhi encodeToCommandBuffer:cmd leftMatrix:mat_dY rightMatrix:mat_W resultMatrix:mat_dPhi];

        if (has_base) {
            MPSMatrixDescriptor* desc_W_b = get_cached_desc(D_out, D_in, D_in * sizeof(float));
            MPSMatrixDescriptor* desc_dX_s = get_cached_desc(B, D_in, D_in * sizeof(float));
            MPSMatrix* mat_W_b = [[[MPSMatrix alloc] initWithBuffer:buf_W_base descriptor:desc_W_b] autorelease];
            MPSMatrix* mat_dX_s = [[[MPSMatrix alloc] initWithBuffer:buf_dX_silu descriptor:desc_dX_s] autorelease];
            MPSMatrixMultiplication* mm_dxs = get_cached_matmul_general(g_device, NO, NO, B, D_in, D_out, 1.0f, 0.0f);
            [mm_dxs encodeToCommandBuffer:cmd leftMatrix:mat_dY rightMatrix:mat_W_b resultMatrix:mat_dX_s];
        }

        // 5. Compute dX via analytical FastKAN RBF basis derivatives:
        id<MTLComputeCommandEncoder> enc_back = [cmd computeCommandEncoder];
        [enc_back setComputePipelineState:g_pipe_backward_rbf];
        [enc_back setBuffer:buf_X offset:0 atIndex:0];
        [enc_back setBuffer:buf_grid offset:0 atIndex:1];
        [enc_back setBuffer:buf_dPhi offset:0 atIndex:2];
        [enc_back setBuffer:(buf_dX_silu ? buf_dX_silu : buf_X) offset:0 atIndex:3];
        [enc_back setBuffer:buf_dX offset:0 atIndex:4];
        [enc_back setBytes:&uB length:sizeof(uint) atIndex:5];
        [enc_back setBytes:&uDin length:sizeof(uint) atIndex:6];
        [enc_back setBytes:&uK length:sizeof(uint) atIndex:7];
        [enc_back setBytes:&u_inv_d length:sizeof(float) atIndex:8];
        [enc_back setBytes:&uBase length:sizeof(uint) atIndex:9];
        [enc_back dispatchThreads:MTLSizeMake(D_in, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc_back endEncoding];

        commit_and_sync(cmd);
        return 0;
    }
}

int metal_kan_relu_backward(
    const float* dY,        // [B, D_out]
    const float* X,         // [B, D_in]
    const float* W_relu,    // [D_out, D_in * num_grids]
    const float* W_base,    // [D_out, D_in]
    const float* grid,      // [num_grids]
    float*       dW_relu,   // [D_out, D_in * num_grids]
    float*       dW_base,   // [D_out, D_in]
    float*       dbias,     // [D_out]
    float*       dX,        // [B, D_in]
    int B,
    int D_in,
    int D_out,
    int num_grids,
    float inv_h,
    int has_base,
    int has_bias
) {
    @autoreleasepool {
        int G_dim = D_in * num_grids;
        id<MTLBuffer> buf_dY       = make_no_copy_buffer((void*)dY, B * D_out * sizeof(float));
        id<MTLBuffer> buf_X        = make_no_copy_buffer((void*)X, B * D_in * sizeof(float));
        id<MTLBuffer> buf_W_relu   = make_no_copy_buffer((void*)W_relu, D_out * G_dim * sizeof(float));
        id<MTLBuffer> buf_W_base   = has_base ? make_no_copy_buffer((void*)W_base, D_out * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_grid     = make_no_copy_buffer((void*)grid, num_grids * sizeof(float));
        id<MTLBuffer> buf_dW_relu  = make_no_copy_buffer((void*)dW_relu, D_out * G_dim * sizeof(float));
        id<MTLBuffer> buf_dW_base  = has_base ? make_no_copy_buffer((void*)dW_base, D_out * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_dbias    = has_bias ? make_no_copy_buffer((void*)dbias, D_out * sizeof(float)) : nil;
        id<MTLBuffer> buf_dX       = make_no_copy_buffer((void*)dX, B * D_in * sizeof(float));

        id<MTLBuffer> buf_Phi      = get_scratch_phi(B * G_dim * sizeof(float));
        id<MTLBuffer> buf_X_silu   = has_base ? get_scratch_silu(B * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_dPhi     = get_scratch_dphi(B * G_dim * sizeof(float));
        id<MTLBuffer> buf_dX_silu  = has_base ? get_scratch_dx_silu(B * D_in * sizeof(float)) : nil;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        uint uB = (uint)B, uDin = (uint)D_in, uDout = (uint)D_out;
        uint uG = (uint)num_grids;
        float u_inv_h = inv_h;
        uint uBase = (uint)has_base, uBias = (uint)has_bias;

        // 1. Bias gradient
        if (has_bias) {
            id<MTLComputeCommandEncoder> enc_bias = [cmd computeCommandEncoder];
            [enc_bias setComputePipelineState:g_pipe_reduce_sum_cols];
            [enc_bias setBuffer:buf_dY offset:0 atIndex:0];
            [enc_bias setBuffer:buf_dbias offset:0 atIndex:1];
            [enc_bias setBytes:&uB length:sizeof(uint) atIndex:2];
            [enc_bias setBytes:&uDout length:sizeof(uint) atIndex:3];
            [enc_bias dispatchThreads:MTLSizeMake(D_out, 1, 1) threadsPerThreadgroup:MTLSizeMake(std::min(256, D_out), 1, 1)];
            [enc_bias endEncoding];
        }

        // 2. Basis evaluation
        id<MTLComputeCommandEncoder> enc_relu = [cmd computeCommandEncoder];
        [enc_relu setComputePipelineState:g_pipe_relu_basis];
        [enc_relu setBuffer:buf_X offset:0 atIndex:0];
        [enc_relu setBuffer:buf_grid offset:0 atIndex:1];
        [enc_relu setBuffer:buf_Phi offset:0 atIndex:2];
        [enc_relu setBytes:&uB length:sizeof(uint) atIndex:3];
        [enc_relu setBytes:&uDin length:sizeof(uint) atIndex:4];
        [enc_relu setBytes:&uG length:sizeof(uint) atIndex:5];
        [enc_relu setBytes:&u_inv_h length:sizeof(float) atIndex:6];
        [enc_relu dispatchThreads:MTLSizeMake(D_in, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc_relu endEncoding];

        if (has_base) {
            id<MTLComputeCommandEncoder> enc_silu = [cmd computeCommandEncoder];
            [enc_silu setComputePipelineState:g_pipe_eval_silu];
            [enc_silu setBuffer:buf_X offset:0 atIndex:0];
            [enc_silu setBuffer:buf_X_silu offset:0 atIndex:1];
            [enc_silu setBytes:&uB length:sizeof(uint) atIndex:2];
            [enc_silu setBytes:&uDin length:sizeof(uint) atIndex:3];
            [enc_silu dispatchThreads:MTLSizeMake(D_in, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc_silu endEncoding];
        }

        // 3. Weight gradients: dW_relu = dY^T @ Phi
        MPSMatrixDescriptor* desc_dY_for_dW = get_cached_desc(B, D_out, D_out * sizeof(float));
        MPSMatrixDescriptor* desc_Phi_for_dW = get_cached_desc(B, G_dim, G_dim * sizeof(float));
        MPSMatrixDescriptor* desc_dW = get_cached_desc(D_out, G_dim, G_dim * sizeof(float));
        MPSMatrix* mat_dY_dW = [[[MPSMatrix alloc] initWithBuffer:buf_dY descriptor:desc_dY_for_dW] autorelease];
        MPSMatrix* mat_Phi_dW = [[[MPSMatrix alloc] initWithBuffer:buf_Phi descriptor:desc_Phi_for_dW] autorelease];
        MPSMatrix* mat_dW = [[[MPSMatrix alloc] initWithBuffer:buf_dW_relu descriptor:desc_dW] autorelease];
        MPSMatrixMultiplication* mm_dW = get_cached_matmul_general(g_device, YES, NO, D_out, G_dim, B, 1.0f, 0.0f);
        [mm_dW encodeToCommandBuffer:cmd leftMatrix:mat_dY_dW rightMatrix:mat_Phi_dW resultMatrix:mat_dW];

        if (has_base) {
            MPSMatrixDescriptor* desc_X_silu = get_cached_desc(B, D_in, D_in * sizeof(float));
            MPSMatrixDescriptor* desc_dW_b = get_cached_desc(D_out, D_in, D_in * sizeof(float));
            MPSMatrix* mat_X_silu = [[[MPSMatrix alloc] initWithBuffer:buf_X_silu descriptor:desc_X_silu] autorelease];
            MPSMatrix* mat_dW_b = [[[MPSMatrix alloc] initWithBuffer:buf_dW_base descriptor:desc_dW_b] autorelease];
            MPSMatrixMultiplication* mm_dWb = get_cached_matmul_general(g_device, YES, NO, D_out, D_in, B, 1.0f, 0.0f);
            [mm_dWb encodeToCommandBuffer:cmd leftMatrix:mat_dY_dW rightMatrix:mat_X_silu resultMatrix:mat_dW_b];
        }

        // 4. Backprop to basis: dPhi = dY @ W_relu
        MPSMatrixDescriptor* desc_dY = get_cached_desc(B, D_out, D_out * sizeof(float));
        MPSMatrixDescriptor* desc_W = get_cached_desc(D_out, G_dim, G_dim * sizeof(float));
        MPSMatrixDescriptor* desc_dPhi = get_cached_desc(B, G_dim, G_dim * sizeof(float));
        MPSMatrix* mat_dY = [[[MPSMatrix alloc] initWithBuffer:buf_dY descriptor:desc_dY] autorelease];
        MPSMatrix* mat_W = [[[MPSMatrix alloc] initWithBuffer:buf_W_relu descriptor:desc_W] autorelease];
        MPSMatrix* mat_dPhi = [[[MPSMatrix alloc] initWithBuffer:buf_dPhi descriptor:desc_dPhi] autorelease];
        MPSMatrixMultiplication* mm_dPhi = get_cached_matmul_general(g_device, NO, NO, B, G_dim, D_out, 1.0f, 0.0f);
        [mm_dPhi encodeToCommandBuffer:cmd leftMatrix:mat_dY rightMatrix:mat_W resultMatrix:mat_dPhi];

        if (has_base) {
            MPSMatrixDescriptor* desc_W_b = get_cached_desc(D_out, D_in, D_in * sizeof(float));
            MPSMatrixDescriptor* desc_dX_s = get_cached_desc(B, D_in, D_in * sizeof(float));
            MPSMatrix* mat_W_b = [[[MPSMatrix alloc] initWithBuffer:buf_W_base descriptor:desc_W_b] autorelease];
            MPSMatrix* mat_dX_s = [[[MPSMatrix alloc] initWithBuffer:buf_dX_silu descriptor:desc_dX_s] autorelease];
            MPSMatrixMultiplication* mm_dxs = get_cached_matmul_general(g_device, NO, NO, B, D_in, D_out, 1.0f, 0.0f);
            [mm_dxs encodeToCommandBuffer:cmd leftMatrix:mat_dY rightMatrix:mat_W_b resultMatrix:mat_dX_s];
        }

        // 5. Compute dX
        id<MTLComputeCommandEncoder> enc_back = [cmd computeCommandEncoder];
        [enc_back setComputePipelineState:g_pipe_backward_relu];
        [enc_back setBuffer:buf_X offset:0 atIndex:0];
        [enc_back setBuffer:buf_grid offset:0 atIndex:1];
        [enc_back setBuffer:buf_dPhi offset:0 atIndex:2];
        [enc_back setBuffer:(buf_dX_silu ? buf_dX_silu : buf_X) offset:0 atIndex:3];
        [enc_back setBuffer:buf_dX offset:0 atIndex:4];
        [enc_back setBytes:&uB length:sizeof(uint) atIndex:5];
        [enc_back setBytes:&uDin length:sizeof(uint) atIndex:6];
        [enc_back setBytes:&uG length:sizeof(uint) atIndex:7];
        [enc_back setBytes:&u_inv_h length:sizeof(float) atIndex:8];
        [enc_back setBytes:&uBase length:sizeof(uint) atIndex:9];
        [enc_back dispatchThreads:MTLSizeMake(D_in, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc_back endEncoding];

        commit_and_sync(cmd);
        return 0;
    }
}

int metal_kan_bspline_backward(
    const float* dY,        // [B, D_out]
    const float* X,         // [B, D_in]
    const float* W_spline,  // [D_out, D_in * num_bases]
    const float* W_base,    // [D_out, D_in]
    float*       dW_spline, // [D_out, D_in * num_bases]
    float*       dW_base,   // [D_out, D_in]
    float*       dbias,     // [D_out]
    float*       dX,        // [B, D_in]
    int B,
    int D_in,
    int D_out,
    int grid_size,
    float grid_min,
    float inv_h,
    int has_base,
    int has_bias
) {
    @autoreleasepool {
        int num_bases = grid_size + 3;
        int K_dim = D_in * num_bases;
        id<MTLBuffer> buf_dY       = make_no_copy_buffer((void*)dY, B * D_out * sizeof(float));
        id<MTLBuffer> buf_X        = make_no_copy_buffer((void*)X, B * D_in * sizeof(float));
        id<MTLBuffer> buf_W_spline = make_no_copy_buffer((void*)W_spline, D_out * K_dim * sizeof(float));
        id<MTLBuffer> buf_W_base   = has_base ? make_no_copy_buffer((void*)W_base, D_out * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_dW_spline= make_no_copy_buffer((void*)dW_spline, D_out * K_dim * sizeof(float));
        id<MTLBuffer> buf_dW_base  = has_base ? make_no_copy_buffer((void*)dW_base, D_out * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_dbias    = has_bias ? make_no_copy_buffer((void*)dbias, D_out * sizeof(float)) : nil;
        id<MTLBuffer> buf_dX       = make_no_copy_buffer((void*)dX, B * D_in * sizeof(float));

        id<MTLBuffer> buf_Phi      = get_scratch_phi(B * K_dim * sizeof(float));
        id<MTLBuffer> buf_X_silu   = has_base ? get_scratch_silu(B * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_dPhi     = get_scratch_dphi(B * K_dim * sizeof(float));
        id<MTLBuffer> buf_dX_silu  = has_base ? get_scratch_dx_silu(B * D_in * sizeof(float)) : nil;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        uint uB = (uint)B, uDin = (uint)D_in, uDout = (uint)D_out;
        uint u_grid_size = (uint)grid_size;
        float u_grid_min = grid_min, u_inv_h = inv_h;
        uint uBase = (uint)has_base, uBias = (uint)has_bias;

        // 1. Bias gradient
        if (has_bias) {
            id<MTLComputeCommandEncoder> enc_bias = [cmd computeCommandEncoder];
            [enc_bias setComputePipelineState:g_pipe_reduce_sum_cols];
            [enc_bias setBuffer:buf_dY offset:0 atIndex:0];
            [enc_bias setBuffer:buf_dbias offset:0 atIndex:1];
            [enc_bias setBytes:&uB length:sizeof(uint) atIndex:2];
            [enc_bias setBytes:&uDout length:sizeof(uint) atIndex:3];
            [enc_bias dispatchThreads:MTLSizeMake(D_out, 1, 1) threadsPerThreadgroup:MTLSizeMake(std::min(256, D_out), 1, 1)];
            [enc_bias endEncoding];
        }

        // 2. Basis evaluation
        id<MTLComputeCommandEncoder> enc_bspline = [cmd computeCommandEncoder];
        [enc_bspline setComputePipelineState:g_pipe_bspline_basis];
        [enc_bspline setBuffer:buf_X offset:0 atIndex:0];
        [enc_bspline setBuffer:buf_Phi offset:0 atIndex:1];
        [enc_bspline setBytes:&uB length:sizeof(uint) atIndex:2];
        [enc_bspline setBytes:&uDin length:sizeof(uint) atIndex:3];
        [enc_bspline setBytes:&u_grid_size length:sizeof(uint) atIndex:4];
        [enc_bspline setBytes:&u_grid_min length:sizeof(float) atIndex:5];
        [enc_bspline setBytes:&u_inv_h length:sizeof(float) atIndex:6];
        [enc_bspline dispatchThreads:MTLSizeMake(D_in, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc_bspline endEncoding];

        if (has_base) {
            id<MTLComputeCommandEncoder> enc_silu = [cmd computeCommandEncoder];
            [enc_silu setComputePipelineState:g_pipe_eval_silu];
            [enc_silu setBuffer:buf_X offset:0 atIndex:0];
            [enc_silu setBuffer:buf_X_silu offset:0 atIndex:1];
            [enc_silu setBytes:&uB length:sizeof(uint) atIndex:2];
            [enc_silu setBytes:&uDin length:sizeof(uint) atIndex:3];
            [enc_silu dispatchThreads:MTLSizeMake(D_in, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc_silu endEncoding];
        }

        // 3. Weight gradients: dW_spline = dY^T @ Phi
        MPSMatrixDescriptor* desc_dY_for_dW = get_cached_desc(B, D_out, D_out * sizeof(float));
        MPSMatrixDescriptor* desc_Phi_for_dW = get_cached_desc(B, K_dim, K_dim * sizeof(float));
        MPSMatrixDescriptor* desc_dW = get_cached_desc(D_out, K_dim, K_dim * sizeof(float));
        MPSMatrix* mat_dY_dW = [[[MPSMatrix alloc] initWithBuffer:buf_dY descriptor:desc_dY_for_dW] autorelease];
        MPSMatrix* mat_Phi_dW = [[[MPSMatrix alloc] initWithBuffer:buf_Phi descriptor:desc_Phi_for_dW] autorelease];
        MPSMatrix* mat_dW = [[[MPSMatrix alloc] initWithBuffer:buf_dW_spline descriptor:desc_dW] autorelease];
        MPSMatrixMultiplication* mm_dW = get_cached_matmul_general(g_device, YES, NO, D_out, K_dim, B, 1.0f, 0.0f);
        [mm_dW encodeToCommandBuffer:cmd leftMatrix:mat_dY_dW rightMatrix:mat_Phi_dW resultMatrix:mat_dW];

        if (has_base) {
            MPSMatrixDescriptor* desc_X_silu = get_cached_desc(B, D_in, D_in * sizeof(float));
            MPSMatrixDescriptor* desc_dW_b = get_cached_desc(D_out, D_in, D_in * sizeof(float));
            MPSMatrix* mat_X_silu = [[[MPSMatrix alloc] initWithBuffer:buf_X_silu descriptor:desc_X_silu] autorelease];
            MPSMatrix* mat_dW_b = [[[MPSMatrix alloc] initWithBuffer:buf_dW_base descriptor:desc_dW_b] autorelease];
            MPSMatrixMultiplication* mm_dWb = get_cached_matmul_general(g_device, YES, NO, D_out, D_in, B, 1.0f, 0.0f);
            [mm_dWb encodeToCommandBuffer:cmd leftMatrix:mat_dY_dW rightMatrix:mat_X_silu resultMatrix:mat_dW_b];
        }

        // 4. Backprop to basis: dPhi = dY @ W_spline
        MPSMatrixDescriptor* desc_dY = get_cached_desc(B, D_out, D_out * sizeof(float));
        MPSMatrixDescriptor* desc_W = get_cached_desc(D_out, K_dim, K_dim * sizeof(float));
        MPSMatrixDescriptor* desc_dPhi = get_cached_desc(B, K_dim, K_dim * sizeof(float));
        MPSMatrix* mat_dY = [[[MPSMatrix alloc] initWithBuffer:buf_dY descriptor:desc_dY] autorelease];
        MPSMatrix* mat_W = [[[MPSMatrix alloc] initWithBuffer:buf_W_spline descriptor:desc_W] autorelease];
        MPSMatrix* mat_dPhi = [[[MPSMatrix alloc] initWithBuffer:buf_dPhi descriptor:desc_dPhi] autorelease];
        MPSMatrixMultiplication* mm_dPhi = get_cached_matmul_general(g_device, NO, NO, B, K_dim, D_out, 1.0f, 0.0f);
        [mm_dPhi encodeToCommandBuffer:cmd leftMatrix:mat_dY rightMatrix:mat_W resultMatrix:mat_dPhi];

        if (has_base) {
            MPSMatrixDescriptor* desc_W_b = get_cached_desc(D_out, D_in, D_in * sizeof(float));
            MPSMatrixDescriptor* desc_dX_s = get_cached_desc(B, D_in, D_in * sizeof(float));
            MPSMatrix* mat_W_b = [[[MPSMatrix alloc] initWithBuffer:buf_W_base descriptor:desc_W_b] autorelease];
            MPSMatrix* mat_dX_s = [[[MPSMatrix alloc] initWithBuffer:buf_dX_silu descriptor:desc_dX_s] autorelease];
            MPSMatrixMultiplication* mm_dxs = get_cached_matmul_general(g_device, NO, NO, B, D_in, D_out, 1.0f, 0.0f);
            [mm_dxs encodeToCommandBuffer:cmd leftMatrix:mat_dY rightMatrix:mat_W_b resultMatrix:mat_dX_s];
        }

        // 5. Compute dX
        id<MTLComputeCommandEncoder> enc_back = [cmd computeCommandEncoder];
        [enc_back setComputePipelineState:g_pipe_backward_bspline];
        [enc_back setBuffer:buf_X offset:0 atIndex:0];
        [enc_back setBuffer:buf_dPhi offset:0 atIndex:1];
        [enc_back setBuffer:(buf_dX_silu ? buf_dX_silu : buf_X) offset:0 atIndex:2];
        [enc_back setBuffer:buf_dX offset:0 atIndex:3];
        [enc_back setBytes:&uB length:sizeof(uint) atIndex:4];
        [enc_back setBytes:&uDin length:sizeof(uint) atIndex:5];
        [enc_back setBytes:&u_grid_size length:sizeof(uint) atIndex:6];
        [enc_back setBytes:&u_grid_min length:sizeof(float) atIndex:7];
        [enc_back setBytes:&u_inv_h length:sizeof(float) atIndex:8];
        [enc_back setBytes:&uBase length:sizeof(uint) atIndex:9];
        [enc_back dispatchThreads:MTLSizeMake(D_in, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc_back endEncoding];

        commit_and_sync(cmd);
        return 0;
    }
}

// Native Metal GPU optimizers and fused training steps.

int metal_kan_adamw_step(
    float* param,
    const float* grad,
    float* m,
    float* v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float lr_t,
    int n_elems
) {
    @autoreleasepool {
        if (!param || !grad || !m || !v || n_elems <= 0) return -1;
        id<MTLBuffer> buf_param = make_no_copy_buffer((void*)param, n_elems * sizeof(float));
        id<MTLBuffer> buf_grad  = make_no_copy_buffer((void*)grad, n_elems * sizeof(float));
        id<MTLBuffer> buf_m     = make_no_copy_buffer((void*)m, n_elems * sizeof(float));
        id<MTLBuffer> buf_v     = make_no_copy_buffer((void*)v, n_elems * sizeof(float));

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_pipe_adamw_step];
        [enc setBuffer:buf_param offset:0 atIndex:0];
        [enc setBuffer:buf_grad offset:0 atIndex:1];
        [enc setBuffer:buf_m offset:0 atIndex:2];
        [enc setBuffer:buf_v offset:0 atIndex:3];
        [enc setBytes:&lr length:sizeof(float) atIndex:4];
        [enc setBytes:&beta1 length:sizeof(float) atIndex:5];
        [enc setBytes:&beta2 length:sizeof(float) atIndex:6];
        [enc setBytes:&eps length:sizeof(float) atIndex:7];
        [enc setBytes:&weight_decay length:sizeof(float) atIndex:8];
        [enc setBytes:&lr_t length:sizeof(float) atIndex:9];
        uint u_elems = (uint)n_elems;
        [enc setBytes:&u_elems length:sizeof(uint) atIndex:10];

        uint max_tg = (uint)g_pipe_adamw_step.maxTotalThreadsPerThreadgroup;
        uint tg_size = std::min(max_tg > 0 ? max_tg : 1024u, (uint)n_elems);
        [enc dispatchThreads:MTLSizeMake(n_elems, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];

        commit_and_sync(cmd);
        return 0;
    }
}

int metal_kan_sgd_step(
    float* param,
    const float* grad,
    float* velocity,
    float lr,
    float momentum,
    float weight_decay,
    int nesterov,
    int n_elems
) {
    @autoreleasepool {
        if (!param || !grad || !velocity || n_elems <= 0) return -1;
        id<MTLBuffer> buf_param = make_no_copy_buffer((void*)param, n_elems * sizeof(float));
        id<MTLBuffer> buf_grad  = make_no_copy_buffer((void*)grad, n_elems * sizeof(float));
        id<MTLBuffer> buf_v     = make_no_copy_buffer((void*)velocity, n_elems * sizeof(float));

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_pipe_sgd_step];
        [enc setBuffer:buf_param offset:0 atIndex:0];
        [enc setBuffer:buf_grad offset:0 atIndex:1];
        [enc setBuffer:buf_v offset:0 atIndex:2];
        [enc setBytes:&lr length:sizeof(float) atIndex:3];
        [enc setBytes:&momentum length:sizeof(float) atIndex:4];
        [enc setBytes:&weight_decay length:sizeof(float) atIndex:5];
        uint u_nest = (uint)nesterov;
        [enc setBytes:&u_nest length:sizeof(uint) atIndex:6];
        uint u_elems = (uint)n_elems;
        [enc setBytes:&u_elems length:sizeof(uint) atIndex:7];

        uint max_tg = (uint)g_pipe_sgd_step.maxTotalThreadsPerThreadgroup;
        uint tg_size = std::min(max_tg > 0 ? max_tg : 1024u, (uint)n_elems);
        [enc dispatchThreads:MTLSizeMake(n_elems, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];

        commit_and_sync(cmd);
        return 0;
    }
}

int metal_kan_lion_step(
    float* param,
    const float* grad,
    float* m,
    float lr,
    float beta1,
    float beta2,
    float weight_decay,
    int n_elems
) {
    @autoreleasepool {
        if (!param || !grad || !m || n_elems <= 0) return -1;
        id<MTLBuffer> buf_param = make_no_copy_buffer((void*)param, n_elems * sizeof(float));
        id<MTLBuffer> buf_grad  = make_no_copy_buffer((void*)grad, n_elems * sizeof(float));
        id<MTLBuffer> buf_m     = make_no_copy_buffer((void*)m, n_elems * sizeof(float));

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_pipe_lion_step];
        [enc setBuffer:buf_param offset:0 atIndex:0];
        [enc setBuffer:buf_grad offset:0 atIndex:1];
        [enc setBuffer:buf_m offset:0 atIndex:2];
        [enc setBytes:&lr length:sizeof(float) atIndex:3];
        [enc setBytes:&beta1 length:sizeof(float) atIndex:4];
        [enc setBytes:&beta2 length:sizeof(float) atIndex:5];
        [enc setBytes:&weight_decay length:sizeof(float) atIndex:6];
        uint u_elems = (uint)n_elems;
        [enc setBytes:&u_elems length:sizeof(uint) atIndex:7];

        uint max_tg = (uint)g_pipe_lion_step.maxTotalThreadsPerThreadgroup;
        uint tg_size = std::min(max_tg > 0 ? max_tg : 1024u, (uint)n_elems);
        [enc dispatchThreads:MTLSizeMake(n_elems, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];

        commit_and_sync(cmd);
        return 0;
    }
}

int metal_kan_rmsprop_step(
    float* param,
    const float* grad,
    float* v,
    float* buf,
    float lr,
    float alpha,
    float eps,
    float weight_decay,
    float momentum,
    int has_momentum,
    int n_elems
) {
    @autoreleasepool {
        if (!param || !grad || !v || n_elems <= 0) return -1;
        id<MTLBuffer> buf_param = make_no_copy_buffer((void*)param, n_elems * sizeof(float));
        id<MTLBuffer> buf_grad  = make_no_copy_buffer((void*)grad, n_elems * sizeof(float));
        id<MTLBuffer> buf_v     = make_no_copy_buffer((void*)v, n_elems * sizeof(float));
        id<MTLBuffer> buf_buf   = buf ? make_no_copy_buffer((void*)buf, n_elems * sizeof(float)) : buf_v;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_pipe_rmsprop_step];
        [enc setBuffer:buf_param offset:0 atIndex:0];
        [enc setBuffer:buf_grad offset:0 atIndex:1];
        [enc setBuffer:buf_v offset:0 atIndex:2];
        [enc setBuffer:buf_buf offset:0 atIndex:3];
        [enc setBytes:&lr length:sizeof(float) atIndex:4];
        [enc setBytes:&alpha length:sizeof(float) atIndex:5];
        [enc setBytes:&eps length:sizeof(float) atIndex:6];
        [enc setBytes:&weight_decay length:sizeof(float) atIndex:7];
        [enc setBytes:&momentum length:sizeof(float) atIndex:8];
        uint u_has_m = (uint)has_momentum;
        [enc setBytes:&u_has_m length:sizeof(uint) atIndex:9];
        uint u_elems = (uint)n_elems;
        [enc setBytes:&u_elems length:sizeof(uint) atIndex:10];

        uint max_tg = (uint)g_pipe_rmsprop_step.maxTotalThreadsPerThreadgroup;
        uint tg_size = std::min(max_tg > 0 ? max_tg : 1024u, (uint)n_elems);
        [enc dispatchThreads:MTLSizeMake(n_elems, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];

        commit_and_sync(cmd);
        return 0;
    }
}


int metal_kan_cheby_train_step(
    const float* X,         // [B, D_in]
    const float* Target,    // [B, D_out]
    float*       W_cheby,   // [D_out, D_in * degree]
    float*       W_base,    // [D_out, D_in] (optional)
    float*       bias,      // [D_out] (optional)
    float*       m_W,       // [D_out, D_in * degree]
    float*       v_W,       // [D_out, D_in * degree]
    float*       m_Wb,      // [D_out, D_in]
    float*       v_Wb,      // [D_out, D_in]
    float*       m_b,       // [D_out]
    float*       v_b,       // [D_out]
    int B,
    int D_in,
    int D_out,
    int degree,
    int has_base,
    int has_bias,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float lr_t
) {
    @autoreleasepool {
        int K_dim = D_in * degree;
        id<MTLBuffer> buf_X        = make_no_copy_buffer((void*)X, B * D_in * sizeof(float));
        id<MTLBuffer> buf_Target   = make_no_copy_buffer((void*)Target, B * D_out * sizeof(float));
        id<MTLBuffer> buf_W_cheby  = make_no_copy_buffer((void*)W_cheby, D_out * K_dim * sizeof(float));
        id<MTLBuffer> buf_W_base   = has_base ? make_no_copy_buffer((void*)W_base, D_out * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_bias     = has_bias ? make_no_copy_buffer((void*)bias, D_out * sizeof(float)) : nil;

        id<MTLBuffer> buf_mW       = make_no_copy_buffer((void*)m_W, D_out * K_dim * sizeof(float));
        id<MTLBuffer> buf_vW       = make_no_copy_buffer((void*)v_W, D_out * K_dim * sizeof(float));
        id<MTLBuffer> buf_mWb      = has_base ? make_no_copy_buffer((void*)m_Wb, D_out * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_vWb      = has_base ? make_no_copy_buffer((void*)v_Wb, D_out * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_mb       = has_bias ? make_no_copy_buffer((void*)m_b, D_out * sizeof(float)) : nil;
        id<MTLBuffer> buf_vb       = has_bias ? make_no_copy_buffer((void*)v_b, D_out * sizeof(float)) : nil;

        id<MTLBuffer> buf_Phi      = get_scratch_phi(B * K_dim * sizeof(float));
        id<MTLBuffer> buf_X_silu   = has_base ? get_scratch_silu(B * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_Y        = get_scratch_y(B * D_out * sizeof(float));
        id<MTLBuffer> buf_dY       = get_scratch_dy(B * D_out * sizeof(float));
        id<MTLBuffer> buf_dW_cheby = get_scratch_dw(D_out * K_dim * sizeof(float));
        id<MTLBuffer> buf_dW_base  = has_base ? get_scratch_dw_base(D_out * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_dbias    = has_bias ? get_scratch_dbias(D_out * sizeof(float)) : nil;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        uint uB = (uint)B, uDin = (uint)D_in, uDout = (uint)D_out, uDeg = (uint)degree;
        uint uBase = (uint)has_base, uBias = (uint)has_bias;

        // 1. Forward base & bias
        if (has_base || has_bias) {
            id<MTLComputeCommandEncoder> enc_prep = [cmd computeCommandEncoder];
            [enc_prep setComputePipelineState:g_pipe_prep];
            [enc_prep setBuffer:buf_X offset:0 atIndex:0];
            [enc_prep setBuffer:(buf_bias ? buf_bias : buf_X) offset:0 atIndex:1];
            [enc_prep setBuffer:(buf_X_silu ? buf_X_silu : buf_X) offset:0 atIndex:2];
            [enc_prep setBuffer:buf_Y offset:0 atIndex:3];
            [enc_prep setBytes:&uB length:sizeof(uint) atIndex:4];
            [enc_prep setBytes:&uDin length:sizeof(uint) atIndex:5];
            [enc_prep setBytes:&uDout length:sizeof(uint) atIndex:6];
            [enc_prep setBytes:&uBase length:sizeof(uint) atIndex:7];
            [enc_prep setBytes:&uBias length:sizeof(uint) atIndex:8];
            uint max_dim = (D_in > D_out) ? D_in : D_out;
            [enc_prep dispatchThreads:MTLSizeMake(max_dim, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc_prep endEncoding];
        }

        if (has_base) {
            MPSMatrixDescriptor* desc_A_b = get_cached_desc(B, D_in, D_in * sizeof(float));
            MPSMatrixDescriptor* desc_B_b = get_cached_desc(D_out, D_in, D_in * sizeof(float));
            MPSMatrixDescriptor* desc_C_b = get_cached_desc(B, D_out, D_out * sizeof(float));
            MPSMatrix* mat_A_b = [[[MPSMatrix alloc] initWithBuffer:buf_X_silu descriptor:desc_A_b] autorelease];
            MPSMatrix* mat_B_b = [[[MPSMatrix alloc] initWithBuffer:buf_W_base descriptor:desc_B_b] autorelease];
            MPSMatrix* mat_C_b = [[[MPSMatrix alloc] initWithBuffer:buf_Y descriptor:desc_C_b] autorelease];
            float beta_b = has_bias ? 1.0f : 0.0f;
            MPSMatrixMultiplication* mm_base = get_cached_matmul(g_device, B, D_out, D_in, 1.0f, beta_b);
            [mm_base encodeToCommandBuffer:cmd leftMatrix:mat_A_b rightMatrix:mat_B_b resultMatrix:mat_C_b];
        }

        // Cheby basis Phi(X)
        id<MTLComputeCommandEncoder> enc_cheby = [cmd computeCommandEncoder];
        [enc_cheby setComputePipelineState:g_pipe_cheby_basis];
        [enc_cheby setBuffer:buf_X offset:0 atIndex:0];
        [enc_cheby setBuffer:buf_Phi offset:0 atIndex:1];
        [enc_cheby setBytes:&uB length:sizeof(uint) atIndex:2];
        [enc_cheby setBytes:&uDin length:sizeof(uint) atIndex:3];
        [enc_cheby setBytes:&uDeg length:sizeof(uint) atIndex:4];
        [enc_cheby dispatchThreads:MTLSizeMake(D_in, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc_cheby endEncoding];

        // Cheby GEMM: Y += Phi @ W_cheby.T
        MPSMatrixDescriptor* desc_Phi = get_cached_desc(B, K_dim, K_dim * sizeof(float));
        MPSMatrixDescriptor* desc_W = get_cached_desc(D_out, K_dim, K_dim * sizeof(float));
        MPSMatrixDescriptor* desc_Y = get_cached_desc(B, D_out, D_out * sizeof(float));
        MPSMatrix* mat_Phi = [[[MPSMatrix alloc] initWithBuffer:buf_Phi descriptor:desc_Phi] autorelease];
        MPSMatrix* mat_W = [[[MPSMatrix alloc] initWithBuffer:buf_W_cheby descriptor:desc_W] autorelease];
        MPSMatrix* mat_Y = [[[MPSMatrix alloc] initWithBuffer:buf_Y descriptor:desc_Y] autorelease];
        float beta_cheby = (has_base || has_bias) ? 1.0f : 0.0f;
        MPSMatrixMultiplication* mm_cheby = get_cached_matmul(g_device, B, D_out, K_dim, 1.0f, beta_cheby);
        [mm_cheby encodeToCommandBuffer:cmd leftMatrix:mat_Phi rightMatrix:mat_W resultMatrix:mat_Y];

        // 2. MSE Loss gradient: dY = (Y - Target) * (2/B)
        uint total_y = (uint)(B * D_out);
        float scale = 2.0f / (float)B;
        id<MTLComputeCommandEncoder> enc_loss = [cmd computeCommandEncoder];
        [enc_loss setComputePipelineState:g_pipe_mse_loss_backward];
        [enc_loss setBuffer:buf_Y offset:0 atIndex:0];
        [enc_loss setBuffer:buf_Target offset:0 atIndex:1];
        [enc_loss setBuffer:buf_dY offset:0 atIndex:2];
        [enc_loss setBytes:&total_y length:sizeof(uint) atIndex:3];
        [enc_loss setBytes:&scale length:sizeof(float) atIndex:4];
        uint tg_loss = std::min(256u, total_y);
        [enc_loss dispatchThreads:MTLSizeMake(total_y, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_loss, 1, 1)];
        [enc_loss endEncoding];

        // 3. Weight gradients:
        if (has_bias) {
            id<MTLComputeCommandEncoder> enc_bias = [cmd computeCommandEncoder];
            [enc_bias setComputePipelineState:g_pipe_reduce_sum_cols];
            [enc_bias setBuffer:buf_dY offset:0 atIndex:0];
            [enc_bias setBuffer:buf_dbias offset:0 atIndex:1];
            [enc_bias setBytes:&uB length:sizeof(uint) atIndex:2];
            [enc_bias setBytes:&uDout length:sizeof(uint) atIndex:3];
            [enc_bias dispatchThreads:MTLSizeMake(D_out, 1, 1) threadsPerThreadgroup:MTLSizeMake(std::min(256, D_out), 1, 1)];
            [enc_bias endEncoding];
        }

        MPSMatrixDescriptor* desc_dY = get_cached_desc(B, D_out, D_out * sizeof(float));
        MPSMatrixDescriptor* desc_dW = get_cached_desc(D_out, K_dim, K_dim * sizeof(float));
        MPSMatrix* mat_dY_dW = [[[MPSMatrix alloc] initWithBuffer:buf_dY descriptor:desc_dY] autorelease];
        MPSMatrix* mat_dW = [[[MPSMatrix alloc] initWithBuffer:buf_dW_cheby descriptor:desc_dW] autorelease];
        MPSMatrixMultiplication* mm_dW = get_cached_matmul_general(g_device, YES, NO, D_out, K_dim, B, 1.0f, 0.0f);
        [mm_dW encodeToCommandBuffer:cmd leftMatrix:mat_dY_dW rightMatrix:mat_Phi resultMatrix:mat_dW];

        if (has_base) {
            MPSMatrixDescriptor* desc_X_silu = get_cached_desc(B, D_in, D_in * sizeof(float));
            MPSMatrixDescriptor* desc_dW_b = get_cached_desc(D_out, D_in, D_in * sizeof(float));
            MPSMatrix* mat_X_silu = [[[MPSMatrix alloc] initWithBuffer:buf_X_silu descriptor:desc_X_silu] autorelease];
            MPSMatrix* mat_dW_b = [[[MPSMatrix alloc] initWithBuffer:buf_dW_base descriptor:desc_dW_b] autorelease];
            MPSMatrixMultiplication* mm_dWb = get_cached_matmul_general(g_device, YES, NO, D_out, D_in, B, 1.0f, 0.0f);
            [mm_dWb encodeToCommandBuffer:cmd leftMatrix:mat_dY_dW rightMatrix:mat_X_silu resultMatrix:mat_dW_b];
        }

        // 4. AdamW update on GPU
        uint n_w = (uint)(D_out * K_dim);
        id<MTLComputeCommandEncoder> enc_opt_w = [cmd computeCommandEncoder];
        [enc_opt_w setComputePipelineState:g_pipe_adamw_step];
        [enc_opt_w setBuffer:buf_W_cheby offset:0 atIndex:0];
        [enc_opt_w setBuffer:buf_dW_cheby offset:0 atIndex:1];
        [enc_opt_w setBuffer:buf_mW offset:0 atIndex:2];
        [enc_opt_w setBuffer:buf_vW offset:0 atIndex:3];
        [enc_opt_w setBytes:&lr length:sizeof(float) atIndex:4];
        [enc_opt_w setBytes:&beta1 length:sizeof(float) atIndex:5];
        [enc_opt_w setBytes:&beta2 length:sizeof(float) atIndex:6];
        [enc_opt_w setBytes:&eps length:sizeof(float) atIndex:7];
        [enc_opt_w setBytes:&weight_decay length:sizeof(float) atIndex:8];
        [enc_opt_w setBytes:&lr_t length:sizeof(float) atIndex:9];
        [enc_opt_w setBytes:&n_w length:sizeof(uint) atIndex:10];
        uint tg_w = std::min(256u, n_w);
        [enc_opt_w dispatchThreads:MTLSizeMake(n_w, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_w, 1, 1)];
        [enc_opt_w endEncoding];

        if (has_base) {
            uint n_wb = (uint)(D_out * D_in);
            id<MTLComputeCommandEncoder> enc_opt_wb = [cmd computeCommandEncoder];
            [enc_opt_wb setComputePipelineState:g_pipe_adamw_step];
            [enc_opt_wb setBuffer:buf_W_base offset:0 atIndex:0];
            [enc_opt_wb setBuffer:buf_dW_base offset:0 atIndex:1];
            [enc_opt_wb setBuffer:buf_mWb offset:0 atIndex:2];
            [enc_opt_wb setBuffer:buf_vWb offset:0 atIndex:3];
            [enc_opt_wb setBytes:&lr length:sizeof(float) atIndex:4];
            [enc_opt_wb setBytes:&beta1 length:sizeof(float) atIndex:5];
            [enc_opt_wb setBytes:&beta2 length:sizeof(float) atIndex:6];
            [enc_opt_wb setBytes:&eps length:sizeof(float) atIndex:7];
            [enc_opt_wb setBytes:&weight_decay length:sizeof(float) atIndex:8];
            [enc_opt_wb setBytes:&lr_t length:sizeof(float) atIndex:9];
            [enc_opt_wb setBytes:&n_wb length:sizeof(uint) atIndex:10];
            uint tg_wb = std::min(256u, n_wb);
            [enc_opt_wb dispatchThreads:MTLSizeMake(n_wb, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_wb, 1, 1)];
            [enc_opt_wb endEncoding];
        }

        if (has_bias) {
            uint n_b = (uint)D_out;
            id<MTLComputeCommandEncoder> enc_opt_b = [cmd computeCommandEncoder];
            [enc_opt_b setComputePipelineState:g_pipe_adamw_step];
            [enc_opt_b setBuffer:buf_bias offset:0 atIndex:0];
            [enc_opt_b setBuffer:buf_dbias offset:0 atIndex:1];
            [enc_opt_b setBuffer:buf_mb offset:0 atIndex:2];
            [enc_opt_b setBuffer:buf_vb offset:0 atIndex:3];
            [enc_opt_b setBytes:&lr length:sizeof(float) atIndex:4];
            [enc_opt_b setBytes:&beta1 length:sizeof(float) atIndex:5];
            [enc_opt_b setBytes:&beta2 length:sizeof(float) atIndex:6];
            [enc_opt_b setBytes:&eps length:sizeof(float) atIndex:7];
            [enc_opt_b setBytes:&weight_decay length:sizeof(float) atIndex:8];
            [enc_opt_b setBytes:&lr_t length:sizeof(float) atIndex:9];
            [enc_opt_b setBytes:&n_b length:sizeof(uint) atIndex:10];
            uint tg_b = std::min(256u, n_b);
            [enc_opt_b dispatchThreads:MTLSizeMake(n_b, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_b, 1, 1)];
            [enc_opt_b endEncoding];
        }

        commit_and_sync(cmd);
        return 0;
    }
}

int metal_kan_fastkan_train_step(
    const float* X,
    const float* Target,
    float*       W_rbf,
    float*       W_base,
    const float* grid,
    float*       bias,
    float*       m_W,
    float*       v_W,
    float*       m_Wb,
    float*       v_Wb,
    float*       m_b,
    float*       v_b,
    int B,
    int D_in,
    int D_out,
    int num_centers,
    float inv_denominator,
    int has_base,
    int has_bias,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float lr_t
) {
    @autoreleasepool {
        int K_dim = D_in * num_centers;
        id<MTLBuffer> buf_X        = make_no_copy_buffer((void*)X, B * D_in * sizeof(float));
        id<MTLBuffer> buf_Target   = make_no_copy_buffer((void*)Target, B * D_out * sizeof(float));
        id<MTLBuffer> buf_W_rbf    = make_no_copy_buffer((void*)W_rbf, D_out * K_dim * sizeof(float));
        id<MTLBuffer> buf_W_base   = has_base ? make_no_copy_buffer((void*)W_base, D_out * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_grid     = make_no_copy_buffer((void*)grid, num_centers * sizeof(float));
        id<MTLBuffer> buf_bias     = has_bias ? make_no_copy_buffer((void*)bias, D_out * sizeof(float)) : nil;

        id<MTLBuffer> buf_mW       = make_no_copy_buffer((void*)m_W, D_out * K_dim * sizeof(float));
        id<MTLBuffer> buf_vW       = make_no_copy_buffer((void*)v_W, D_out * K_dim * sizeof(float));
        id<MTLBuffer> buf_mWb      = has_base ? make_no_copy_buffer((void*)m_Wb, D_out * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_vWb      = has_base ? make_no_copy_buffer((void*)v_Wb, D_out * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_mb       = has_bias ? make_no_copy_buffer((void*)m_b, D_out * sizeof(float)) : nil;
        id<MTLBuffer> buf_vb       = has_bias ? make_no_copy_buffer((void*)v_b, D_out * sizeof(float)) : nil;

        id<MTLBuffer> buf_Phi      = get_scratch_phi(B * K_dim * sizeof(float));
        id<MTLBuffer> buf_X_silu   = has_base ? get_scratch_silu(B * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_Y        = get_scratch_y(B * D_out * sizeof(float));
        id<MTLBuffer> buf_dY       = get_scratch_dy(B * D_out * sizeof(float));
        id<MTLBuffer> buf_dW_rbf   = get_scratch_dw(D_out * K_dim * sizeof(float));
        id<MTLBuffer> buf_dW_base  = has_base ? get_scratch_dw_base(D_out * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_dbias    = has_bias ? get_scratch_dbias(D_out * sizeof(float)) : nil;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        uint uB = (uint)B, uDin = (uint)D_in, uDout = (uint)D_out, uK = (uint)num_centers;
        float u_inv_d = inv_denominator;
        uint uBase = (uint)has_base, uBias = (uint)has_bias;

        if (has_base || has_bias) {
            id<MTLComputeCommandEncoder> enc_prep = [cmd computeCommandEncoder];
            [enc_prep setComputePipelineState:g_pipe_prep];
            [enc_prep setBuffer:buf_X offset:0 atIndex:0];
            [enc_prep setBuffer:(buf_bias ? buf_bias : buf_X) offset:0 atIndex:1];
            [enc_prep setBuffer:(buf_X_silu ? buf_X_silu : buf_X) offset:0 atIndex:2];
            [enc_prep setBuffer:buf_Y offset:0 atIndex:3];
            [enc_prep setBytes:&uB length:sizeof(uint) atIndex:4];
            [enc_prep setBytes:&uDin length:sizeof(uint) atIndex:5];
            [enc_prep setBytes:&uDout length:sizeof(uint) atIndex:6];
            [enc_prep setBytes:&uBase length:sizeof(uint) atIndex:7];
            [enc_prep setBytes:&uBias length:sizeof(uint) atIndex:8];
            uint max_dim = (D_in > D_out) ? D_in : D_out;
            [enc_prep dispatchThreads:MTLSizeMake(max_dim, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc_prep endEncoding];
        }

        if (has_base) {
            MPSMatrixDescriptor* desc_A_b = get_cached_desc(B, D_in, D_in * sizeof(float));
            MPSMatrixDescriptor* desc_B_b = get_cached_desc(D_out, D_in, D_in * sizeof(float));
            MPSMatrixDescriptor* desc_C_b = get_cached_desc(B, D_out, D_out * sizeof(float));
            MPSMatrix* mat_A_b = [[[MPSMatrix alloc] initWithBuffer:buf_X_silu descriptor:desc_A_b] autorelease];
            MPSMatrix* mat_B_b = [[[MPSMatrix alloc] initWithBuffer:buf_W_base descriptor:desc_B_b] autorelease];
            MPSMatrix* mat_C_b = [[[MPSMatrix alloc] initWithBuffer:buf_Y descriptor:desc_C_b] autorelease];
            float beta_b = has_bias ? 1.0f : 0.0f;
            MPSMatrixMultiplication* mm_base = get_cached_matmul(g_device, B, D_out, D_in, 1.0f, beta_b);
            [mm_base encodeToCommandBuffer:cmd leftMatrix:mat_A_b rightMatrix:mat_B_b resultMatrix:mat_C_b];
        }

        // FastKAN RBF basis Phi(X)
        id<MTLComputeCommandEncoder> enc_rbf = [cmd computeCommandEncoder];
        [enc_rbf setComputePipelineState:g_pipe_rbf_basis];
        [enc_rbf setBuffer:buf_X offset:0 atIndex:0];
        [enc_rbf setBuffer:buf_grid offset:0 atIndex:1];
        [enc_rbf setBuffer:buf_Phi offset:0 atIndex:2];
        [enc_rbf setBytes:&uB length:sizeof(uint) atIndex:3];
        [enc_rbf setBytes:&uDin length:sizeof(uint) atIndex:4];
        [enc_rbf setBytes:&uK length:sizeof(uint) atIndex:5];
        [enc_rbf setBytes:&u_inv_d length:sizeof(float) atIndex:6];
        [enc_rbf dispatchThreads:MTLSizeMake(D_in, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc_rbf endEncoding];

        // GEMM: Y += Phi @ W_rbf.T
        MPSMatrixDescriptor* desc_Phi = get_cached_desc(B, K_dim, K_dim * sizeof(float));
        MPSMatrixDescriptor* desc_W = get_cached_desc(D_out, K_dim, K_dim * sizeof(float));
        MPSMatrixDescriptor* desc_Y = get_cached_desc(B, D_out, D_out * sizeof(float));
        MPSMatrix* mat_Phi = [[[MPSMatrix alloc] initWithBuffer:buf_Phi descriptor:desc_Phi] autorelease];
        MPSMatrix* mat_W = [[[MPSMatrix alloc] initWithBuffer:buf_W_rbf descriptor:desc_W] autorelease];
        MPSMatrix* mat_Y = [[[MPSMatrix alloc] initWithBuffer:buf_Y descriptor:desc_Y] autorelease];
        float beta_rbf = (has_base || has_bias) ? 1.0f : 0.0f;
        MPSMatrixMultiplication* mm_rbf = get_cached_matmul(g_device, B, D_out, K_dim, 1.0f, beta_rbf);
        [mm_rbf encodeToCommandBuffer:cmd leftMatrix:mat_Phi rightMatrix:mat_W resultMatrix:mat_Y];

        // 2. Loss gradient
        uint total_y = (uint)(B * D_out);
        float scale = 2.0f / (float)B;
        id<MTLComputeCommandEncoder> enc_loss = [cmd computeCommandEncoder];
        [enc_loss setComputePipelineState:g_pipe_mse_loss_backward];
        [enc_loss setBuffer:buf_Y offset:0 atIndex:0];
        [enc_loss setBuffer:buf_Target offset:0 atIndex:1];
        [enc_loss setBuffer:buf_dY offset:0 atIndex:2];
        [enc_loss setBytes:&total_y length:sizeof(uint) atIndex:3];
        [enc_loss setBytes:&scale length:sizeof(float) atIndex:4];
        uint tg_loss = std::min(256u, total_y);
        [enc_loss dispatchThreads:MTLSizeMake(total_y, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_loss, 1, 1)];
        [enc_loss endEncoding];

        // 3. Gradients
        if (has_bias) {
            id<MTLComputeCommandEncoder> enc_bias = [cmd computeCommandEncoder];
            [enc_bias setComputePipelineState:g_pipe_reduce_sum_cols];
            [enc_bias setBuffer:buf_dY offset:0 atIndex:0];
            [enc_bias setBuffer:buf_dbias offset:0 atIndex:1];
            [enc_bias setBytes:&uB length:sizeof(uint) atIndex:2];
            [enc_bias setBytes:&uDout length:sizeof(uint) atIndex:3];
            [enc_bias dispatchThreads:MTLSizeMake(D_out, 1, 1) threadsPerThreadgroup:MTLSizeMake(std::min(256, D_out), 1, 1)];
            [enc_bias endEncoding];
        }

        MPSMatrixDescriptor* desc_dY = get_cached_desc(B, D_out, D_out * sizeof(float));
        MPSMatrixDescriptor* desc_dW = get_cached_desc(D_out, K_dim, K_dim * sizeof(float));
        MPSMatrix* mat_dY_dW = [[[MPSMatrix alloc] initWithBuffer:buf_dY descriptor:desc_dY] autorelease];
        MPSMatrix* mat_dW = [[[MPSMatrix alloc] initWithBuffer:buf_dW_rbf descriptor:desc_dW] autorelease];
        MPSMatrixMultiplication* mm_dW = get_cached_matmul_general(g_device, YES, NO, D_out, K_dim, B, 1.0f, 0.0f);
        [mm_dW encodeToCommandBuffer:cmd leftMatrix:mat_dY_dW rightMatrix:mat_Phi resultMatrix:mat_dW];

        if (has_base) {
            MPSMatrixDescriptor* desc_X_silu = get_cached_desc(B, D_in, D_in * sizeof(float));
            MPSMatrixDescriptor* desc_dW_b = get_cached_desc(D_out, D_in, D_in * sizeof(float));
            MPSMatrix* mat_X_silu = [[[MPSMatrix alloc] initWithBuffer:buf_X_silu descriptor:desc_X_silu] autorelease];
            MPSMatrix* mat_dW_b = [[[MPSMatrix alloc] initWithBuffer:buf_dW_base descriptor:desc_dW_b] autorelease];
            MPSMatrixMultiplication* mm_dWb = get_cached_matmul_general(g_device, YES, NO, D_out, D_in, B, 1.0f, 0.0f);
            [mm_dWb encodeToCommandBuffer:cmd leftMatrix:mat_dY_dW rightMatrix:mat_X_silu resultMatrix:mat_dW_b];
        }

        // 4. AdamW update on GPU
        uint n_w = (uint)(D_out * K_dim);
        id<MTLComputeCommandEncoder> enc_opt_w = [cmd computeCommandEncoder];
        [enc_opt_w setComputePipelineState:g_pipe_adamw_step];
        [enc_opt_w setBuffer:buf_W_rbf offset:0 atIndex:0];
        [enc_opt_w setBuffer:buf_dW_rbf offset:0 atIndex:1];
        [enc_opt_w setBuffer:buf_mW offset:0 atIndex:2];
        [enc_opt_w setBuffer:buf_vW offset:0 atIndex:3];
        [enc_opt_w setBytes:&lr length:sizeof(float) atIndex:4];
        [enc_opt_w setBytes:&beta1 length:sizeof(float) atIndex:5];
        [enc_opt_w setBytes:&beta2 length:sizeof(float) atIndex:6];
        [enc_opt_w setBytes:&eps length:sizeof(float) atIndex:7];
        [enc_opt_w setBytes:&weight_decay length:sizeof(float) atIndex:8];
        [enc_opt_w setBytes:&lr_t length:sizeof(float) atIndex:9];
        [enc_opt_w setBytes:&n_w length:sizeof(uint) atIndex:10];
        uint tg_w = std::min(256u, n_w);
        [enc_opt_w dispatchThreads:MTLSizeMake(n_w, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_w, 1, 1)];
        [enc_opt_w endEncoding];

        if (has_base) {
            uint n_wb = (uint)(D_out * D_in);
            id<MTLComputeCommandEncoder> enc_opt_wb = [cmd computeCommandEncoder];
            [enc_opt_wb setComputePipelineState:g_pipe_adamw_step];
            [enc_opt_wb setBuffer:buf_W_base offset:0 atIndex:0];
            [enc_opt_wb setBuffer:buf_dW_base offset:0 atIndex:1];
            [enc_opt_wb setBuffer:buf_mWb offset:0 atIndex:2];
            [enc_opt_wb setBuffer:buf_vWb offset:0 atIndex:3];
            [enc_opt_wb setBytes:&lr length:sizeof(float) atIndex:4];
            [enc_opt_wb setBytes:&beta1 length:sizeof(float) atIndex:5];
            [enc_opt_wb setBytes:&beta2 length:sizeof(float) atIndex:6];
            [enc_opt_wb setBytes:&eps length:sizeof(float) atIndex:7];
            [enc_opt_wb setBytes:&weight_decay length:sizeof(float) atIndex:8];
            [enc_opt_wb setBytes:&lr_t length:sizeof(float) atIndex:9];
            [enc_opt_wb setBytes:&n_wb length:sizeof(uint) atIndex:10];
            uint tg_wb = std::min(256u, n_wb);
            [enc_opt_wb dispatchThreads:MTLSizeMake(n_wb, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_wb, 1, 1)];
            [enc_opt_wb endEncoding];
        }

        if (has_bias) {
            uint n_b = (uint)D_out;
            id<MTLComputeCommandEncoder> enc_opt_b = [cmd computeCommandEncoder];
            [enc_opt_b setComputePipelineState:g_pipe_adamw_step];
            [enc_opt_b setBuffer:buf_bias offset:0 atIndex:0];
            [enc_opt_b setBuffer:buf_dbias offset:0 atIndex:1];
            [enc_opt_b setBuffer:buf_mb offset:0 atIndex:2];
            [enc_opt_b setBuffer:buf_vb offset:0 atIndex:3];
            [enc_opt_b setBytes:&lr length:sizeof(float) atIndex:4];
            [enc_opt_b setBytes:&beta1 length:sizeof(float) atIndex:5];
            [enc_opt_b setBytes:&beta2 length:sizeof(float) atIndex:6];
            [enc_opt_b setBytes:&eps length:sizeof(float) atIndex:7];
            [enc_opt_b setBytes:&weight_decay length:sizeof(float) atIndex:8];
            [enc_opt_b setBytes:&lr_t length:sizeof(float) atIndex:9];
            [enc_opt_b setBytes:&n_b length:sizeof(uint) atIndex:10];
            uint tg_b = std::min(256u, n_b);
            [enc_opt_b dispatchThreads:MTLSizeMake(n_b, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_b, 1, 1)];
            [enc_opt_b endEncoding];
        }

        commit_and_sync(cmd);
        return 0;
    }
}

int metal_kan_bspline_train_step(
    const float* X,
    const float* Target,
    float*       W_spline,
    float*       W_base,
    float*       bias,
    float*       m_W,
    float*       v_W,
    float*       m_Wb,
    float*       v_Wb,
    float*       m_b,
    float*       v_b,
    int B,
    int D_in,
    int D_out,
    int grid_size,
    float grid_min,
    float grid_max,
    int has_base,
    int has_bias,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float lr_t
) {
    @autoreleasepool {
        int num_bases = grid_size + 3;
        int K_dim = D_in * num_bases;
        id<MTLBuffer> buf_X        = make_no_copy_buffer((void*)X, B * D_in * sizeof(float));
        id<MTLBuffer> buf_Target   = make_no_copy_buffer((void*)Target, B * D_out * sizeof(float));
        id<MTLBuffer> buf_W_spl    = make_no_copy_buffer((void*)W_spline, D_out * K_dim * sizeof(float));
        id<MTLBuffer> buf_W_base   = has_base ? make_no_copy_buffer((void*)W_base, D_out * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_bias     = has_bias ? make_no_copy_buffer((void*)bias, D_out * sizeof(float)) : nil;

        id<MTLBuffer> buf_mW       = make_no_copy_buffer((void*)m_W, D_out * K_dim * sizeof(float));
        id<MTLBuffer> buf_vW       = make_no_copy_buffer((void*)v_W, D_out * K_dim * sizeof(float));
        id<MTLBuffer> buf_mWb      = has_base ? make_no_copy_buffer((void*)m_Wb, D_out * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_vWb      = has_base ? make_no_copy_buffer((void*)v_Wb, D_out * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_mb       = has_bias ? make_no_copy_buffer((void*)m_b, D_out * sizeof(float)) : nil;
        id<MTLBuffer> buf_vb       = has_bias ? make_no_copy_buffer((void*)v_b, D_out * sizeof(float)) : nil;

        id<MTLBuffer> buf_Phi      = get_scratch_phi(B * K_dim * sizeof(float));
        id<MTLBuffer> buf_X_silu   = has_base ? get_scratch_silu(B * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_Y        = get_scratch_y(B * D_out * sizeof(float));
        id<MTLBuffer> buf_dY       = get_scratch_dy(B * D_out * sizeof(float));
        id<MTLBuffer> buf_dW_spl   = get_scratch_dw(D_out * K_dim * sizeof(float));
        id<MTLBuffer> buf_dW_base  = has_base ? get_scratch_dw_base(D_out * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_dbias    = has_bias ? get_scratch_dbias(D_out * sizeof(float)) : nil;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        uint uB = (uint)B, uDin = (uint)D_in, uDout = (uint)D_out, u_grid_size = (uint)grid_size;
        float u_grid_min = grid_min;
        float u_inv_h = (float)grid_size / (grid_max - grid_min);
        uint uBase = (uint)has_base, uBias = (uint)has_bias;

        if (has_base || has_bias) {
            id<MTLComputeCommandEncoder> enc_prep = [cmd computeCommandEncoder];
            [enc_prep setComputePipelineState:g_pipe_prep];
            [enc_prep setBuffer:buf_X offset:0 atIndex:0];
            [enc_prep setBuffer:(buf_bias ? buf_bias : buf_X) offset:0 atIndex:1];
            [enc_prep setBuffer:(buf_X_silu ? buf_X_silu : buf_X) offset:0 atIndex:2];
            [enc_prep setBuffer:buf_Y offset:0 atIndex:3];
            [enc_prep setBytes:&uB length:sizeof(uint) atIndex:4];
            [enc_prep setBytes:&uDin length:sizeof(uint) atIndex:5];
            [enc_prep setBytes:&uDout length:sizeof(uint) atIndex:6];
            [enc_prep setBytes:&uBase length:sizeof(uint) atIndex:7];
            [enc_prep setBytes:&uBias length:sizeof(uint) atIndex:8];
            uint max_dim = (D_in > D_out) ? D_in : D_out;
            [enc_prep dispatchThreads:MTLSizeMake(max_dim, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc_prep endEncoding];
        }

        if (has_base) {
            MPSMatrixDescriptor* desc_A_b = get_cached_desc(B, D_in, D_in * sizeof(float));
            MPSMatrixDescriptor* desc_B_b = get_cached_desc(D_out, D_in, D_in * sizeof(float));
            MPSMatrixDescriptor* desc_C_b = get_cached_desc(B, D_out, D_out * sizeof(float));
            MPSMatrix* mat_A_b = [[[MPSMatrix alloc] initWithBuffer:buf_X_silu descriptor:desc_A_b] autorelease];
            MPSMatrix* mat_B_b = [[[MPSMatrix alloc] initWithBuffer:buf_W_base descriptor:desc_B_b] autorelease];
            MPSMatrix* mat_C_b = [[[MPSMatrix alloc] initWithBuffer:buf_Y descriptor:desc_C_b] autorelease];
            float beta_b = has_bias ? 1.0f : 0.0f;
            MPSMatrixMultiplication* mm_base = get_cached_matmul(g_device, B, D_out, D_in, 1.0f, beta_b);
            [mm_base encodeToCommandBuffer:cmd leftMatrix:mat_A_b rightMatrix:mat_B_b resultMatrix:mat_C_b];
        }

        // BSpline basis
        id<MTLComputeCommandEncoder> enc_spl = [cmd computeCommandEncoder];
        [enc_spl setComputePipelineState:g_pipe_bspline_basis];
        [enc_spl setBuffer:buf_X offset:0 atIndex:0];
        [enc_spl setBuffer:buf_Phi offset:0 atIndex:1];
        [enc_spl setBytes:&uB length:sizeof(uint) atIndex:2];
        [enc_spl setBytes:&uDin length:sizeof(uint) atIndex:3];
        [enc_spl setBytes:&u_grid_size length:sizeof(uint) atIndex:4];
        [enc_spl setBytes:&u_grid_min length:sizeof(float) atIndex:5];
        [enc_spl setBytes:&u_inv_h length:sizeof(float) atIndex:6];
        [enc_spl dispatchThreads:MTLSizeMake(D_in, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc_spl endEncoding];

        // GEMM: Y += Phi @ W_spl.T
        MPSMatrixDescriptor* desc_Phi = get_cached_desc(B, K_dim, K_dim * sizeof(float));
        MPSMatrixDescriptor* desc_W = get_cached_desc(D_out, K_dim, K_dim * sizeof(float));
        MPSMatrixDescriptor* desc_Y = get_cached_desc(B, D_out, D_out * sizeof(float));
        MPSMatrix* mat_Phi = [[[MPSMatrix alloc] initWithBuffer:buf_Phi descriptor:desc_Phi] autorelease];
        MPSMatrix* mat_W = [[[MPSMatrix alloc] initWithBuffer:buf_W_spl descriptor:desc_W] autorelease];
        MPSMatrix* mat_Y = [[[MPSMatrix alloc] initWithBuffer:buf_Y descriptor:desc_Y] autorelease];
        float beta_spl = (has_base || has_bias) ? 1.0f : 0.0f;
        MPSMatrixMultiplication* mm_spl = get_cached_matmul(g_device, B, D_out, K_dim, 1.0f, beta_spl);
        [mm_spl encodeToCommandBuffer:cmd leftMatrix:mat_Phi rightMatrix:mat_W resultMatrix:mat_Y];

        // 2. Loss gradient
        uint total_y = (uint)(B * D_out);
        float scale = 2.0f / (float)B;
        id<MTLComputeCommandEncoder> enc_loss = [cmd computeCommandEncoder];
        [enc_loss setComputePipelineState:g_pipe_mse_loss_backward];
        [enc_loss setBuffer:buf_Y offset:0 atIndex:0];
        [enc_loss setBuffer:buf_Target offset:0 atIndex:1];
        [enc_loss setBuffer:buf_dY offset:0 atIndex:2];
        [enc_loss setBytes:&total_y length:sizeof(uint) atIndex:3];
        [enc_loss setBytes:&scale length:sizeof(float) atIndex:4];
        uint tg_loss = std::min(256u, total_y);
        [enc_loss dispatchThreads:MTLSizeMake(total_y, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_loss, 1, 1)];
        [enc_loss endEncoding];

        // 3. Gradients
        if (has_bias) {
            id<MTLComputeCommandEncoder> enc_bias = [cmd computeCommandEncoder];
            [enc_bias setComputePipelineState:g_pipe_reduce_sum_cols];
            [enc_bias setBuffer:buf_dY offset:0 atIndex:0];
            [enc_bias setBuffer:buf_dbias offset:0 atIndex:1];
            [enc_bias setBytes:&uB length:sizeof(uint) atIndex:2];
            [enc_bias setBytes:&uDout length:sizeof(uint) atIndex:3];
            [enc_bias dispatchThreads:MTLSizeMake(D_out, 1, 1) threadsPerThreadgroup:MTLSizeMake(std::min(256, D_out), 1, 1)];
            [enc_bias endEncoding];
        }

        MPSMatrixDescriptor* desc_dY = get_cached_desc(B, D_out, D_out * sizeof(float));
        MPSMatrixDescriptor* desc_dW = get_cached_desc(D_out, K_dim, K_dim * sizeof(float));
        MPSMatrix* mat_dY_dW = [[[MPSMatrix alloc] initWithBuffer:buf_dY descriptor:desc_dY] autorelease];
        MPSMatrix* mat_dW = [[[MPSMatrix alloc] initWithBuffer:buf_dW_spl descriptor:desc_dW] autorelease];
        MPSMatrixMultiplication* mm_dW = get_cached_matmul_general(g_device, YES, NO, D_out, K_dim, B, 1.0f, 0.0f);
        [mm_dW encodeToCommandBuffer:cmd leftMatrix:mat_dY_dW rightMatrix:mat_Phi resultMatrix:mat_dW];

        if (has_base) {
            MPSMatrixDescriptor* desc_X_silu = get_cached_desc(B, D_in, D_in * sizeof(float));
            MPSMatrixDescriptor* desc_dW_b = get_cached_desc(D_out, D_in, D_in * sizeof(float));
            MPSMatrix* mat_X_silu = [[[MPSMatrix alloc] initWithBuffer:buf_X_silu descriptor:desc_X_silu] autorelease];
            MPSMatrix* mat_dW_b = [[[MPSMatrix alloc] initWithBuffer:buf_dW_base descriptor:desc_dW_b] autorelease];
            MPSMatrixMultiplication* mm_dWb = get_cached_matmul_general(g_device, YES, NO, D_out, D_in, B, 1.0f, 0.0f);
            [mm_dWb encodeToCommandBuffer:cmd leftMatrix:mat_dY_dW rightMatrix:mat_X_silu resultMatrix:mat_dW_b];
        }

        // 4. AdamW update on GPU
        uint n_w = (uint)(D_out * K_dim);
        id<MTLComputeCommandEncoder> enc_opt_w = [cmd computeCommandEncoder];
        [enc_opt_w setComputePipelineState:g_pipe_adamw_step];
        [enc_opt_w setBuffer:buf_W_spl offset:0 atIndex:0];
        [enc_opt_w setBuffer:buf_dW_spl offset:0 atIndex:1];
        [enc_opt_w setBuffer:buf_mW offset:0 atIndex:2];
        [enc_opt_w setBuffer:buf_vW offset:0 atIndex:3];
        [enc_opt_w setBytes:&lr length:sizeof(float) atIndex:4];
        [enc_opt_w setBytes:&beta1 length:sizeof(float) atIndex:5];
        [enc_opt_w setBytes:&beta2 length:sizeof(float) atIndex:6];
        [enc_opt_w setBytes:&eps length:sizeof(float) atIndex:7];
        [enc_opt_w setBytes:&weight_decay length:sizeof(float) atIndex:8];
        [enc_opt_w setBytes:&lr_t length:sizeof(float) atIndex:9];
        [enc_opt_w setBytes:&n_w length:sizeof(uint) atIndex:10];
        uint tg_w = std::min(256u, n_w);
        [enc_opt_w dispatchThreads:MTLSizeMake(n_w, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_w, 1, 1)];
        [enc_opt_w endEncoding];

        if (has_base) {
            uint n_wb = (uint)(D_out * D_in);
            id<MTLComputeCommandEncoder> enc_opt_wb = [cmd computeCommandEncoder];
            [enc_opt_wb setComputePipelineState:g_pipe_adamw_step];
            [enc_opt_wb setBuffer:buf_W_base offset:0 atIndex:0];
            [enc_opt_wb setBuffer:buf_dW_base offset:0 atIndex:1];
            [enc_opt_wb setBuffer:buf_mWb offset:0 atIndex:2];
            [enc_opt_wb setBuffer:buf_vWb offset:0 atIndex:3];
            [enc_opt_wb setBytes:&lr length:sizeof(float) atIndex:4];
            [enc_opt_wb setBytes:&beta1 length:sizeof(float) atIndex:5];
            [enc_opt_wb setBytes:&beta2 length:sizeof(float) atIndex:6];
            [enc_opt_wb setBytes:&eps length:sizeof(float) atIndex:7];
            [enc_opt_wb setBytes:&weight_decay length:sizeof(float) atIndex:8];
            [enc_opt_wb setBytes:&lr_t length:sizeof(float) atIndex:9];
            [enc_opt_wb setBytes:&n_wb length:sizeof(uint) atIndex:10];
            uint tg_wb = std::min(256u, n_wb);
            [enc_opt_wb dispatchThreads:MTLSizeMake(n_wb, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_wb, 1, 1)];
            [enc_opt_wb endEncoding];
        }

        if (has_bias) {
            uint n_b = (uint)D_out;
            id<MTLComputeCommandEncoder> enc_opt_b = [cmd computeCommandEncoder];
            [enc_opt_b setComputePipelineState:g_pipe_adamw_step];
            [enc_opt_b setBuffer:buf_bias offset:0 atIndex:0];
            [enc_opt_b setBuffer:buf_dbias offset:0 atIndex:1];
            [enc_opt_b setBuffer:buf_mb offset:0 atIndex:2];
            [enc_opt_b setBuffer:buf_vb offset:0 atIndex:3];
            [enc_opt_b setBytes:&lr length:sizeof(float) atIndex:4];
            [enc_opt_b setBytes:&beta1 length:sizeof(float) atIndex:5];
            [enc_opt_b setBytes:&beta2 length:sizeof(float) atIndex:6];
            [enc_opt_b setBytes:&eps length:sizeof(float) atIndex:7];
            [enc_opt_b setBytes:&weight_decay length:sizeof(float) atIndex:8];
            [enc_opt_b setBytes:&lr_t length:sizeof(float) atIndex:9];
            [enc_opt_b setBytes:&n_b length:sizeof(uint) atIndex:10];
            uint tg_b = std::min(256u, n_b);
            [enc_opt_b dispatchThreads:MTLSizeMake(n_b, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_b, 1, 1)];
            [enc_opt_b endEncoding];
        }

        commit_and_sync(cmd);
        return 0;
    }
}

int metal_kan_newton_schulz5(
    const float* G_in,
    float* out,
    int rows,
    int cols,
    int steps,
    float eps
) {
    if (!G_in || !out || rows <= 0 || cols <= 0) return -1;
    bool transpose = (rows > cols);
    int M = transpose ? cols : rows;
    int N = transpose ? rows : cols;

    static thread_local std::vector<float> tl_X;
    static thread_local std::vector<float> tl_A;
    static thread_local std::vector<float> tl_A2;
    static thread_local std::vector<float> tl_B;
    static thread_local std::vector<float> tl_BX;

    tl_X.resize(M * N);
    float* X = tl_X.data();

    if (transpose) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                X[c * N + r] = G_in[r * cols + c];
            }
        }
    } else {
        std::memcpy(X, G_in, M * N * sizeof(float));
    }

    // Frobenius norm
    double sum_sq = 0.0;
    for (int i = 0; i < M * N; ++i) {
        sum_sq += (double)X[i] * (double)X[i];
    }
    float norm = (float)std::sqrt(sum_sq) + eps;
    float inv_norm = 1.0f / norm;
    for (int i = 0; i < M * N; ++i) {
        X[i] *= inv_norm;
    }

    const float a = 3.4445f;
    const float b = -4.7750f;
    const float c = 2.0315f;

    tl_A.resize(M * M);
    tl_A2.resize(M * M);
    tl_B.resize(M * M);
    tl_BX.resize(M * N);

    float* A = tl_A.data();
    float* A2 = tl_A2.data();
    float* B = tl_B.data();
    float* BX = tl_BX.data();


    for (int s = 0; s < steps; ++s) {
        // A = X @ X.T: [M, N] x [N, M] -> [M, M]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    M, M, N,
                    1.0f, X, N,
                    X, N,
                    0.0f, A, M);

        // A2 = A @ A: [M, M] x [M, M] -> [M, M]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, M, M,
                    1.0f, A, M,
                    A, M,
                    0.0f, A2, M);

        // B = b * A + c * A2
        for (int i = 0; i < M * M; ++i) {
            B[i] = b * A[i] + c * A2[i];
        }

        // BX = B @ X: [M, M] x [M, N] -> [M, N]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, M,
                    1.0f, B, M,
                    X, N,
                    0.0f, BX, N);

        // X = a * X + BX
        for (int i = 0; i < M * N; ++i) {
            X[i] = a * X[i] + BX[i];
        }
    }

    if (transpose) {
        for (int r = 0; r < M; ++r) {
            for (int c = 0; c < N; ++c) {
                out[c * cols + r] = X[r * N + c];
            }
        }
    } else {
        std::memcpy(out, X, M * N * sizeof(float));
    }

    return 0;
}

int metal_kan_quantize_int8(
    const float* weights,
    uint8_t* qweight,
    float* scales,
    float* biases,
    int num_groups,
    int group_size
) {
    if (!weights || !qweight || !scales || !biases || num_groups <= 0 || group_size <= 0) return -1;
    for (int g = 0; g < num_groups; ++g) {
        const float* group = weights + g * group_size;
        uint8_t* qgroup = qweight + g * group_size;
        float min_val = group[0];
        float max_val = group[0];
        for (int i = 1; i < group_size; ++i) {
            float v = group[i];
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }
        float range_val = std::max(max_val - min_val, 1e-7f);
        float scale = range_val / 255.0f;
        float bias = min_val;
        scales[g] = scale;
        biases[g] = bias;
        float inv_scale = 1.0f / scale;
        for (int i = 0; i < group_size; ++i) {
            float v = (group[i] - bias) * inv_scale + 0.5f;
            int q = (int)v;
            if (q < 0) q = 0;
            else if (q > 255) q = 255;
            qgroup[i] = (uint8_t)q;
        }
    }
    return 0;
}

int metal_kan_dequantize_int8(
    const uint8_t* qweight,
    const float* scales,
    const float* biases,
    float* weights,
    int num_groups,
    int group_size
) {
    if (!qweight || !scales || !biases || !weights || num_groups <= 0 || group_size <= 0) return -1;
    for (int g = 0; g < num_groups; ++g) {
        const uint8_t* qgroup = qweight + g * group_size;
        float* group = weights + g * group_size;
        float scale = scales[g];
        float bias = biases[g];
        for (int i = 0; i < group_size; ++i) {
            group[i] = (float)qgroup[i] * scale + bias;
        }
    }
    return 0;
}

int metal_kan_quantize_int4(
    const float* weights,
    uint8_t* qweight,
    float* scales,
    float* biases,
    int num_groups,
    int group_size
) {
    if (!weights || !qweight || !scales || !biases || num_groups <= 0 || group_size <= 0) return -1;
    int packed_group_size = group_size / 2;
    for (int g = 0; g < num_groups; ++g) {
        const float* group = weights + g * group_size;
        uint8_t* qgroup = qweight + g * packed_group_size;
        float min_val = group[0];
        float max_val = group[0];
        for (int i = 1; i < group_size; ++i) {
            float v = group[i];
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }
        float range_val = std::max(max_val - min_val, 1e-7f);
        float scale = range_val / 15.0f;
        float bias = min_val;
        scales[g] = scale;
        biases[g] = bias;
        float inv_scale = 1.0f / scale;
        for (int i = 0; i < packed_group_size; ++i) {
            float v0 = (group[2 * i] - bias) * inv_scale + 0.5f;
            int q0 = (int)v0;
            if (q0 < 0) q0 = 0;
            else if (q0 > 15) q0 = 15;

            float v1 = (group[2 * i + 1] - bias) * inv_scale + 0.5f;
            int q1 = (int)v1;
            if (q1 < 0) q1 = 0;
            else if (q1 > 15) q1 = 15;

            qgroup[i] = (uint8_t)(q0 | (q1 << 4));
        }
    }
    return 0;
}


int metal_kan_dequantize_int4(
    const uint8_t* qweight,
    const float* scales,
    const float* biases,
    float* weights,
    int num_groups,
    int group_size
) {
    if (!qweight || !scales || !biases || !weights || num_groups <= 0 || group_size <= 0) return -1;
    int packed_group_size = group_size / 2;
    for (int g = 0; g < num_groups; ++g) {
        const uint8_t* qgroup = qweight + g * packed_group_size;
        float* group = weights + g * group_size;
        float scale = scales[g];
        float bias = biases[g];
        for (int i = 0; i < packed_group_size; ++i) {
            uint8_t byte = qgroup[i];
            uint8_t q0 = byte & 0x0F;
            uint8_t q1 = (byte >> 4) & 0x0F;
            group[2 * i] = (float)q0 * scale + bias;
            group[2 * i + 1] = (float)q1 * scale + bias;
        }
    }
    return 0;
}

int metal_kan_prune_layer(
    float* basis_w,
    float* base_w,
    int out_dim,
    int in_dim,
    int num_basis,
    int has_base,
    float threshold
) {
    if (!basis_w || out_dim <= 0 || in_dim <= 0 || num_basis <= 0) return -1;
    for (int j = 0; j < out_dim; ++j) {
        for (int i = 0; i < in_dim; ++i) {
            float sum = 0.0f;
            int offset = j * (in_dim * num_basis) + i * num_basis;
            for (int b = 0; b < num_basis; ++b) {
                sum += std::fabs(basis_w[offset + b]);
            }
            if (sum < threshold) {
                for (int b = 0; b < num_basis; ++b) {
                    basis_w[offset + b] = 0.0f;
                }
                if (has_base && base_w != nullptr) {
                    base_w[j * in_dim + i] = 0.0f;
                }
            }
        }
    }
    return 0;
}

int metal_kan_node_importance_pair(
    const float* basis_w1,
    const float* base_w1,
    int out_dim1,
    int in_dim1,
    int k1,
    int has_base1,
    const float* basis_w2,
    const float* base_w2,
    int out_dim2,
    int in_dim2,
    int k2,
    int has_base2,
    float* scores_out
) {
    if (!basis_w1 || !basis_w2 || !scores_out || out_dim1 <= 0 || in_dim1 <= 0 || out_dim2 <= 0 || in_dim2 <= 0) return -1;
    int H = out_dim1;
    std::vector<float> out_mag(H, 0.0f);
    std::vector<float> in_mag(H, 0.0f);

    for (int j = 0; j < H; ++j) {
        float sum = 0.0f;
        int row_len = in_dim1 * k1;
        for (int c = 0; c < row_len; ++c) {
            sum += std::fabs(basis_w1[j * row_len + c]);
        }
        if (has_base1 && base_w1 != nullptr) {
            for (int c = 0; c < in_dim1; ++c) {
                sum += std::fabs(base_w1[j * in_dim1 + c]);
            }
        }
        out_mag[j] = sum;
    }

    for (int j = 0; j < H; ++j) {
        float sum = 0.0f;
        for (int o = 0; o < out_dim2; ++o) {
            int offset = o * (in_dim2 * k2) + j * k2;
            for (int b = 0; b < k2; ++b) {
                sum += std::fabs(basis_w2[offset + b]);
            }
            if (has_base2 && base_w2 != nullptr) {
                sum += std::fabs(base_w2[o * in_dim2 + j]);
            }
        }
        in_mag[j] = sum;
    }

    for (int j = 0; j < H; ++j) {
        scores_out[j] = out_mag[j] * in_mag[j];
    }
    return 0;
}

// Muon optimizer: GPU Newton-Schulz 5 iteration kernel.

int metal_kan_newton_schulz5_gpu(
    const float* G_in,
    float* out,
    int rows,
    int cols,
    int steps,
    float eps
) {
    @autoreleasepool {
        if (!G_in || !out || rows <= 0 || cols <= 0) return -1;
        bool transpose = (rows > cols);
        int M = transpose ? cols : rows;
        int N = transpose ? rows : cols;

        // Frobenius norm
        double sum_sq = 0.0;
        int total = rows * cols;
        for (int i = 0; i < total; ++i) {
            sum_sq += (double)G_in[i] * (double)G_in[i];
        }
        float norm = (float)std::sqrt(sum_sq) + eps;
        float inv_norm = 1.0f / norm;

        size_t bytes_X  = M * N * sizeof(float);
        size_t bytes_MM = M * M * sizeof(float);

        static id<MTLBuffer> s_ns_x = nil;
        static size_t s_ns_x_cap = 0;
        static id<MTLBuffer> s_ns_a = nil;
        static size_t s_ns_a_cap = 0;
        static id<MTLBuffer> s_ns_a2 = nil;
        static size_t s_ns_a2_cap = 0;
        static id<MTLBuffer> s_ns_b = nil;
        static size_t s_ns_b_cap = 0;
        static id<MTLBuffer> s_ns_bx = nil;
        static size_t s_ns_bx_cap = 0;

        auto get_cached_ns = [](id<MTLBuffer>& s_buf, size_t& s_cap, size_t bytes) -> id<MTLBuffer> {
            if (bytes > s_cap) {
                if (s_buf) [s_buf release];
                s_cap = bytes * 2;
                s_buf = [[g_device newBufferWithLength:s_cap options:MTLResourceStorageModeShared] retain];
            }
            return s_buf;
        };

        id<MTLBuffer> buf_X  = get_cached_ns(s_ns_x, s_ns_x_cap, bytes_X);
        id<MTLBuffer> buf_A  = get_cached_ns(s_ns_a, s_ns_a_cap, bytes_MM);
        id<MTLBuffer> buf_A2 = get_cached_ns(s_ns_a2, s_ns_a2_cap, bytes_MM);
        id<MTLBuffer> buf_B  = get_cached_ns(s_ns_b, s_ns_b_cap, bytes_MM);
        id<MTLBuffer> buf_BX = get_cached_ns(s_ns_bx, s_ns_bx_cap, bytes_X);

        float* x_ptr = (float*)[buf_X contents];
        if (transpose) {
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    x_ptr[c * N + r] = G_in[r * cols + c] * inv_norm;
                }
            }
        } else {
            for (int i = 0; i < total; ++i) {
                x_ptr[i] = G_in[i] * inv_norm;
            }
        }

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];

        const float a_coeff = 3.4445f;
        const float b_coeff = -4.7750f;
        const float c_coeff = 2.0315f;

        MPSMatrixDescriptor* desc_X  = get_cached_desc(M, N, N * sizeof(float));
        MPSMatrixDescriptor* desc_A  = get_cached_desc(M, M, M * sizeof(float));
        MPSMatrixDescriptor* desc_BX = get_cached_desc(M, N, N * sizeof(float));

        MPSMatrix* mat_X  = [[[MPSMatrix alloc] initWithBuffer:buf_X descriptor:desc_X] autorelease];
        MPSMatrix* mat_A  = [[[MPSMatrix alloc] initWithBuffer:buf_A descriptor:desc_A] autorelease];
        MPSMatrix* mat_A2 = [[[MPSMatrix alloc] initWithBuffer:buf_A2 descriptor:desc_A] autorelease];
        MPSMatrix* mat_B  = [[[MPSMatrix alloc] initWithBuffer:buf_B descriptor:desc_A] autorelease];
        MPSMatrix* mat_BX = [[[MPSMatrix alloc] initWithBuffer:buf_BX descriptor:desc_BX] autorelease];

        MPSMatrixMultiplication* mm_XXt = get_cached_matmul_general(g_device, NO, YES, M, M, N, 1.0f, 0.0f);
        MPSMatrixMultiplication* mm_AA  = get_cached_matmul_general(g_device, NO, NO, M, M, M, 1.0f, 0.0f);
        MPSMatrixMultiplication* mm_BX  = get_cached_matmul_general(g_device, NO, NO, M, N, M, 1.0f, 0.0f);

        uint u_MM = (uint)(M * M);
        uint u_MN = (uint)(M * N);
        uint tg_MM = std::min(1024u, u_MM);
        uint tg_MN = std::min(1024u, u_MN);

        for (int s = 0; s < steps; ++s) {
            // A = X @ X^T: [M, N] x [N, M] -> [M, M]
            [mm_XXt encodeToCommandBuffer:cmd leftMatrix:mat_X rightMatrix:mat_X resultMatrix:mat_A];

            // A2 = A @ A: [M, M] x [M, M] -> [M, M]
            [mm_AA encodeToCommandBuffer:cmd leftMatrix:mat_A rightMatrix:mat_A resultMatrix:mat_A2];

            // B = b * A + c * A2
            id<MTLComputeCommandEncoder> enc_B = [cmd computeCommandEncoder];
            [enc_B setComputePipelineState:g_pipe_ns5_combine_B_f32];
            [enc_B setBuffer:buf_A offset:0 atIndex:0];
            [enc_B setBuffer:buf_A2 offset:0 atIndex:1];
            [enc_B setBuffer:buf_B offset:0 atIndex:2];
            [enc_B setBytes:&b_coeff length:sizeof(float) atIndex:3];
            [enc_B setBytes:&c_coeff length:sizeof(float) atIndex:4];
            [enc_B setBytes:&u_MM length:sizeof(uint) atIndex:5];
            [enc_B dispatchThreads:MTLSizeMake(u_MM, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_MM, 1, 1)];
            [enc_B endEncoding];

            // BX = B @ X: [M, M] x [M, N] -> [M, N]
            [mm_BX encodeToCommandBuffer:cmd leftMatrix:mat_B rightMatrix:mat_X resultMatrix:mat_BX];

            // X = a * X + BX
            id<MTLComputeCommandEncoder> enc_X = [cmd computeCommandEncoder];
            [enc_X setComputePipelineState:g_pipe_ns5_combine_X_f32];
            [enc_X setBuffer:buf_X offset:0 atIndex:0];
            [enc_X setBuffer:buf_BX offset:0 atIndex:1];
            [enc_X setBytes:&a_coeff length:sizeof(float) atIndex:2];
            [enc_X setBytes:&u_MN length:sizeof(uint) atIndex:3];
            [enc_X dispatchThreads:MTLSizeMake(u_MN, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_MN, 1, 1)];
            [enc_X endEncoding];
        }

        commit_and_sync(cmd);

        const float* res = (const float*)[buf_X contents];
        if (transpose) {
            for (int r = 0; r < M; ++r) {
                for (int c = 0; c < N; ++c) {
                    out[c * cols + r] = res[r * N + c];
                }
            }
        } else {
            std::memcpy(out, res, bytes_X);
        }

        return 0;
    }
}

// Fused Chebyshev backward pass.

int metal_kan_cheby_backward_fused(
    const float* dY,        // [B, D_out]
    const float* X,         // [B, D_in]
    const float* W_cheby,   // [D_out, D_in * degree]
    const float* W_base,    // [D_out, D_in] (optional)
    float*       dW_cheby,  // [D_out, D_in * degree]
    float*       dW_base,   // [D_out, D_in] (optional)
    float*       dbias,     // [D_out] (optional)
    float*       dX,        // [B, D_in]
    int B,
    int D_in,
    int D_out,
    int degree,
    int has_base,
    int has_bias
) {
    @autoreleasepool {
        if (!dY || !X || !W_cheby || !dW_cheby || !dX || B <= 0 || D_in <= 0 || D_out <= 0 || degree <= 0) return -1;
        uint K_dim = (uint)(D_in * degree);
        id<MTLBuffer> buf_dY       = make_no_copy_buffer((void*)dY, B * D_out * sizeof(float));
        id<MTLBuffer> buf_X        = make_no_copy_buffer((void*)X, B * D_in * sizeof(float));
        id<MTLBuffer> buf_W_cheby  = make_no_copy_buffer((void*)W_cheby, D_out * K_dim * sizeof(float));
        id<MTLBuffer> buf_W_base   = (has_base && W_base) ? make_no_copy_buffer((void*)W_base, D_out * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_dW_cheby = make_no_copy_buffer((void*)dW_cheby, D_out * K_dim * sizeof(float));
        id<MTLBuffer> buf_dW_base  = (has_base && dW_base) ? make_no_copy_buffer((void*)dW_base, D_out * D_in * sizeof(float)) : nil;
        id<MTLBuffer> buf_dbias    = (has_bias && dbias) ? make_no_copy_buffer((void*)dbias, D_out * sizeof(float)) : nil;
        id<MTLBuffer> buf_dX       = make_no_copy_buffer((void*)dX, B * D_in * sizeof(float));

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        uint uB = (uint)B, uDin = (uint)D_in, uDout = (uint)D_out, uDeg = (uint)degree;
        uint uBase = (uint)has_base;

        // dbias
        if (has_bias && dbias) {
            id<MTLComputeCommandEncoder> enc_bias = [cmd computeCommandEncoder];
            [enc_bias setComputePipelineState:g_pipe_reduce_sum_cols];
            [enc_bias setBuffer:buf_dY offset:0 atIndex:0];
            [enc_bias setBuffer:buf_dbias offset:0 atIndex:1];
            [enc_bias setBytes:&uB length:sizeof(uint) atIndex:2];
            [enc_bias setBytes:&uDout length:sizeof(uint) atIndex:3];
            [enc_bias dispatchThreads:MTLSizeMake(D_out, 1, 1) threadsPerThreadgroup:MTLSizeMake(std::min(256, D_out), 1, 1)];
            [enc_bias endEncoding];
        }

        // Fused dW_cheby streaming over B in registers
        id<MTLComputeCommandEncoder> enc_dw = [cmd computeCommandEncoder];
        [enc_dw setComputePipelineState:g_pipe_fused_backward_cheby_dW];
        [enc_dw setBuffer:buf_dY offset:0 atIndex:0];
        [enc_dw setBuffer:buf_X offset:0 atIndex:1];
        [enc_dw setBuffer:buf_dW_cheby offset:0 atIndex:2];
        [enc_dw setBytes:&uB length:sizeof(uint) atIndex:3];
        [enc_dw setBytes:&uDin length:sizeof(uint) atIndex:4];
        [enc_dw setBytes:&uDout length:sizeof(uint) atIndex:5];
        [enc_dw setBytes:&uDeg length:sizeof(uint) atIndex:6];
        [enc_dw dispatchThreads:MTLSizeMake(D_in, D_out, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc_dw endEncoding];

        // If has_base: dW_base = dY^T @ silu(X)
        if (has_base && dW_base) {
            id<MTLBuffer> buf_X_silu = get_scratch_silu(B * D_in * sizeof(float));
            id<MTLComputeCommandEncoder> enc_silu = [cmd computeCommandEncoder];
            [enc_silu setComputePipelineState:g_pipe_eval_silu];
            [enc_silu setBuffer:buf_X offset:0 atIndex:0];
            [enc_silu setBuffer:buf_X_silu offset:0 atIndex:1];
            [enc_silu setBytes:&uB length:sizeof(uint) atIndex:2];
            [enc_silu setBytes:&uDin length:sizeof(uint) atIndex:3];
            [enc_silu dispatchThreads:MTLSizeMake(D_in, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc_silu endEncoding];

            MPSMatrixDescriptor* desc_dY = get_cached_desc(B, D_out, D_out * sizeof(float));
            MPSMatrixDescriptor* desc_Xs = get_cached_desc(B, D_in, D_in * sizeof(float));
            MPSMatrixDescriptor* desc_dWb = get_cached_desc(D_out, D_in, D_in * sizeof(float));
            MPSMatrix* mat_dY = [[[MPSMatrix alloc] initWithBuffer:buf_dY descriptor:desc_dY] autorelease];
            MPSMatrix* mat_Xs = [[[MPSMatrix alloc] initWithBuffer:buf_X_silu descriptor:desc_Xs] autorelease];
            MPSMatrix* mat_dWb = [[[MPSMatrix alloc] initWithBuffer:buf_dW_base descriptor:desc_dWb] autorelease];
            MPSMatrixMultiplication* mm_dWb = get_cached_matmul_general(g_device, YES, NO, D_out, D_in, B, 1.0f, 0.0f);
            [mm_dWb encodeToCommandBuffer:cmd leftMatrix:mat_dY rightMatrix:mat_Xs resultMatrix:mat_dWb];
        }

        // Fused dX streaming basis derivatives in registers
        id<MTLComputeCommandEncoder> enc_dx = [cmd computeCommandEncoder];
        [enc_dx setComputePipelineState:g_pipe_fused_backward_cheby_dX];
        [enc_dx setBuffer:buf_dY offset:0 atIndex:0];
        [enc_dx setBuffer:buf_X offset:0 atIndex:1];
        [enc_dx setBuffer:buf_W_cheby offset:0 atIndex:2];
        [enc_dx setBuffer:(buf_W_base ? buf_W_base : buf_X) offset:0 atIndex:3];
        [enc_dx setBuffer:buf_dX offset:0 atIndex:4];
        [enc_dx setBytes:&uB length:sizeof(uint) atIndex:5];
        [enc_dx setBytes:&uDin length:sizeof(uint) atIndex:6];
        [enc_dx setBytes:&uDout length:sizeof(uint) atIndex:7];
        [enc_dx setBytes:&uDeg length:sizeof(uint) atIndex:8];
        [enc_dx setBytes:&uBase length:sizeof(uint) atIndex:9];
        [enc_dx dispatchThreads:MTLSizeMake(D_in, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc_dx endEncoding];

        commit_and_sync(cmd);
        return 0;
    }
}

// Sub-4-bit quantization: Ternary (1.58-bit) and INT2 APIs.

int metal_kan_quantize_ternary(
    const float* W,
    uint32_t* W_packed,
    float* scales,
    int D_out,
    int K_dim,
    int group_size
) {
    if (!W || !W_packed || !scales || D_out <= 0 || K_dim <= 0 || group_size <= 0) return -1;
    int words_per_row = (K_dim + 15) / 16;
    int groups_per_row = (K_dim + group_size - 1) / group_size;

    std::memset(W_packed, 0, D_out * words_per_row * sizeof(uint32_t));

    for (int j = 0; j < D_out; ++j) {
        const float* w_row = W + j * K_dim;
        uint32_t* packed_row = W_packed + j * words_per_row;
        float* s_row = scales + j * groups_per_row;

        for (int g = 0; g < groups_per_row; ++g) {
            int start_idx = g * group_size;
            int end_idx = std::min(start_idx + group_size, K_dim);
            int count = end_idx - start_idx;

            // Compute mean absolute scale
            double sum_abs = 0.0;
            for (int k = start_idx; k < end_idx; ++k) {
                sum_abs += std::fabs((double)w_row[k]);
            }
            float sc = (count > 0 && sum_abs > 0.0) ? (float)(sum_abs / count) : 1e-4f;
            s_row[g] = sc;
            float inv_sc = 1.0f / sc;

            for (int k = start_idx; k < end_idx; ++k) {
                float val = w_row[k] * inv_sc;
                // Ternary quantize: -1 if val < -0.5, 0 if |val| <= 0.5, +1 if val > 0.5
                // Mapping: -1 -> code 0 (00b), 0 -> code 1 (01b), +1 -> code 2 (10b)
                uint32_t code = 1;
                if (val < -0.5f) code = 0;
                else if (val > 0.5f) code = 2;

                int word_idx = k / 16;
                int bit_shift = (k % 16) * 2;
                packed_row[word_idx] |= (code << bit_shift);
            }
        }
    }
    return 0;
}

int metal_kan_forward_ternary_cheby(
    const float* X,
    const uint32_t* W_packed,
    const float* scales,
    const float* bias,
    float* Y,
    int B,
    int D_in,
    int D_out,
    int degree,
    int group_size,
    int has_bias
) {
    @autoreleasepool {
        if (!X || !W_packed || !scales || !Y || B <= 0 || D_in <= 0 || D_out <= 0 || degree <= 0) return -1;
        uint K_dim = (uint)(D_in * degree);
        uint words_per_row = (K_dim + 15) / 16;
        uint groups_per_row = (K_dim + group_size - 1) / group_size;

        id<MTLBuffer> buf_X        = make_no_copy_buffer((void*)X, B * D_in * sizeof(float));
        id<MTLBuffer> buf_W_packed = make_no_copy_buffer((void*)W_packed, D_out * words_per_row * sizeof(uint32_t));
        id<MTLBuffer> buf_scales   = make_no_copy_buffer((void*)scales, D_out * groups_per_row * sizeof(float));
        id<MTLBuffer> buf_bias     = (has_bias && bias) ? make_no_copy_buffer((void*)bias, D_out * sizeof(float)) : nil;
        id<MTLBuffer> buf_Y        = make_no_copy_buffer((void*)Y, B * D_out * sizeof(float));

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_pipe_kan_forward_ternary_cheby];
        [enc setBuffer:buf_X offset:0 atIndex:0];
        [enc setBuffer:buf_W_packed offset:0 atIndex:1];
        [enc setBuffer:buf_scales offset:0 atIndex:2];
        [enc setBuffer:(buf_bias ? buf_bias : buf_X) offset:0 atIndex:3];
        [enc setBuffer:buf_Y offset:0 atIndex:4];

        uint uB = (uint)B, uDin = (uint)D_in, uDout = (uint)D_out, uDeg = (uint)degree;
        uint uGroup = (uint)group_size, uBias = (uint)has_bias;
        [enc setBytes:&uB length:sizeof(uint) atIndex:5];
        [enc setBytes:&uDin length:sizeof(uint) atIndex:6];
        [enc setBytes:&uDout length:sizeof(uint) atIndex:7];
        [enc setBytes:&uDeg length:sizeof(uint) atIndex:8];
        [enc setBytes:&uGroup length:sizeof(uint) atIndex:9];
        [enc setBytes:&uBias length:sizeof(uint) atIndex:10];

        [enc dispatchThreads:MTLSizeMake(D_out, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc endEncoding];

        commit_and_sync(cmd);
        return 0;
    }
}

int metal_kan_quantize_int2(
    const float* W,
    uint32_t* W_packed,
    float* scales,
    const float* codebook,
    int D_out,
    int K_dim,
    int group_size
) {
    if (!W || !W_packed || !scales || !codebook || D_out <= 0 || K_dim <= 0 || group_size <= 0) return -1;
    int words_per_row = (K_dim + 15) / 16;
    int groups_per_row = (K_dim + group_size - 1) / group_size;

    std::memset(W_packed, 0, D_out * words_per_row * sizeof(uint32_t));

    for (int j = 0; j < D_out; ++j) {
        const float* w_row = W + j * K_dim;
        uint32_t* packed_row = W_packed + j * words_per_row;
        float* s_row = scales + j * groups_per_row;

        for (int g = 0; g < groups_per_row; ++g) {
            int start_idx = g * group_size;
            int end_idx = std::min(start_idx + group_size, K_dim);

            float max_val = 0.0f;
            for (int k = start_idx; k < end_idx; ++k) {
                float a = std::fabs(w_row[k]);
                if (a > max_val) max_val = a;
            }
            float sc = (max_val > 0.0f) ? max_val : 1e-4f;
            s_row[g] = sc;
            float inv_sc = 1.0f / sc;

            for (int k = start_idx; k < end_idx; ++k) {
                float val = w_row[k] * inv_sc;
                // Nearest centroid among 4 levels
                int best_c = 0;
                float best_dist = std::fabs(val - codebook[0]);
                for (int c = 1; c < 4; ++c) {
                    float dist = std::fabs(val - codebook[c]);
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_c = c;
                    }
                }

                int word_idx = k / 16;
                int bit_shift = (k % 16) * 2;
                packed_row[word_idx] |= ((uint32_t)best_c << bit_shift);
            }
        }
    }
    return 0;
}

int metal_kan_forward_int2_cheby(
    const float* X,
    const uint32_t* W_packed,
    const float* scales,
    const float* codebook,
    const float* bias,
    float* Y,
    int B,
    int D_in,
    int D_out,
    int degree,
    int group_size,
    int has_bias
) {
    @autoreleasepool {
        if (!X || !W_packed || !scales || !codebook || !Y || B <= 0 || D_in <= 0 || D_out <= 0 || degree <= 0) return -1;
        uint K_dim = (uint)(D_in * degree);
        uint words_per_row = (K_dim + 15) / 16;
        uint groups_per_row = (K_dim + group_size - 1) / group_size;

        id<MTLBuffer> buf_X        = make_no_copy_buffer((void*)X, B * D_in * sizeof(float));
        id<MTLBuffer> buf_W_packed = make_no_copy_buffer((void*)W_packed, D_out * words_per_row * sizeof(uint32_t));
        id<MTLBuffer> buf_scales   = make_no_copy_buffer((void*)scales, D_out * groups_per_row * sizeof(float));
        id<MTLBuffer> buf_codebook = make_no_copy_buffer((void*)codebook, 4 * sizeof(float));
        id<MTLBuffer> buf_bias     = (has_bias && bias) ? make_no_copy_buffer((void*)bias, D_out * sizeof(float)) : nil;
        id<MTLBuffer> buf_Y        = make_no_copy_buffer((void*)Y, B * D_out * sizeof(float));

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_pipe_kan_forward_int2_cheby];
        [enc setBuffer:buf_X offset:0 atIndex:0];
        [enc setBuffer:buf_W_packed offset:0 atIndex:1];
        [enc setBuffer:buf_scales offset:0 atIndex:2];
        [enc setBuffer:buf_codebook offset:0 atIndex:3];
        [enc setBuffer:(buf_bias ? buf_bias : buf_X) offset:0 atIndex:4];
        [enc setBuffer:buf_Y offset:0 atIndex:5];

        uint uB = (uint)B, uDin = (uint)D_in, uDout = (uint)D_out, uDeg = (uint)degree;
        uint uGroup = (uint)group_size, uBias = (uint)has_bias;
        [enc setBytes:&uB length:sizeof(uint) atIndex:6];
        [enc setBytes:&uDin length:sizeof(uint) atIndex:7];
        [enc setBytes:&uDout length:sizeof(uint) atIndex:8];
        [enc setBytes:&uDeg length:sizeof(uint) atIndex:9];
        [enc setBytes:&uGroup length:sizeof(uint) atIndex:10];
        [enc setBytes:&uBias length:sizeof(uint) atIndex:11];

        [enc dispatchThreads:MTLSizeMake(D_out, B, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc endEncoding];

        commit_and_sync(cmd);
        return 0;
    }
}

// Element-wise and activation operations.

int metal_kan_swiglu_forward(
    const float* gate,
    const float* up,
    float* out,
    int n_elems
) {
    @autoreleasepool {
        if (!gate || !up || !out || n_elems <= 0) return -1;
        id<MTLBuffer> buf_gate = make_no_copy_buffer((void*)gate, n_elems * sizeof(float));
        id<MTLBuffer> buf_up   = make_no_copy_buffer((void*)up, n_elems * sizeof(float));
        id<MTLBuffer> buf_out  = make_no_copy_buffer((void*)out, n_elems * sizeof(float));

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_pipe_swiglu_forward];
        [enc setBuffer:buf_gate offset:0 atIndex:0];
        [enc setBuffer:buf_up offset:0 atIndex:1];
        [enc setBuffer:buf_out offset:0 atIndex:2];
        uint u_elems = (uint)n_elems;
        [enc setBytes:&u_elems length:sizeof(uint) atIndex:3];

        uint max_tg = (uint)g_pipe_swiglu_forward.maxTotalThreadsPerThreadgroup;
        uint tg_size = std::min(max_tg > 0 ? max_tg : 1024u, u_elems);
        [enc dispatchThreads:MTLSizeMake(u_elems, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];

        commit_and_sync(cmd);
        return 0;
    }
}

int metal_kan_swiglu_backward(
    const float* d_hidden,
    const float* gate,
    const float* up,
    float* d_gate,
    float* d_up,
    int n_elems
) {
    @autoreleasepool {
        if (!d_hidden || !gate || !up || !d_gate || !d_up || n_elems <= 0) return -1;
        id<MTLBuffer> buf_dh = make_no_copy_buffer((void*)d_hidden, n_elems * sizeof(float));
        id<MTLBuffer> buf_g  = make_no_copy_buffer((void*)gate, n_elems * sizeof(float));
        id<MTLBuffer> buf_u  = make_no_copy_buffer((void*)up, n_elems * sizeof(float));
        id<MTLBuffer> buf_dg = make_no_copy_buffer((void*)d_gate, n_elems * sizeof(float));
        id<MTLBuffer> buf_du = make_no_copy_buffer((void*)d_up, n_elems * sizeof(float));

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_pipe_swiglu_backward];
        [enc setBuffer:buf_dh offset:0 atIndex:0];
        [enc setBuffer:buf_g offset:0 atIndex:1];
        [enc setBuffer:buf_u offset:0 atIndex:2];
        [enc setBuffer:buf_dg offset:0 atIndex:3];
        [enc setBuffer:buf_du offset:0 atIndex:4];
        uint u_elems = (uint)n_elems;
        [enc setBytes:&u_elems length:sizeof(uint) atIndex:5];

        uint max_tg = (uint)g_pipe_swiglu_backward.maxTotalThreadsPerThreadgroup;
        uint tg_size = std::min(max_tg > 0 ? max_tg : 1024u, u_elems);
        [enc dispatchThreads:MTLSizeMake(u_elems, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];

        commit_and_sync(cmd);
        return 0;
    }
}

int metal_kan_elementwise_add(
    const float* a,
    const float* b,
    float* out,
    int n_elems
) {
    @autoreleasepool {
        if (!a || !b || !out || n_elems <= 0) return -1;
        id<MTLBuffer> buf_a   = make_no_copy_buffer((void*)a, n_elems * sizeof(float));
        id<MTLBuffer> buf_b   = make_no_copy_buffer((void*)b, n_elems * sizeof(float));
        id<MTLBuffer> buf_out = make_no_copy_buffer((void*)out, n_elems * sizeof(float));

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_pipe_elementwise_add];
        [enc setBuffer:buf_a offset:0 atIndex:0];
        [enc setBuffer:buf_b offset:0 atIndex:1];
        [enc setBuffer:buf_out offset:0 atIndex:2];
        uint u_elems = (uint)n_elems;
        [enc setBytes:&u_elems length:sizeof(uint) atIndex:3];

        uint max_tg = (uint)g_pipe_elementwise_add.maxTotalThreadsPerThreadgroup;
        uint tg_size = std::min(max_tg > 0 ? max_tg : 1024u, u_elems);
        [enc dispatchThreads:MTLSizeMake(u_elems, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];

        commit_and_sync(cmd);
        return 0;
    }
}

int metal_kan_adam_step(
    float* param,
    const float* grad,
    float* m,
    float* v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float lr_t,
    int n_elems
) {
    @autoreleasepool {
        if (!param || !grad || !m || !v || n_elems <= 0) return -1;
        id<MTLBuffer> buf_param = make_no_copy_buffer((void*)param, n_elems * sizeof(float));
        id<MTLBuffer> buf_grad  = make_no_copy_buffer((void*)grad, n_elems * sizeof(float));
        id<MTLBuffer> buf_m     = make_no_copy_buffer((void*)m, n_elems * sizeof(float));
        id<MTLBuffer> buf_v     = make_no_copy_buffer((void*)v, n_elems * sizeof(float));

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_pipe_adam_coupled_step];
        [enc setBuffer:buf_param offset:0 atIndex:0];
        [enc setBuffer:buf_grad offset:0 atIndex:1];
        [enc setBuffer:buf_m offset:0 atIndex:2];
        [enc setBuffer:buf_v offset:0 atIndex:3];
        [enc setBytes:&lr length:sizeof(float) atIndex:4];
        [enc setBytes:&beta1 length:sizeof(float) atIndex:5];
        [enc setBytes:&beta2 length:sizeof(float) atIndex:6];
        [enc setBytes:&eps length:sizeof(float) atIndex:7];
        [enc setBytes:&weight_decay length:sizeof(float) atIndex:8];
        [enc setBytes:&lr_t length:sizeof(float) atIndex:9];
        uint u_elems = (uint)n_elems;
        [enc setBytes:&u_elems length:sizeof(uint) atIndex:10];

        uint max_tg = (uint)g_pipe_adam_coupled_step.maxTotalThreadsPerThreadgroup;
        uint tg_size = std::min(max_tg > 0 ? max_tg : 1024u, u_elems);
        [enc dispatchThreads:MTLSizeMake(u_elems, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];

        commit_and_sync(cmd);
        return 0;
    }
}

int metal_kan_muon_step(
    float* param,
    const float* grad,
    float* v,
    float lr,
    float momentum,
    float weight_decay,
    int nesterov,
    int ns_steps,
    int rows,
    int cols
) {
    @autoreleasepool {
        int n_elems = rows * cols;
        if (!param || !grad || !v || n_elems <= 0) return -1;

        id<MTLBuffer> buf_param  = make_no_copy_buffer((void*)param, n_elems * sizeof(float));
        id<MTLBuffer> buf_grad   = make_no_copy_buffer((void*)grad, n_elems * sizeof(float));
        id<MTLBuffer> buf_v      = make_no_copy_buffer((void*)v, n_elems * sizeof(float));

        static id<MTLBuffer> s_muon_update = nil;
        static size_t s_muon_update_cap = 0;
        static id<MTLBuffer> s_muon_ortho = nil;
        static size_t s_muon_ortho_cap = 0;

        size_t bytes = n_elems * sizeof(float);
        if (bytes > s_muon_update_cap) {
            if (s_muon_update) [s_muon_update release];
            s_muon_update_cap = bytes * 2;
            s_muon_update = [[g_device newBufferWithLength:s_muon_update_cap options:MTLResourceStorageModeShared] retain];
        }
        if (bytes > s_muon_ortho_cap) {
            if (s_muon_ortho) [s_muon_ortho release];
            s_muon_ortho_cap = bytes * 2;
            s_muon_ortho = [[g_device newBufferWithLength:s_muon_ortho_cap options:MTLResourceStorageModeShared] retain];
        }

        // Momentum and Nesterov update
        id<MTLCommandBuffer> cmd1 = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc1 = [cmd1 computeCommandEncoder];
        [enc1 setComputePipelineState:g_pipe_muon_momentum_update];
        [enc1 setBuffer:buf_param offset:0 atIndex:0];
        [enc1 setBuffer:buf_grad offset:0 atIndex:1];
        [enc1 setBuffer:buf_v offset:0 atIndex:2];
        [enc1 setBuffer:s_muon_update offset:0 atIndex:3];
        [enc1 setBytes:&momentum length:sizeof(float) atIndex:4];
        [enc1 setBytes:&weight_decay length:sizeof(float) atIndex:5];
        uint u_nest = (uint)nesterov;
        [enc1 setBytes:&u_nest length:sizeof(uint) atIndex:6];
        uint u_elems = (uint)n_elems;
        [enc1 setBytes:&u_elems length:sizeof(uint) atIndex:7];

        uint max_tg = (uint)g_pipe_muon_momentum_update.maxTotalThreadsPerThreadgroup;
        uint tg_size = std::min(max_tg > 0 ? max_tg : 1024u, u_elems);
        [enc1 dispatchThreads:MTLSizeMake(u_elems, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc1 endEncoding];
        commit_and_sync(cmd1);

        // Newton-Schulz 5 orthogonalization
        float* update_ptr = (float*)[s_muon_update contents];
        float* ortho_ptr = (float*)[s_muon_ortho contents];
        metal_kan_newton_schulz5_gpu(update_ptr, ortho_ptr, rows, cols, ns_steps, 1e-7f);

        // Aspect ratio scaling and parameter update
        float aspect = std::sqrt(std::max(1.0f, (float)rows / (float)cols));
        float step_lr = lr * aspect;

        id<MTLCommandBuffer> cmd2 = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc2 = [cmd2 computeCommandEncoder];
        [enc2 setComputePipelineState:g_pipe_muon_param_update];
        [enc2 setBuffer:buf_param offset:0 atIndex:0];
        [enc2 setBuffer:s_muon_ortho offset:0 atIndex:1];
        [enc2 setBytes:&step_lr length:sizeof(float) atIndex:2];
        [enc2 setBytes:&u_elems length:sizeof(uint) atIndex:3];
        [enc2 dispatchThreads:MTLSizeMake(u_elems, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc2 endEncoding];
        commit_and_sync(cmd2);

        return 0;
    }
}

int metal_kan_dequantize_ternary(
    const uint32_t* packed,
    const float* scales,
    float* out,
    int D_out,
    int K_dim,
    int group_size
) {
    @autoreleasepool {
        if (!packed || !scales || !out || D_out <= 0 || K_dim <= 0 || group_size <= 0) return -1;
        int words_per_row = (K_dim + 15) / 16;
        int groups_per_row = (K_dim + group_size - 1) / group_size;

        id<MTLBuffer> buf_packed = make_no_copy_buffer((void*)packed, D_out * words_per_row * sizeof(uint32_t));
        id<MTLBuffer> buf_scales = make_no_copy_buffer((void*)scales, D_out * groups_per_row * sizeof(float));
        id<MTLBuffer> buf_out    = make_no_copy_buffer((void*)out, D_out * K_dim * sizeof(float));

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_pipe_dequantize_ternary_fast];
        [enc setBuffer:buf_packed offset:0 atIndex:0];
        [enc setBuffer:buf_scales offset:0 atIndex:1];
        [enc setBuffer:buf_out offset:0 atIndex:2];
        uint uDout = (uint)D_out, uKdim = (uint)K_dim, uGroup = (uint)group_size;
        [enc setBytes:&uDout length:sizeof(uint) atIndex:3];
        [enc setBytes:&uKdim length:sizeof(uint) atIndex:4];
        [enc setBytes:&uGroup length:sizeof(uint) atIndex:5];

        [enc dispatchThreads:MTLSizeMake(K_dim, D_out, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc endEncoding];

        commit_and_sync(cmd);
        return 0;
    }
}

int metal_kan_dequantize_int2(
    const uint32_t* packed,
    const float* scales,
    const float* codebook,
    float* out,
    int D_out,
    int K_dim,
    int group_size
) {
    @autoreleasepool {
        if (!packed || !scales || !codebook || !out || D_out <= 0 || K_dim <= 0 || group_size <= 0) return -1;
        int words_per_row = (K_dim + 15) / 16;
        int groups_per_row = (K_dim + group_size - 1) / group_size;

        id<MTLBuffer> buf_packed   = make_no_copy_buffer((void*)packed, D_out * words_per_row * sizeof(uint32_t));
        id<MTLBuffer> buf_scales   = make_no_copy_buffer((void*)scales, D_out * groups_per_row * sizeof(float));
        id<MTLBuffer> buf_codebook = make_no_copy_buffer((void*)codebook, 4 * sizeof(float));
        id<MTLBuffer> buf_out      = make_no_copy_buffer((void*)out, D_out * K_dim * sizeof(float));

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_pipe_dequantize_int2_fast];
        [enc setBuffer:buf_packed offset:0 atIndex:0];
        [enc setBuffer:buf_scales offset:0 atIndex:1];
        [enc setBuffer:buf_codebook offset:0 atIndex:2];
        [enc setBuffer:buf_out offset:0 atIndex:3];
        uint uDout = (uint)D_out, uKdim = (uint)K_dim, uGroup = (uint)group_size;
        [enc setBytes:&uDout length:sizeof(uint) atIndex:4];
        [enc setBytes:&uKdim length:sizeof(uint) atIndex:5];
        [enc setBytes:&uGroup length:sizeof(uint) atIndex:6];

        [enc dispatchThreads:MTLSizeMake(K_dim, D_out, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc endEncoding];

        commit_and_sync(cmd);
        return 0;
    }
}

int metal_kan_calc_mse_loss_backward(
    const float* pred,
    const float* target,
    float* dY,
    int n_elems,
    float scale
) {
    @autoreleasepool {
        if (!pred || !target || !dY || n_elems <= 0) return -1;
        id<MTLBuffer> buf_pred   = make_no_copy_buffer((void*)pred, n_elems * sizeof(float));
        id<MTLBuffer> buf_target = make_no_copy_buffer((void*)target, n_elems * sizeof(float));
        id<MTLBuffer> buf_dY     = make_no_copy_buffer((void*)dY, n_elems * sizeof(float));

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_pipe_calc_mse_loss_backward];
        [enc setBuffer:buf_pred offset:0 atIndex:0];
        [enc setBuffer:buf_target offset:0 atIndex:1];
        [enc setBuffer:buf_dY offset:0 atIndex:2];
        uint u_elems = (uint)n_elems;
        [enc setBytes:&u_elems length:sizeof(uint) atIndex:3];
        [enc setBytes:&scale length:sizeof(float) atIndex:4];

        uint max_tg = (uint)g_pipe_calc_mse_loss_backward.maxTotalThreadsPerThreadgroup;
        uint tg_size = std::min(max_tg > 0 ? max_tg : 1024u, u_elems);
        [enc dispatchThreads:MTLSizeMake(u_elems, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];

        commit_and_sync(cmd);
        return 0;
    }
}

} // extern "C"





