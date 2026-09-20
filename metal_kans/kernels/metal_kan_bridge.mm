#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <mach/mach_time.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

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

// ==============================================================================
// DIRECT SIMDGROUP MATRIX FP32 GEMM EXPORT
// ==============================================================================

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

// ==============================================================================
// FP16 HALF-PRECISION FORWARD KERNELS
// ==============================================================================

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

} // extern "C"

