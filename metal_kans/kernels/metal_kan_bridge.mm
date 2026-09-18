#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
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

static double g_timebase_factor = 0.0;

static void init_timebase() {
    if (g_timebase_factor == 0.0) {
        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);
        g_timebase_factor = (double)tb.numer / (double)tb.denom * 1e-9;
    }
}

extern "C" {

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

        if (!g_pipe_cheby_tiled || !g_pipe_fastkan_tiled || !g_pipe_relu_tiled ||
            !g_pipe_wavkan_tiled || !g_pipe_fourier_tiled || !g_pipe_jacobi_tiled ||
            !g_pipe_rational_tiled || !g_pipe_bspline_tiled) {
            std::cerr << "[MetalKAN] Failed to create one or more compute pipelines!" << std::endl;
            return -4;
        }

        return 0;
    }
}

static id<MTLBuffer> make_no_copy_buffer(void* ptr, size_t bytes) {
    return [g_device newBufferWithBytesNoCopy:ptr
                                      length:bytes
                                     options:MTLResourceStorageModeShared
                                 deallocator:nil];
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
        id<MTLComputePipelineState> pipe = g_pipe_cheby_tiled;
        if (!pipe) return -1;

        id<MTLBuffer> buf_X       = make_no_copy_buffer((void*)X, B * D_in * sizeof(float));
        id<MTLBuffer> buf_W_cheby = make_no_copy_buffer((void*)W_cheby, D_out * D_in * 4 * sizeof(float));
        id<MTLBuffer> buf_W_base  = has_base ? make_no_copy_buffer((void*)W_base, D_out * D_in * sizeof(float)) : buf_X;
        id<MTLBuffer> buf_bias    = has_bias ? make_no_copy_buffer((void*)bias, D_out * sizeof(float)) : buf_X;
        id<MTLBuffer> buf_Y       = make_no_copy_buffer((void*)Y, B * D_out * sizeof(float));

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipe];

        [enc setBuffer:buf_X offset:0 atIndex:0];
        [enc setBuffer:buf_W_cheby offset:0 atIndex:1];
        [enc setBuffer:buf_W_base offset:0 atIndex:2];
        [enc setBuffer:buf_bias offset:0 atIndex:3];
        [enc setBuffer:buf_Y offset:0 atIndex:4];

        uint uB = (uint)B;
        uint uD_in = (uint)D_in;
        uint uD_out = (uint)D_out;
        uint u_has_base = (uint)has_base;
        uint u_has_bias = (uint)has_bias;

        [enc setBytes:&uB length:sizeof(uint) atIndex:5];
        [enc setBytes:&uD_in length:sizeof(uint) atIndex:6];
        [enc setBytes:&uD_out length:sizeof(uint) atIndex:7];
        [enc setBytes:&u_has_base length:sizeof(uint) atIndex:8];
        [enc setBytes:&u_has_bias length:sizeof(uint) atIndex:9];

        MTLSize grid = MTLSizeMake((D_out + 31) / 32, (B + 31) / 32, 1);
        MTLSize tg   = MTLSizeMake(16, 16, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];

        [enc endEncoding];
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
        id<MTLComputePipelineState> pipe = g_pipe_fastkan_tiled;
        if (!pipe) return -1;

        id<MTLBuffer> buf_X      = make_no_copy_buffer((void*)X, B * D_in * sizeof(float));
        id<MTLBuffer> buf_W_rbf  = make_no_copy_buffer((void*)W_rbf, D_out * D_in * num_centers * sizeof(float));
        id<MTLBuffer> buf_W_base = has_base ? make_no_copy_buffer((void*)W_base, D_out * D_in * sizeof(float)) : buf_X;
        id<MTLBuffer> buf_grid   = make_no_copy_buffer((void*)grid, num_centers * sizeof(float));
        id<MTLBuffer> buf_bias   = has_bias ? make_no_copy_buffer((void*)bias, D_out * sizeof(float)) : buf_X;
        id<MTLBuffer> buf_Y      = make_no_copy_buffer((void*)Y, B * D_out * sizeof(float));

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipe];

        [enc setBuffer:buf_X offset:0 atIndex:0];
        [enc setBuffer:buf_W_rbf offset:0 atIndex:1];
        [enc setBuffer:buf_W_base offset:0 atIndex:2];
        [enc setBuffer:buf_grid offset:0 atIndex:3];
        [enc setBuffer:buf_bias offset:0 atIndex:4];
        [enc setBuffer:buf_Y offset:0 atIndex:5];

        uint uB = (uint)B;
        uint uD_in = (uint)D_in;
        uint uD_out = (uint)D_out;
        uint u_centers = (uint)num_centers;
        float u_inv_denom = inv_denominator;
        uint u_has_base = (uint)has_base;
        uint u_has_bias = (uint)has_bias;

        [enc setBytes:&uB length:sizeof(uint) atIndex:6];
        [enc setBytes:&uD_in length:sizeof(uint) atIndex:7];
        [enc setBytes:&uD_out length:sizeof(uint) atIndex:8];
        [enc setBytes:&u_centers length:sizeof(uint) atIndex:9];
        [enc setBytes:&u_inv_denom length:sizeof(float) atIndex:10];
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
        id<MTLComputePipelineState> pipe = g_pipe_relu_tiled;
        if (!pipe) return -1;

        id<MTLBuffer> buf_X      = make_no_copy_buffer((void*)X, B * D_in * sizeof(float));
        id<MTLBuffer> buf_W_relu = make_no_copy_buffer((void*)W_relu, D_out * D_in * num_grids * sizeof(float));
        id<MTLBuffer> buf_W_base = has_base ? make_no_copy_buffer((void*)W_base, D_out * D_in * sizeof(float)) : buf_X;
        id<MTLBuffer> buf_grid   = make_no_copy_buffer((void*)grid, num_grids * sizeof(float));
        id<MTLBuffer> buf_bias   = has_bias ? make_no_copy_buffer((void*)bias, D_out * sizeof(float)) : buf_X;
        id<MTLBuffer> buf_Y      = make_no_copy_buffer((void*)Y, B * D_out * sizeof(float));

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipe];

        [enc setBuffer:buf_X offset:0 atIndex:0];
        [enc setBuffer:buf_W_relu offset:0 atIndex:1];
        [enc setBuffer:buf_W_base offset:0 atIndex:2];
        [enc setBuffer:buf_grid offset:0 atIndex:3];
        [enc setBuffer:buf_bias offset:0 atIndex:4];
        [enc setBuffer:buf_Y offset:0 atIndex:5];

        uint uB = (uint)B;
        uint uD_in = (uint)D_in;
        uint uD_out = (uint)D_out;
        uint u_num_grids = (uint)num_grids;
        float u_inv_h = inv_h;
        uint u_has_base = (uint)has_base;
        uint u_has_bias = (uint)has_bias;

        [enc setBytes:&uB length:sizeof(uint) atIndex:6];
        [enc setBytes:&uD_in length:sizeof(uint) atIndex:7];
        [enc setBytes:&uD_out length:sizeof(uint) atIndex:8];
        [enc setBytes:&u_num_grids length:sizeof(uint) atIndex:9];
        [enc setBytes:&u_inv_h length:sizeof(float) atIndex:10];
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

int metal_kan_wavkan_forward(
    const float* X,
    const float* W_wav,
    const float* W_base,
    const float* translation,
    const float* scale,
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
        id<MTLComputePipelineState> pipe = g_pipe_wavkan_tiled;
        if (!pipe) return -1;

        id<MTLBuffer> buf_X       = make_no_copy_buffer((void*)X, B * D_in * sizeof(float));
        id<MTLBuffer> buf_W_wav   = make_no_copy_buffer((void*)W_wav, D_out * D_in * num_wavelets * sizeof(float));
        id<MTLBuffer> buf_W_base  = has_base ? make_no_copy_buffer((void*)W_base, D_out * D_in * sizeof(float)) : buf_X;
        id<MTLBuffer> buf_trans   = make_no_copy_buffer((void*)translation, D_in * num_wavelets * sizeof(float));
        id<MTLBuffer> buf_scale   = make_no_copy_buffer((void*)scale, D_in * num_wavelets * sizeof(float));
        id<MTLBuffer> buf_bias    = has_bias ? make_no_copy_buffer((void*)bias, D_out * sizeof(float)) : buf_X;
        id<MTLBuffer> buf_Y       = make_no_copy_buffer((void*)Y, B * D_out * sizeof(float));

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipe];

        [enc setBuffer:buf_X offset:0 atIndex:0];
        [enc setBuffer:buf_W_wav offset:0 atIndex:1];
        [enc setBuffer:buf_W_base offset:0 atIndex:2];
        [enc setBuffer:buf_trans offset:0 atIndex:3];
        [enc setBuffer:buf_scale offset:0 atIndex:4];
        [enc setBuffer:buf_bias offset:0 atIndex:5];
        [enc setBuffer:buf_Y offset:0 atIndex:6];

        uint uB = (uint)B;
        uint uD_in = (uint)D_in;
        uint uD_out = (uint)D_out;
        uint u_wavelets = (uint)num_wavelets;
        uint u_type = (uint)wavelet_type;
        uint u_has_base = (uint)has_base;
        uint u_has_bias = (uint)has_bias;

        [enc setBytes:&uB length:sizeof(uint) atIndex:7];
        [enc setBytes:&uD_in length:sizeof(uint) atIndex:8];
        [enc setBytes:&uD_out length:sizeof(uint) atIndex:9];
        [enc setBytes:&u_wavelets length:sizeof(uint) atIndex:10];
        [enc setBytes:&u_type length:sizeof(uint) atIndex:11];
        [enc setBytes:&u_has_base length:sizeof(uint) atIndex:12];
        [enc setBytes:&u_has_bias length:sizeof(uint) atIndex:13];

        MTLSize grid_sz = MTLSizeMake((D_out + 31) / 32, (B + 31) / 32, 1);
        MTLSize tg   = MTLSizeMake(16, 16, 1);
        [enc dispatchThreadgroups:grid_sz threadsPerThreadgroup:tg];

        [enc endEncoding];
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
        id<MTLComputePipelineState> pipe = g_pipe_fourier_tiled;
        if (!pipe) return -1;

        uint num_bases = 2 * num_freqs + 1;
        id<MTLBuffer> buf_X       = make_no_copy_buffer((void*)X, B * D_in * sizeof(float));
        id<MTLBuffer> buf_W_four  = make_no_copy_buffer((void*)W_fourier, D_out * D_in * num_bases * sizeof(float));
        id<MTLBuffer> buf_W_base  = has_base ? make_no_copy_buffer((void*)W_base, D_out * D_in * sizeof(float)) : buf_X;
        id<MTLBuffer> buf_bias    = has_bias ? make_no_copy_buffer((void*)bias, D_out * sizeof(float)) : buf_X;
        id<MTLBuffer> buf_Y       = make_no_copy_buffer((void*)Y, B * D_out * sizeof(float));

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipe];

        [enc setBuffer:buf_X offset:0 atIndex:0];
        [enc setBuffer:buf_W_four offset:0 atIndex:1];
        [enc setBuffer:buf_W_base offset:0 atIndex:2];
        [enc setBuffer:buf_bias offset:0 atIndex:3];
        [enc setBuffer:buf_Y offset:0 atIndex:4];

        uint uB = (uint)B;
        uint uD_in = (uint)D_in;
        uint uD_out = (uint)D_out;
        uint u_freqs = (uint)num_freqs;
        uint u_has_base = (uint)has_base;
        uint u_has_bias = (uint)has_bias;

        [enc setBytes:&uB length:sizeof(uint) atIndex:5];
        [enc setBytes:&uD_in length:sizeof(uint) atIndex:6];
        [enc setBytes:&uD_out length:sizeof(uint) atIndex:7];
        [enc setBytes:&u_freqs length:sizeof(uint) atIndex:8];
        [enc setBytes:&u_has_base length:sizeof(uint) atIndex:9];
        [enc setBytes:&u_has_bias length:sizeof(uint) atIndex:10];

        MTLSize grid_sz = MTLSizeMake((D_out + 31) / 32, (B + 31) / 32, 1);
        MTLSize tg   = MTLSizeMake(16, 16, 1);
        [enc dispatchThreadgroups:grid_sz threadsPerThreadgroup:tg];

        [enc endEncoding];
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
        id<MTLComputePipelineState> pipe = g_pipe_jacobi_tiled;
        if (!pipe) return -1;

        id<MTLBuffer> buf_X      = make_no_copy_buffer((void*)X, B * D_in * sizeof(float));
        id<MTLBuffer> buf_W_jac  = make_no_copy_buffer((void*)W_jacobi, D_out * D_in * degree * sizeof(float));
        id<MTLBuffer> buf_W_base = has_base ? make_no_copy_buffer((void*)W_base, D_out * D_in * sizeof(float)) : buf_X;
        id<MTLBuffer> buf_bias   = has_bias ? make_no_copy_buffer((void*)bias, D_out * sizeof(float)) : buf_X;
        id<MTLBuffer> buf_Y      = make_no_copy_buffer((void*)Y, B * D_out * sizeof(float));

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipe];

        [enc setBuffer:buf_X offset:0 atIndex:0];
        [enc setBuffer:buf_W_jac offset:0 atIndex:1];
        [enc setBuffer:buf_W_base offset:0 atIndex:2];
        [enc setBuffer:buf_bias offset:0 atIndex:3];
        [enc setBuffer:buf_Y offset:0 atIndex:4];

        uint uB = (uint)B;
        uint uD_in = (uint)D_in;
        uint uD_out = (uint)D_out;
        uint u_deg = (uint)degree;
        float u_alpha = alpha;
        float u_beta = beta;
        uint u_has_base = (uint)has_base;
        uint u_has_bias = (uint)has_bias;

        [enc setBytes:&uB length:sizeof(uint) atIndex:5];
        [enc setBytes:&uD_in length:sizeof(uint) atIndex:6];
        [enc setBytes:&uD_out length:sizeof(uint) atIndex:7];
        [enc setBytes:&u_deg length:sizeof(uint) atIndex:8];
        [enc setBytes:&u_alpha length:sizeof(float) atIndex:9];
        [enc setBytes:&u_beta length:sizeof(float) atIndex:10];
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
        id<MTLComputePipelineState> pipe = g_pipe_bspline_tiled;
        if (!pipe) return -1;

        uint num_bases = grid_size + spline_order;
        uint num_knots = grid_size + 2 * spline_order + 1;

        id<MTLBuffer> buf_X      = make_no_copy_buffer((void*)X, B * D_in * sizeof(float));
        id<MTLBuffer> buf_W_spl  = make_no_copy_buffer((void*)W_spline, D_out * D_in * num_bases * sizeof(float));
        id<MTLBuffer> buf_W_base = has_base ? make_no_copy_buffer((void*)W_base, D_out * D_in * sizeof(float)) : buf_X;
        id<MTLBuffer> buf_grid   = make_no_copy_buffer((void*)grid, D_in * num_knots * sizeof(float));
        id<MTLBuffer> buf_bias   = has_bias ? make_no_copy_buffer((void*)bias, D_out * sizeof(float)) : buf_X;
        id<MTLBuffer> buf_Y      = make_no_copy_buffer((void*)Y, B * D_out * sizeof(float));

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipe];

        [enc setBuffer:buf_X offset:0 atIndex:0];
        [enc setBuffer:buf_W_spl offset:0 atIndex:1];
        [enc setBuffer:buf_W_base offset:0 atIndex:2];
        [enc setBuffer:buf_grid offset:0 atIndex:3];
        [enc setBuffer:buf_bias offset:0 atIndex:4];
        [enc setBuffer:buf_Y offset:0 atIndex:5];

        uint uB = (uint)B;
        uint uD_in = (uint)D_in;
        uint uD_out = (uint)D_out;
        uint u_gsize = (uint)grid_size;
        uint u_sorder = (uint)spline_order;
        uint u_has_base = (uint)has_base;
        uint u_has_bias = (uint)has_bias;

        [enc setBytes:&uB length:sizeof(uint) atIndex:6];
        [enc setBytes:&uD_in length:sizeof(uint) atIndex:7];
        [enc setBytes:&uD_out length:sizeof(uint) atIndex:8];
        [enc setBytes:&u_gsize length:sizeof(uint) atIndex:9];
        [enc setBytes:&u_sorder length:sizeof(uint) atIndex:10];
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

} // extern "C"
