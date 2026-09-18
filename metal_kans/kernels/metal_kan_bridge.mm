#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLComputePipelineState> g_pipe_cheby_tiled = nil;
static id<MTLComputePipelineState> g_pipe_fastkan_tiled = nil;
static id<MTLComputePipelineState> g_pipe_relu_tiled = nil;

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

        g_pipe_cheby_tiled   = make_pipe(@"kan_cheby_tiled_deg4");
        g_pipe_fastkan_tiled = make_pipe(@"kan_fastkan_tiled");
        g_pipe_relu_tiled    = make_pipe(@"kan_relu_tiled");

        if (!g_pipe_cheby_tiled || !g_pipe_fastkan_tiled || !g_pipe_relu_tiled) {
            std::cerr << "[MetalKAN] Failed to create compute pipelines!" << std::endl;
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

        // Tiled dispatch: TILE_M = 32, TILE_N = 32
        MTLSize grid = MTLSizeMake((D_out + 31) / 32, (B + 31) / 32, 1);
        MTLSize tg   = MTLSizeMake(16, 16, 1); // 256 threads
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
    int warmup_iters,
    int bench_iters
) {
    for (int i = 0; i < warmup_iters; i++) {
        metal_kan_cheby_forward(X, W_cheby, W_base, bias, Y, B, D_in, D_out, K, has_base, has_bias);
    }

    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < bench_iters; i++) {
        metal_kan_cheby_forward(X, W_cheby, W_base, bias, Y, B, D_in, D_out, K, has_base, has_bias);
    }
    uint64_t t1 = mach_absolute_time();

    double total_sec = (double)(t1 - t0) * g_timebase_factor;
    return (total_sec / (double)bench_iters) * 1000.0;
}

} // extern "C"
