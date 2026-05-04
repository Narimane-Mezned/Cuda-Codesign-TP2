#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define INPUT_SIZE  67108864
#define HIDDEN_SIZE 8192
#define BLOCK_SIZE  256
#define REPEATS     100


// KERNEL 1 : Naive reduction

__global__ void sumReductionNaive(float* input, float* output, int n) {
    __shared__ float sharedData[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sharedData[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = 1.0f / (1.0f + expf(-sharedData[0])); // sigmoid
}


// KERNEL 2 : Sequential addressing (no warp divergence)

__global__ void sumReductionSequential(float* input, float* output, int n) {
    __shared__ float sharedData[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sharedData[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // Sequential addressing : threads actifs dans la moitie basse
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = 1.0f / (1.0f + expf(-sharedData[0])); // sigmoid
}


// KERNEL 3 : Sequential + unroll last warp

__device__ void warpReduce(volatile float* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void sumReductionUnroll(float* input, float* output, int n) {
    __shared__ float sharedData[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sharedData[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // Unroll last warp (no __syncthreads needed)
    if (tid < 32) warpReduce(sharedData, tid);

    if (tid == 0)
        output[blockIdx.x] = 1.0f / (1.0f + expf(-sharedData[0])); // sigmoid
}


// KERNEL 4 : Warp shuffle reduction

__global__ void sumReductionShuffle(float* input, float* output, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (idx < n) ? input[idx] : 0.0f;

    // Warp shuffle reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);

    __shared__ float sharedData[BLOCK_SIZE / 32];

    if (tid % 32 == 0)
        sharedData[tid / 32] = val;
    __syncthreads();

    // Final reduction across warps
    if (tid < BLOCK_SIZE / 32) {
        val = sharedData[tid];
        for (int offset = (BLOCK_SIZE / 32) / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if (tid == 0)
        output[blockIdx.x] = 1.0f / (1.0f + expf(-val)); // sigmoid
}


// Reduction finale sur GPU (sortie)

__global__ void finalReduction(float* input, float* output, int n) {
    __shared__ float sharedData[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sharedData[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sharedData[tid] += sharedData[tid + stride];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sharedData[0];
}


// Fonction de mesure

float measureKernel(void (*kernel)(float*, float*, int),
    float* d_input, float* d_hidden, float* d_output,
    int input_size, int hidden_size, const char* name) {

    int grid_hidden = (input_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_output = (hidden_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm up
    kernel << <grid_hidden, BLOCK_SIZE >> > (d_input, d_hidden, input_size);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < REPEATS; i++) {
        kernel << <grid_hidden, BLOCK_SIZE >> > (d_input, d_hidden, input_size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);
    time_ms /= REPEATS;

    printf("%s Time: %f ms\n", name, time_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return time_ms;
}


// MAIN

int main() {
    int input_size = INPUT_SIZE;
    int hidden_size = HIDDEN_SIZE;

    // Allocation host
    float* h_input = (float*)malloc(input_size * sizeof(float));
    float* h_hidden = (float*)malloc(hidden_size * sizeof(float));

    // Init : x[i] = 1.0, w[i][j] = 1.0 donc input GPU = x*w = 1.0
    for (int i = 0; i < input_size; i++)
        h_input[i] = 1.0f;

    // Allocation GPU
    float* d_input, * d_hidden, * d_output_final;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_hidden, hidden_size * sizeof(float));
    cudaMalloc(&d_output_final, BLOCK_SIZE * sizeof(float));

    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

    printf("=== Neural Network Sum Reduction ===\n");
    printf("Input size:  %d\n", input_size);
    printf("Hidden size: %d\n", hidden_size);
    printf("Block size:  %d\n\n", BLOCK_SIZE);

    float naive_time, seq_time, unroll_time, shuffle_time;

    // Kernel 1 : Naive
    naive_time = measureKernel(sumReductionNaive, d_input, d_hidden, d_output_final, input_size, hidden_size, "Naive");

    // Kernel 2 : Sequential
    seq_time = measureKernel(sumReductionSequential, d_input, d_hidden, d_output_final, input_size, hidden_size, "Sequential");

    // Kernel 3 : Unroll
    unroll_time = measureKernel(sumReductionUnroll, d_input, d_hidden, d_output_final, input_size, hidden_size, "Unroll last warp");

    // Kernel 4 : Shuffle
    shuffle_time = measureKernel(sumReductionShuffle, d_input, d_hidden, d_output_final, input_size, hidden_size, "Warp Shuffle");

    printf("\n=== Speedup vs Naive ===\n");
    printf("Sequential:     %.2f x\n", naive_time / seq_time);
    printf("Unroll:         %.2f x\n", naive_time / unroll_time);
    printf("Warp Shuffle:   %.2f x\n", naive_time / shuffle_time);

    free(h_input);
    free(h_hidden);
    cudaFree(d_input);
    cudaFree(d_hidden);
    cudaFree(d_output_final);

    char k;
    scanf("%c", &k);
    return 0;
}