#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

using std::cout;
using std::generate;
using std::vector;


// Naive kernel (reference)

__global__ void matrixMulXrow(const float* a, const float* b, float* c, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= N || col >= N) return;
    float tmp = 0;
    for (int k = 0; k < N; k++) tmp += a[row * N + k] * b[k * N + col];
    c[row * N + col] = tmp;
}

int main() {
    int N = 8192;
    size_t bytes = N * N * sizeof(float);
    float Nbr_GFLOPS = 2.0f * N / 1000.0f * N / 1000.0f * N / 1000.0f;

    vector<float> h_a(N * N), h_b(N * N), h_c(N * N);
    generate(h_a.begin(), h_a.end(), []() { return (float)(rand() % 100); });
    generate(h_b.begin(), h_b.end(), []() { return (float)(rand() % 100); });

    float* d_a, * d_b, * d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    cudaEvent_t e1, e2, e3, e4;
    cudaEventCreate(&e1);
    cudaEventCreate(&e2);
    cudaEventCreate(&e3);
    cudaEventCreate(&e4);

    float Naive_time, cuBLAS_time;

    
    // Test 1 : Naive kernel (matrixMulXrow)
    
    cout << "\n--- Naive Kernel (matrixMulXrow) ---\n";

    int THREADS = 32;
    int BLOCKS = N / THREADS;
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    cudaEventRecord(e1);
    matrixMulXrow << <blocks, threads >> > (d_a, d_b, d_c, N);
    cudaEventRecord(e2);
    cudaEventSynchronize(e2);
    cudaEventElapsedTime(&Naive_time, e1, e2);

    printf("Naive Kernel Time: %f ms\n", Naive_time);
    printf("Naive Performance: %f GFLOPS\n", Nbr_GFLOPS * 1000.0f / Naive_time);

    
    // Test 2 : cuBLAS
    
    cout << "\n--- cuBLAS ---\n";

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    // Warm up
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        N, N, N,
        &alpha,
        d_b, N,
        d_a, N,
        &beta,
        d_c, N);
    cudaDeviceSynchronize();

    cudaEventRecord(e3);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        N, N, N,
        &alpha,
        d_b, N,
        d_a, N,
        &beta,
        d_c, N);
    cudaEventRecord(e4);
    cudaEventSynchronize(e4);
    cudaEventElapsedTime(&cuBLAS_time, e3, e4);

    printf("cuBLAS Time: %f ms\n", cuBLAS_time);
    printf("cuBLAS Performance: %f GFLOPS\n", Nbr_GFLOPS * 1000.0f / cuBLAS_time);

    
    // Comparison
   
    cout << "\n--- Comparison ---\n";
    printf("Speedup cuBLAS vs Naive: %f x\n", Naive_time / cuBLAS_time);

    cublasDestroy(handle);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    char kml;
    scanf("%c", &kml);
    return 0;
}