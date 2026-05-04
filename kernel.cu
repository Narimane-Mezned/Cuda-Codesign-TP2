#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include <mma.h>

using namespace nvcuda;
using std::cout;
using std::generate;
using std::vector;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16


// Naive kernel (reference)

__global__ void matrixMulXrow(const float* a, const float* b, float* c, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= N || col >= N) return;
    float tmp = 0;
    for (int k = 0; k < N; k++) tmp += a[row * N + k] * b[k * N + col];
    c[row * N + col] = tmp;
}


// Tensor Core kernel (WMMA)

__global__ void matrixMulTensorCore(const half* a, const half* b, float* c, int N) {
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    if (warpM * WMMA_M >= N || warpN * WMMA_N >= N) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < N; k += WMMA_K) {
        wmma::load_matrix_sync(a_frag, a + warpM * WMMA_M * N + k, N);
        wmma::load_matrix_sync(b_frag, b + k * N + warpN * WMMA_N, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(c + warpM * WMMA_M * N + warpN * WMMA_N, c_frag, N, wmma::mem_row_major);
}

int main() {
    int N = 8192;
    size_t bytes_float = N * N * sizeof(float);
    size_t bytes_half = N * N * sizeof(half);
    float Nbr_GFLOPS = 2.0f * N / 1000.0f * N / 1000.0f * N / 1000.0f;

    vector<float> h_a(N * N), h_b(N * N), h_c(N * N);
    generate(h_a.begin(), h_a.end(), []() { return (float)(rand() % 10); });
    generate(h_b.begin(), h_b.end(), []() { return (float)(rand() % 10); });

    
    // Allocations GPU
    
    float* d_a, * d_b, * d_c;
    half* d_a_half, * d_b_half;

    cudaMalloc(&d_a, bytes_float);
    cudaMalloc(&d_b, bytes_float);
    cudaMalloc(&d_c, bytes_float);
    cudaMalloc(&d_a_half, bytes_half);
    cudaMalloc(&d_b_half, bytes_half);

    cudaMemcpy(d_a, h_a.data(), bytes_float, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes_float, cudaMemcpyHostToDevice);

    // Conversion float -> half sur GPU
    // On utilise cuBLAS pour la conversion
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Convertir manuellement float -> half sur host
    vector<__half> h_a_half(N * N), h_b_half(N * N);
    for (int i = 0; i < N * N; i++) {
        h_a_half[i] = __float2half(h_a[i]);
        h_b_half[i] = __float2half(h_b[i]);
    }
    cudaMemcpy(d_a_half, h_a_half.data(), bytes_half, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_half, h_b_half.data(), bytes_half, cudaMemcpyHostToDevice);

    cudaEvent_t e1, e2, e3, e4, e5, e6;
    cudaEventCreate(&e1); cudaEventCreate(&e2);
    cudaEventCreate(&e3); cudaEventCreate(&e4);
    cudaEventCreate(&e5); cudaEventCreate(&e6);

    float Naive_time, cuBLAS_time, TC_time;

    // Test 1 : Naive
    cout << "\n--- Naive Kernel ---\n";
    int THREADS = 32;
    int BLOCKS = N / THREADS;
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    cudaEventRecord(e1);
    matrixMulXrow << <blocks, threads >> > (d_a, d_b, d_c, N);
    cudaEventRecord(e2);
    cudaEventSynchronize(e2);
    cudaEventElapsedTime(&Naive_time, e1, e2);
    printf("Naive Time:   %f ms  |  %f GFLOPS\n", Naive_time, Nbr_GFLOPS * 1000.0f / Naive_time);

    
    // Test 2 : cuBLAS
    
    cout << "\n--- cuBLAS ---\n";
    float alpha = 1.0f, beta = 0.0f;

    // Warm up
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_b, N, d_a, N, &beta, d_c, N);
    cudaDeviceSynchronize();

    cudaEventRecord(e3);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_b, N, d_a, N, &beta, d_c, N);
    cudaEventRecord(e4);
    cudaEventSynchronize(e4);
    cudaEventElapsedTime(&cuBLAS_time, e3, e4);
    printf("cuBLAS Time:  %f ms  |  %f GFLOPS\n", cuBLAS_time, Nbr_GFLOPS * 1000.0f / cuBLAS_time);

    
    // Test 3 : Tensor Cores (WMMA)
   
    cout << "\n--- Tensor Cores (WMMA) ---\n";
    dim3 tc_threads(128, 1);
    dim3 tc_blocks((N / WMMA_M + 3) / 4, N / WMMA_N);

    // Warm up
    matrixMulTensorCore << <tc_blocks, tc_threads >> > (d_a_half, d_b_half, d_c, N);
    cudaDeviceSynchronize();

    cudaEventRecord(e5);
    matrixMulTensorCore << <tc_blocks, tc_threads >> > (d_a_half, d_b_half, d_c, N);
    cudaEventRecord(e6);
    cudaEventSynchronize(e6);
    cudaEventElapsedTime(&TC_time, e5, e6);
    printf("TensorCore Time: %f ms  |  %f GFLOPS\n", TC_time, Nbr_GFLOPS * 1000.0f / TC_time);

    
    // Comparison
    
    cout << "\n--- Comparison ---\n";
    printf("Speedup cuBLAS    vs Naive: %.2f x\n", Naive_time / cuBLAS_time);
    printf("Speedup TensorCore vs Naive: %.2f x\n", Naive_time / TC_time);
    printf("Speedup TensorCore vs cuBLAS: %.2f x\n", cuBLAS_time / TC_time);

    cublasDestroy(handle);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFree(d_a_half); cudaFree(d_b_half);

    char kml;
    scanf("%c", &kml);
    return 0;
}