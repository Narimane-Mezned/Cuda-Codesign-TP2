#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using std::cout;
using std::generate;
using std::vector;

__global__ void matrixMulXrow(const int* a, const int* b, int* c, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= N || col >= N) return;
    int tmp = 0;
    for (int k = 0; k < N; k++) tmp += a[row * N + k] * b[k * N + col];
    c[row * N + col] = tmp;
}

__global__ void matrixMulYrow(const int* a, const int* b, int* c, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;
    int tmp = 0;
    for (int k = 0; k < N; k++) tmp += a[row * N + k] * b[k * N + col];
    c[row * N + col] = tmp;
}

int main() {
    int N = 8192;
    size_t bytes = N * N * sizeof(int);
    float Nbr_GFLOPS = 2.0f * N / 1000.0f * N / 1000.0f * N / 1000.0f;

    vector<int> h_a(N * N), h_b(N * N), h_c(N * N);
    generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
    generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

    cout << "Step1 : h_a and h_b generation\n";
    cout << "Step2 : Mem Allocation on host\n";

    int* d_a, * d_b, * d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cout << "Step3 : Launch Event to measure Time\n";

    cudaEvent_t e1, e2, e3, e4;
    cudaEventCreate(&e1);
    cudaEventCreate(&e2);
    cudaEventCreate(&e3);
    cudaEventCreate(&e4);

    float Host2Dev_time, Kernel_time, Dev2Host_time;

    cout << "Step3 : Copy Data To Device\n";
    cudaEventRecord(e1);
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(e2);
    cudaEventSynchronize(e2);
    cudaEventElapsedTime(&Host2Dev_time, e1, e2);

    int THREADS = 32;
    int BLOCKS = N / THREADS;
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

   
    cudaEventRecord(e2);
    matrixMulYrow << <blocks, threads >> > (d_a, d_b, d_c, N);
    cudaEventRecord(e3);
    cudaEventSynchronize(e3);
    cudaEventElapsedTime(&Kernel_time, e2, e3);

    cudaEventRecord(e3);
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(e4);
    cudaEventSynchronize(e4);
    cudaEventElapsedTime(&Dev2Host_time, e3, e4);

    float Total = Host2Dev_time + Kernel_time + Dev2Host_time;

    printf("Time elapsed on Host To Device Transfer: %f ms.\n\n", Host2Dev_time);
    printf("Time elapsed on matrix multiplication on GPU: %f ms.\n\n", Kernel_time);
    printf("Time elapsed on Device To Host Transfer: %f ms.\n\n", Dev2Host_time);
    printf("Total Time: %f ms.\n\n", Total);
    printf("Kernel Execution Performance: %f GFLOPS.\n\n", Nbr_GFLOPS * 1000.0f / Kernel_time);

    cout << "COMPLETED SUCCESSFULLY\n";

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    char kml;
    scanf("%c", &kml);
    return 0;
}