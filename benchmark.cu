#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#define N 1000000
#define M 1024

__global__ void vectorAdd(float* A, float* B, float* C) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

__global__ void matrixMul(float* A, float* B, float* C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        float value = 0;
        for (int k = 0; k < width; ++k) {
            value += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = value;
    }
}

void benchmarkVectorAdd() {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    size_t size = N * sizeof(float);

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    auto start = std::chrono::high_resolution_clock::now();
    vectorAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::chrono::duration<double> duration = end - start;
    std::cout << "Vector Add Time: " << duration.count() << " seconds" << std::endl;

    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void benchmarkMatrixMul() {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    size_t size = M * M * sizeof(float);

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    for (int i = 0; i < M * M; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 numBlocks((M + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    auto start = std::chrono::high_resolution_clock::now();
    matrixMul<<<numBlocks, blockSize>>>(d_A, d_B, d_C, M);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::chrono::duration<double> duration = end - start;
    std::cout << "Matrix Mul Time: " << duration.count() << " seconds" << std::endl;

    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void benchmarkMemoryBandwidth() {
    float *h_A, *d_A;

    size_t size = N * sizeof(float);

    h_A = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_A[i] = i;
    }

    cudaMalloc((void**)&d_A, size);

    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Memory Bandwidth Time: " << duration.count() << " seconds" << std::endl;

    free(h_A);
    cudaFree(d_A);
}

int main() {
    benchmarkVectorAdd();
    benchmarkMatrixMul();
    benchmarkMemoryBandwidth();
    return 0;
}