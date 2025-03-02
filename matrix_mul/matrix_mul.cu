#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#define M 1024

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

int main() {
    benchmarkMatrixMul();
    return 0;
}