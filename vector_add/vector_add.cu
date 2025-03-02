#include <iostream>
#include <chrono>

#define N 1000000

__global__ void vectorAdd(float* A, float* B, float* C) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        C[i] = A[i] + B[i];
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

int main() {
    benchmarkVectorAdd();
    return 0;
}