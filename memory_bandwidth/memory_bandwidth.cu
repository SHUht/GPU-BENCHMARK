#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#define N 1000000

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
    benchmarkMemoryBandwidth();
    return 0;
}