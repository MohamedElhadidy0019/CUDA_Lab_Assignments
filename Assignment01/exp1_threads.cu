#include <iostream>
#include <cuda_runtime.h>

// Experiment 1: effect of different thread counts on kernel runtime

const int N = 200000000;

#define CUDA_CHECK(call) {                                                   \
    cudaError_t err = call;                                                  \
    if (err != cudaSuccess) {                                                \
        std::cerr << "CUDA error at line " << __LINE__ << ": "              \
                  << cudaGetErrorString(err) << std::endl;                   \
        exit(1);                                                             \
    }                                                                        \
}

__global__ void addVectors(const double* a, const double* b, double* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

void runWithThreadCount(int threadsPerBlock, double* d_a, double* d_b, double* d_c) {
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    cudaEventRecord(ev0);
    addVectors<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, N);
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    CUDA_CHECK(cudaGetLastError());

    float ms;
    cudaEventElapsedTime(&ms, ev0, ev1);

    std::cout << "  threads/block=" << threadsPerBlock
              << "  blocks=" << numBlocks
              << "  kernel time=" << ms / 1000.0 << " s" << std::endl;

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
}

int main() {
    // Allocate and initialize on CPU
    double* h_a = new double[N];
    double* h_b = new double[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<double>(rand()) / RAND_MAX;
        h_b[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    // Allocate and upload to GPU once (transfer is not what we're measuring)
    double *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(double), cudaMemcpyHostToDevice));

    std::cout << "Kernel runtime for different thread counts (N=" << N << "):\n" << std::endl;
    for (int t : {32, 64, 128, 256, 512, 1024})
        runWithThreadCount(t, d_a, d_b, d_c);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    delete[] h_a; delete[] h_b;
    return 0;
}
