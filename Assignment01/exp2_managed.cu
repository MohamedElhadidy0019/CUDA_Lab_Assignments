#include <iostream>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

// Experiment 2: managed memory vs manual memory

const int N = 200000000;
const int THREADS_PER_BLOCK = 256;

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

void runManual() {
    std::cout << "--- Manual memory (cudaMalloc + cudaMemcpy) ---" << std::endl;

    double* h_a = new double[N];
    double* h_b = new double[N];
    double* h_c = new double[N];

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<double>(rand()) / RAND_MAX;
        h_b[i] = static_cast<double>(rand()) / RAND_MAX;
    }
    double time_init = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    std::cout << "  CPU init time    : " << time_init << " s" << std::endl;

    double *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(double)));

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);

    cudaEventRecord(ev0);
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(double), cudaMemcpyHostToDevice));
    cudaEventRecord(ev1); cudaEventSynchronize(ev1);
    float h2d_ms; cudaEventElapsedTime(&h2d_ms, ev0, ev1);
    std::cout << "  H2D transfer     : " << h2d_ms / 1000.0 << " s" << std::endl;

    int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaEventRecord(ev0);
    addVectors<<<numBlocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    cudaEventRecord(ev1); cudaEventSynchronize(ev1);
    float kernel_ms; cudaEventElapsedTime(&kernel_ms, ev0, ev1);
    std::cout << "  Kernel time      : " << kernel_ms / 1000.0 << " s" << std::endl;

    cudaEventRecord(ev0);
    CUDA_CHECK(cudaMemcpy(h_c, d_c, N * sizeof(double), cudaMemcpyDeviceToHost));
    cudaEventRecord(ev1); cudaEventSynchronize(ev1);
    float d2h_ms; cudaEventElapsedTime(&d2h_ms, ev0, ev1);
    std::cout << "  D2H transfer     : " << d2h_ms / 1000.0 << " s" << std::endl;

    double total = time_init + (h2d_ms + kernel_ms + d2h_ms) / 1000.0;
    std::cout << "  Total time       : " << total << " s" << std::endl;

    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    delete[] h_a; delete[] h_b; delete[] h_c;
}

void runManaged() {
    std::cout << "\n--- Managed memory (cudaMallocManaged) ---" << std::endl;

    double *a, *b, *c;
    CUDA_CHECK(cudaMallocManaged(&a, N * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&b, N * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&c, N * sizeof(double)));

    // Init directly on the managed pointer (CPU side)
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        a[i] = static_cast<double>(rand()) / RAND_MAX;
        b[i] = static_cast<double>(rand()) / RAND_MAX;
    }
    double time_init = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    std::cout << "  CPU init time    : " << time_init << " s" << std::endl;

    // No explicit transfer needed — CUDA handles it automatically
    // (so we can't measure transfer time separately)
    int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    cudaEventRecord(ev0);
    addVectors<<<numBlocks, THREADS_PER_BLOCK>>>(a, b, c, N);
    cudaEventRecord(ev1); cudaEventSynchronize(ev1);
    CUDA_CHECK(cudaGetLastError());

    float kernel_ms; cudaEventElapsedTime(&kernel_ms, ev0, ev1);
    std::cout << "  Kernel time      : " << kernel_ms / 1000.0 << " s  (includes implicit transfer)" << std::endl;

    // Access result on CPU — triggers implicit D2H transfer
    t0 = std::chrono::high_resolution_clock::now();
    double dummy = c[0]; // force CPU access
    (void)dummy;
    double time_access = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    std::cout << "  First CPU access : " << time_access << " s  (triggers D2H migration)" << std::endl;

    double total = time_init + kernel_ms / 1000.0 + time_access;
    std::cout << "  Total time       : " << total << " s" << std::endl;

    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    cudaFree(a); cudaFree(b); cudaFree(c);
}

int main() {
    std::cout << "N = " << N << "\n" << std::endl;
    runManual();
    runManaged();
    return 0;
}
