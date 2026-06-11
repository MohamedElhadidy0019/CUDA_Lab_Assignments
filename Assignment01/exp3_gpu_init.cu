#include <iostream>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <curand.h>

// Experiment 3: CPU init vs GPU init

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

#define CURAND_CHECK(call) {                                                 \
    curandStatus_t err = call;                                               \
    if (err != CURAND_STATUS_SUCCESS) {                                      \
        std::cerr << "cuRAND error at line " << __LINE__ << std::endl;      \
        exit(1);                                                             \
    }                                                                        \
}

__global__ void addVectors(const double* a, const double* b, double* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

void runCpuInit() {
    std::cout << "--- CPU initialization ---" << std::endl;

    double* h_a = new double[N];
    double* h_b = new double[N];
    double* h_c = new double[N];

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<double>(rand()) / RAND_MAX;
        h_b[i] = static_cast<double>(rand()) / RAND_MAX;
    }
    double time_init = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    std::cout << "  Init time        : " << time_init << " s" << std::endl;

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
    std::cout << "  Add kernel time  : " << kernel_ms / 1000.0 << " s" << std::endl;

    double total = time_init + (h2d_ms + kernel_ms) / 1000.0;
    std::cout << "  Total (no D2H)   : " << total << " s" << std::endl;

    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    delete[] h_a; delete[] h_b; delete[] h_c;
}

void runGpuInit() {
    std::cout << "\n--- GPU initialization (cuRAND host API) ---" << std::endl;

    double *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(double)));

    // cuRAND host API: generates all N doubles in one optimised bulk call
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 42ULL));

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);

    cudaEventRecord(ev0);
    CURAND_CHECK(curandGenerateUniformDouble(gen, d_a, N));
    CURAND_CHECK(curandGenerateUniformDouble(gen, d_b, N));
    cudaEventRecord(ev1); cudaEventSynchronize(ev1);
    float init_ms; cudaEventElapsedTime(&init_ms, ev0, ev1);
    std::cout << "  GPU init time    : " << init_ms / 1000.0 << " s" << std::endl;

    int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaEventRecord(ev0);
    addVectors<<<numBlocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    cudaEventRecord(ev1); cudaEventSynchronize(ev1);
    float kernel_ms; cudaEventElapsedTime(&kernel_ms, ev0, ev1);
    std::cout << "  Add kernel time  : " << kernel_ms / 1000.0 << " s" << std::endl;

    double total = (init_ms + kernel_ms) / 1000.0;
    std::cout << "  Total (no D2H)   : " << total << " s" << std::endl;

    curandDestroyGenerator(gen);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

int main() {
    std::cout << "N = " << N << "\n" << std::endl;
    runCpuInit();
    runGpuInit();
    return 0;
}
