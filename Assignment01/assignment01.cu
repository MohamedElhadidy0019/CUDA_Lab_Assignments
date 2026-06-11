#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

// 900M doubles * 3 arrays * 8 bytes = 21.6 GB, exceeds GTX 1060's 6 GB
// Max safe N = 6000 MB / (3 * 8 bytes) ~ 250M
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

int main() {
    int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    std::cout << "N               = " << N << std::endl;
    std::cout << "Threads/block   = " << THREADS_PER_BLOCK << std::endl;
    std::cout << "Num blocks      = " << numBlocks << std::endl;
    std::cout << "Total threads   = " << (long long)numBlocks * THREADS_PER_BLOCK << "\n" << std::endl;

    // --- Allocate host memory ---
    double* h_a    = new double[N];
    double* h_b    = new double[N];
    double* h_c_cpu = new double[N];  // CPU result
    double* h_c_gpu = new double[N];  // GPU result (copied back)

    // ----------------------------------------------------------------
    // 1. Time CPU initialization
    // ----------------------------------------------------------------
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<double>(rand()) / RAND_MAX;
        h_b[i] = static_cast<double>(rand()) / RAND_MAX;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double time_init = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "[1] CPU init time        : " << time_init << " s" << std::endl;

    // --- Allocate GPU memory ---
    double *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(double)));

    // ----------------------------------------------------------------
    // 2. Time Host -> Device transfer + bandwidth
    // ----------------------------------------------------------------
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    cudaEventRecord(ev0);
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(double), cudaMemcpyHostToDevice));
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);

    float time_h2d_ms;
    cudaEventElapsedTime(&time_h2d_ms, ev0, ev1);
    double time_h2d = time_h2d_ms / 1000.0;
    double bytes_h2d = 2.0 * N * sizeof(double);
    double bw_h2d = bytes_h2d / time_h2d / 1e9;
    std::cout << "[2] Host->Device time    : " << time_h2d << " s  |  bandwidth: " << bw_h2d << " GB/s" << std::endl;

    // ----------------------------------------------------------------
    // 3. Time kernel execution
    // ----------------------------------------------------------------
    cudaEventRecord(ev0);
    addVectors<<<numBlocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    CUDA_CHECK(cudaGetLastError());

    float time_kernel_ms;
    cudaEventElapsedTime(&time_kernel_ms, ev0, ev1);
    double time_kernel = time_kernel_ms / 1000.0;
    std::cout << "[3] GPU kernel time      : " << time_kernel << " s" << std::endl;

    // ----------------------------------------------------------------
    // 4. Time Device -> Host transfer + bandwidth
    // ----------------------------------------------------------------
    cudaEventRecord(ev0);
    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, N * sizeof(double), cudaMemcpyDeviceToHost));
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);

    float time_d2h_ms;
    cudaEventElapsedTime(&time_d2h_ms, ev0, ev1);
    double time_d2h = time_d2h_ms / 1000.0;
    double bytes_d2h = 1.0 * N * sizeof(double);
    double bw_d2h = bytes_d2h / time_d2h / 1e9;
    std::cout << "[4] Device->Host time    : " << time_d2h << " s  |  bandwidth: " << bw_d2h << " GB/s" << std::endl;

    // ----------------------------------------------------------------
    // 5. Time CPU addition
    // ----------------------------------------------------------------
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++)
        h_c_cpu[i] = h_a[i] + h_b[i];
    t1 = std::chrono::high_resolution_clock::now();
    double time_cpu_add = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "[5] CPU addition time    : " << time_cpu_add << " s\n" << std::endl;

    // ----------------------------------------------------------------
    // 6. Compare GPU and CPU results
    // ----------------------------------------------------------------
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (std::abs(h_c_gpu[i] - h_c_cpu[i]) > 1e-9)
            errors++;
    }
    if (errors == 0)
        std::cout << "Verification: PASSED (GPU and CPU results match)" << std::endl;
    else
        std::cout << "Verification: FAILED (" << errors << " mismatches)" << std::endl;

    // --- Cleanup ---
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    delete[] h_a; delete[] h_b; delete[] h_c_cpu; delete[] h_c_gpu;

    return 0;
}
