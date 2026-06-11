#include <iostream>
#include <cfloat>
#include <chrono>
#include <cuda_runtime.h>
#include <curand.h>

const int N = 900000000;
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

__global__ void findMaxKernel(const float* data, int n, float* block_max, int* block_idx) {
    extern __shared__ float sdata[];
    int* sidx = (int*)&sdata[blockDim.x];

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) { sdata[tid] = data[i]; sidx[tid] = i; }
    else        { sdata[tid] = -FLT_MAX; sidx[tid] = -1; }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) {
            sdata[tid] = sdata[tid + s];
            sidx[tid]  = sidx[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_max[blockIdx.x] = sdata[0];
        block_idx[blockIdx.x] = sidx[0];
    }
}

// second-pass kernel: reduce partial results keeping original indices
// data_max/data_idx are the partial results from the previous pass
__global__ void reduceMaxKernel(const float* data_max, const int* data_idx, int n,
                                float* block_max, int* block_idx) {
    extern __shared__ float sdata[];
    int* sidx = (int*)&sdata[blockDim.x];

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    // load partial results — indices are already the original global indices
    if (i < n) { sdata[tid] = data_max[i]; sidx[tid] = data_idx[i]; }
    else        { sdata[tid] = -FLT_MAX;   sidx[tid] = -1; }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) {
            sdata[tid] = sdata[tid + s];
            sidx[tid]  = sidx[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_max[blockIdx.x] = sdata[0];
        block_idx[blockIdx.x] = sidx[0];
    }
}

int main() {
    int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int sharedMemSize = THREADS_PER_BLOCK * (sizeof(float) + sizeof(int));

    std::cout << "N             = " << N << std::endl;
    std::cout << "Threads/block = " << THREADS_PER_BLOCK << "\n" << std::endl;

    // ----------------------------------------------------------------
    // 1. Init on GPU
    // ----------------------------------------------------------------
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, (long long)N * sizeof(float)));

    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 42ULL));

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);

    cudaEventRecord(ev0);
    CURAND_CHECK(curandGenerateUniform(gen, d_data, N));
    cudaEventRecord(ev1); cudaEventSynchronize(ev1);
    float init_ms; cudaEventElapsedTime(&init_ms, ev0, ev1);
    std::cout << "[1] GPU init time     : " << init_ms / 1000.0 << " s" << std::endl;

    // ----------------------------------------------------------------
    // 2. Multi-pass GPU reduction — everything stays on GPU
    // ----------------------------------------------------------------
    float* d_cur_max;
    int*   d_cur_idx;
    CUDA_CHECK(cudaMalloc(&d_cur_max, numBlocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cur_idx, numBlocks * sizeof(int)));

    cudaEventRecord(ev0);

    // Pass 1: 900M → numBlocks results
    findMaxKernel<<<numBlocks, THREADS_PER_BLOCK, sharedMemSize>>>(
        d_data, N, d_cur_max, d_cur_idx);
    CUDA_CHECK(cudaGetLastError());

    int count = numBlocks;
    int pass  = 2;

    // Keep reducing until one block remains
    while (count > 1) {
        int blocks = (count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        float* d_next_max;
        int*   d_next_idx;
        CUDA_CHECK(cudaMalloc(&d_next_max, blocks * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_next_idx, blocks * sizeof(int)));

        reduceMaxKernel<<<blocks, THREADS_PER_BLOCK, sharedMemSize>>>(
            d_cur_max, d_cur_idx, count, d_next_max, d_next_idx);
        CUDA_CHECK(cudaGetLastError());

        cudaFree(d_cur_max);
        cudaFree(d_cur_idx);
        d_cur_max = d_next_max;
        d_cur_idx = d_next_idx;
        count = blocks;

        std::cout << "  pass " << pass++ << ": " << count << " elements remaining" << std::endl;
    }

    cudaEventRecord(ev1); cudaEventSynchronize(ev1);
    float reduce_ms; cudaEventElapsedTime(&reduce_ms, ev0, ev1);
    std::cout << "[2] GPU reduction     : " << reduce_ms / 1000.0 << " s" << std::endl;

    // ----------------------------------------------------------------
    // 3. Copy only the single result (2 values) to CPU
    // ----------------------------------------------------------------
    float gpu_max_val;
    int   gpu_max_idx;
    CUDA_CHECK(cudaMemcpy(&gpu_max_val, d_cur_max, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&gpu_max_idx, d_cur_idx, sizeof(int),   cudaMemcpyDeviceToHost));
    std::cout << "[3] GPU result        : index=" << gpu_max_idx
              << "  value=" << gpu_max_val << "\n" << std::endl;

    // ----------------------------------------------------------------
    // 4. CPU comparison
    // ----------------------------------------------------------------
    float* h_data = new float[(long long)N];

    cudaEventRecord(ev0);
    CUDA_CHECK(cudaMemcpy(h_data, d_data, (long long)N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventRecord(ev1); cudaEventSynchronize(ev1);
    float d2h_ms; cudaEventElapsedTime(&d2h_ms, ev0, ev1);
    std::cout << "[4] D2H full array    : " << d2h_ms / 1000.0 << " s" << std::endl;

    auto t0 = std::chrono::high_resolution_clock::now();
    float cpu_max_val = -FLT_MAX;
    int   cpu_max_idx = -1;
    for (int i = 0; i < N; i++) {
        if (h_data[i] > cpu_max_val) {
            cpu_max_val = h_data[i];
            cpu_max_idx = i;
        }
    }
    double cpu_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    std::cout << "[5] CPU search time   : " << cpu_time << " s" << std::endl;
    std::cout << "[6] CPU result        : index=" << cpu_max_idx
              << "  value=" << cpu_max_val << "\n" << std::endl;

    // ----------------------------------------------------------------
    // 5. Compare
    // ----------------------------------------------------------------
    if (gpu_max_val == cpu_max_val)
        std::cout << "Verification: PASSED (values match)" << std::endl;
    else
        std::cout << "Verification: FAILED  GPU=" << gpu_max_val
                  << "  CPU=" << cpu_max_val << std::endl;

    curandDestroyGenerator(gen);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    cudaFree(d_data); cudaFree(d_cur_max); cudaFree(d_cur_idx);
    delete[] h_data;
    return 0;
}
