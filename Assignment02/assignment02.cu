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

// ----------------------------------------------------------------
// GPU max reduction kernel
// Each block finds its local max value and index using shared memory.
// Results written to block_max and block_idx (one entry per block).
// ----------------------------------------------------------------
__global__ void findMaxKernel(const float* data, int n, float* block_max, int* block_idx) {
    // shared memory holds values and indices for this block's threads
    extern __shared__ float sdata[];
    int* sidx = (int*)&sdata[blockDim.x];

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    // load into shared memory, out-of-bounds threads get -FLT_MAX
    if (i < n) {
        sdata[tid] = data[i];
        sidx[tid]  = i;
    } else {
        sdata[tid] = -FLT_MAX;
        sidx[tid]  = -1;
    }
    __syncthreads();

    // reduction: each round halves the active threads
    // thread 0 ends up with the block's max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) {
                sdata[tid] = sdata[tid + s];
                sidx[tid]  = sidx[tid + s];
            }
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
    std::cout << "N             = " << N << std::endl;
    std::cout << "Threads/block = " << THREADS_PER_BLOCK << std::endl;
    std::cout << "Num blocks    = " << numBlocks << "\n" << std::endl;

    // ----------------------------------------------------------------
    // 1. Allocate GPU memory and initialize with cuRAND
    // ----------------------------------------------------------------
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, (long long)N * sizeof(float)));

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 42ULL));

    cudaEventRecord(ev0);
    CURAND_CHECK(curandGenerateUniform(gen, d_data, N));
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    float init_ms;
    cudaEventElapsedTime(&init_ms, ev0, ev1);
    std::cout << "[1] GPU init time     : " << init_ms / 1000.0 << " s" << std::endl;

    // ----------------------------------------------------------------
    // 2. Find max on GPU
    // ----------------------------------------------------------------
    float* d_block_max;
    int*   d_block_idx;
    CUDA_CHECK(cudaMalloc(&d_block_max, numBlocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_block_idx, numBlocks * sizeof(int)));

    // shared memory size: THREADS_PER_BLOCK floats + THREADS_PER_BLOCK ints
    int sharedMemSize = THREADS_PER_BLOCK * (sizeof(float) + sizeof(int));

    cudaEventRecord(ev0);
    findMaxKernel<<<numBlocks, THREADS_PER_BLOCK, sharedMemSize>>>(
        d_data, N, d_block_max, d_block_idx);
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    CUDA_CHECK(cudaGetLastError());
    float kernel_ms;
    cudaEventElapsedTime(&kernel_ms, ev0, ev1);
    std::cout << "[2] GPU kernel time   : " << kernel_ms / 1000.0 << " s" << std::endl;

    // copy partial results to CPU and do final reduction
    float* h_block_max = new float[numBlocks];
    int*   h_block_idx = new int[numBlocks];

    cudaEventRecord(ev0);
    CUDA_CHECK(cudaMemcpy(h_block_max, d_block_max, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_block_idx, d_block_idx, numBlocks * sizeof(int),   cudaMemcpyDeviceToHost));
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    float d2h_partial_ms;
    cudaEventElapsedTime(&d2h_partial_ms, ev0, ev1);
    std::cout << "[3] D2H partial time  : " << d2h_partial_ms / 1000.0 << " s  ("
              << numBlocks << " block results)" << std::endl;

    // final reduction on CPU over partial results
    float gpu_max_val = -FLT_MAX;
    int   gpu_max_idx = -1;
    for (int i = 0; i < numBlocks; i++) {
        if (h_block_max[i] > gpu_max_val) {
            gpu_max_val = h_block_max[i];
            gpu_max_idx = h_block_idx[i];
        }
    }
    std::cout << "[4] GPU result        : index=" << gpu_max_idx
              << "  value=" << gpu_max_val << "\n" << std::endl;

    // ----------------------------------------------------------------
    // 3. Copy full array to CPU and find max on CPU
    // ----------------------------------------------------------------
    float* h_data = new float[(long long)N];

    cudaEventRecord(ev0);
    CUDA_CHECK(cudaMemcpy(h_data, d_data, (long long)N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    float d2h_ms;
    cudaEventElapsedTime(&d2h_ms, ev0, ev1);
    std::cout << "[5] D2H full array    : " << d2h_ms / 1000.0 << " s  |  bandwidth: "
              << ((long long)N * sizeof(float)) / (d2h_ms / 1000.0) / 1e9 << " GB/s" << std::endl;

    auto t0 = std::chrono::high_resolution_clock::now();
    float cpu_max_val = -FLT_MAX;
    int   cpu_max_idx = -1;
    for (int i = 0; i < N; i++) {
        if (h_data[i] > cpu_max_val) {
            cpu_max_val = h_data[i];
            cpu_max_idx = i;
        }
    }
    double cpu_ms = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    std::cout << "[6] CPU search time   : " << cpu_ms << " s" << std::endl;
    std::cout << "[7] CPU result        : index=" << cpu_max_idx
              << "  value=" << cpu_max_val << "\n" << std::endl;

    // ----------------------------------------------------------------
    // 4. Compare results
    // ----------------------------------------------------------------
    if (gpu_max_val == cpu_max_val)
        std::cout << "Verification: PASSED (values match)" << std::endl;
    else
        std::cout << "Verification: FAILED  GPU=" << gpu_max_val
                  << "  CPU=" << cpu_max_val << std::endl;

    // cleanup
    curandDestroyGenerator(gen);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaFree(d_data);
    cudaFree(d_block_max);
    cudaFree(d_block_idx);
    delete[] h_block_max;
    delete[] h_block_idx;
    delete[] h_data;

    return 0;
}
