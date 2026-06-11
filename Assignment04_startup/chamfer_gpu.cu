#include "chamfer.h"
#include <cfloat>
#include <cuda_runtime.h>

__global__ void chamfer_distance_kernel_shared(const Point3D* cloud1, int n, const Point3D* cloud2, int m, float* distances) {
    __shared__ float sx[256];
    __shared__ float sy[256];
    __shared__ float sz[256];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;

    bool valid = (tid < n);
    float px = 0, py = 0, pz = 0;
    if (valid) {
        px = cloud1[tid].x;
        py = cloud1[tid].y;
        pz = cloud1[tid].z;
    }

    float minDist2 = FLT_MAX;

    int tiles = (m + blockDim.x - 1) / blockDim.x;
    for (int tile = 0; tile < tiles; tile++) {
        int idx = tile * blockDim.x + lid;
        if (idx < m) {
            sx[lid] = cloud2[idx].x;
            sy[lid] = cloud2[idx].y;
            sz[lid] = cloud2[idx].z;
        }
        __syncthreads();

        if (valid) {
            int tile_end = min(blockDim.x, m - tile * blockDim.x);
            for (int j = 0; j < tile_end; j++) {
                float dx = px - sx[j];
                float dy = py - sy[j];
                float dz = pz - sz[j];
                float dist2 = dx*dx + dy*dy + dz*dz;
                if (dist2 < minDist2) minDist2 = dist2;
            }
        }
        __syncthreads();
    }

    if (valid)
        distances[tid] = sqrtf(minDist2);
}

float chamfer_distance(const std::vector<Point3D>& cloud1, const std::vector<Point3D>& cloud2) {
    int n = (int)cloud1.size();
    int m = (int)cloud2.size();

    Point3D *d_cloud1, *d_cloud2;
    float *d_dist1, *d_dist2;

    cudaMalloc(&d_cloud1, n * sizeof(Point3D));
    cudaMalloc(&d_cloud2, m * sizeof(Point3D));
    cudaMalloc(&d_dist1,  n * sizeof(float));
    cudaMalloc(&d_dist2,  m * sizeof(float));

    cudaMemcpy(d_cloud1, cloud1.data(), n * sizeof(Point3D), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cloud2, cloud2.data(), m * sizeof(Point3D), cudaMemcpyHostToDevice);

    int blockSize = 256;
    chamfer_distance_kernel_shared<<<(n + blockSize-1)/blockSize, blockSize>>>(d_cloud1, n, d_cloud2, m, d_dist1);
    chamfer_distance_kernel_shared<<<(m + blockSize-1)/blockSize, blockSize>>>(d_cloud2, m, d_cloud1, n, d_dist2);

    cudaDeviceSynchronize();

    std::vector<float> dist1(n), dist2(m);
    cudaMemcpy(dist1.data(), d_dist1, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dist2.data(), d_dist2, m * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_cloud1); cudaFree(d_cloud2);
    cudaFree(d_dist1);  cudaFree(d_dist2);

    double sum1 = 0, sum2 = 0;
    for (float d : dist1) sum1 += d;
    for (float d : dist2) sum2 += d;

    return (float)(sum1/n + sum2/m);
}
