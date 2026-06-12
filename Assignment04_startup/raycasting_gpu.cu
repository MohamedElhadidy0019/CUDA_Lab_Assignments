#include "raycasting.h"
#include <cuda_runtime.h>
#include <cmath>

__global__ void rayCasting_kernel(
    const uint8_t* __restrict__ grid,
    int sizeX, int sizeY, int sizeZ,
    float voxelSize,
    float vpx, float vpy, float vpz,
    float maxDistance,
    uint8_t* hit_flags)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = sizeX * sizeY * sizeZ;
    if (idx >= total) return;

    // skip unoccupied voxels
    if (grid[idx] == 0) {
        hit_flags[idx] = 0;
        return;
    }

    // derive (x, y, z) from flat index
    int x = idx % sizeX;
    int y = (idx / sizeX) % sizeY;
    int z = idx / (sizeX * sizeY);

    // voxel center
    float cx = (x + 0.5f) * voxelSize;
    float cy = (y + 0.5f) * voxelSize;
    float cz = (z + 0.5f) * voxelSize;

    // ray direction from viewpoint to voxel center, normalized
    float dx = cx - vpx;
    float dy = cy - vpy;
    float dz = cz - vpz;
    float len = sqrtf(dx*dx + dy*dy + dz*dz);
    dx /= len;  dy /= len;  dz /= len;

    // march the ray
    float stepSize = voxelSize / 2.0f;
    float px = vpx, py = vpy, pz = vpz;
    float distance = 0.0f;

    while (distance < maxDistance) {
        int gx = (int)(px / voxelSize);
        int gy = (int)(py / voxelSize);
        int gz = (int)(pz / voxelSize);

        if (gx >= 0 && gx < sizeX && gy >= 0 && gy < sizeY && gz >= 0 && gz < sizeZ) {
            int gidx = gx + gy * sizeX + gz * sizeX * sizeY;
            if (grid[gidx] != 0) {
                // visible only if first hit is this voxel
                hit_flags[idx] = (gidx == idx) ? 1 : 0;
                return;
            }
        }

        px += dx * stepSize;
        py += dy * stepSize;
        pz += dz * stepSize;
        distance += stepSize;
    }

    hit_flags[idx] = 0;
}

uint8_t* uploadGrid(const VoxelGrid& grid) {
    int total = grid.sizeX * grid.sizeY * grid.sizeZ;
    std::vector<uint8_t> grid_host(total);
    for (int i = 0; i < total; i++)
        grid_host[i] = grid.grid[i] ? 1 : 0;

    uint8_t* d_grid;
    cudaMalloc(&d_grid, total * sizeof(uint8_t));
    cudaMemcpy(d_grid, grid_host.data(), total * sizeof(uint8_t), cudaMemcpyHostToDevice);
    return d_grid;
}

void freeGrid(uint8_t* d_grid) {
    cudaFree(d_grid);
}

std::vector<Point3D> rayCasting_viewpoint(
    const VoxelGrid& grid,
    uint8_t* d_grid,
    const Point3D& viewpoint,
    float maxDistance)
{
    int total = grid.sizeX * grid.sizeY * grid.sizeZ;

    uint8_t* d_hit_flags;
    cudaMalloc(&d_hit_flags, total * sizeof(uint8_t));
    cudaMemset(d_hit_flags, 0, total * sizeof(uint8_t));

    int blockSize = 256;
    int gridSize  = (total + blockSize - 1) / blockSize;
    rayCasting_kernel<<<gridSize, blockSize>>>(
        d_grid,
        grid.sizeX, grid.sizeY, grid.sizeZ,
        grid.voxelSize,
        viewpoint.x, viewpoint.y, viewpoint.z,
        maxDistance,
        d_hit_flags);

    cudaDeviceSynchronize();

    std::vector<uint8_t> hit_flags(total);
    cudaMemcpy(hit_flags.data(), d_hit_flags, total * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaFree(d_hit_flags);

    std::vector<Point3D> visible_points;
    for (int idx = 0; idx < total; idx++) {
        if (hit_flags[idx] == 1) {
            int x = idx % grid.sizeX;
            int y = (idx / grid.sizeX) % grid.sizeY;
            int z = idx / (grid.sizeX * grid.sizeY);
            visible_points.push_back(grid.gridIndexToPoint(x, y, z));
        }
    }

    return visible_points;
}
