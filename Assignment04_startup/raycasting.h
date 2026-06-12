#pragma once
#include <cstdint>
#include <vector>
#include "chamfer.h"
#include "voxelgrid.h"

// upload grid to GPU once before the viewpoint loop — returns d_grid pointer
uint8_t* uploadGrid(const VoxelGrid& grid);

// free the grid from GPU after the viewpoint loop
void freeGrid(uint8_t* d_grid);

// ray cast for one viewpoint — uses pre-uploaded d_grid
std::vector<Point3D> rayCasting_viewpoint(
    const VoxelGrid& grid,
    uint8_t* d_grid,
    const Point3D& viewpoint,
    float maxDistance
);
