#pragma once
#include <vector>
#include "chamfer.h"
#include "voxelgrid.h"

std::vector<Point3D> rayCasting_viewpoint(
    const VoxelGrid& grid,
    const Point3D& viewpoint,
    float maxDistance
);
