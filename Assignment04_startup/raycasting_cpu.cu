#include "raycasting.h"
#include <cmath>

std::vector<Point3D> rayCasting_viewpoint(
    const VoxelGrid& grid,
    const Point3D& viewpoint,
    float maxDistance)
{
    std::vector<Point3D> visible_points;

    for (int x = 0; x < grid.sizeX; x++) {
        for (int y = 0; y < grid.sizeY; y++) {
            for (int z = 0; z < grid.sizeZ; z++) {
                if (!grid.isOccupied(x, y, z)) continue;

                Point3D voxel_center = grid.gridIndexToPoint(x, y, z);
                Point3D ray_direction = voxel_center - viewpoint;

                // Normalize
                float len = std::sqrt(ray_direction.x * ray_direction.x +
                                      ray_direction.y * ray_direction.y +
                                      ray_direction.z * ray_direction.z);
                Point3D norm_dir = ray_direction * (1.0f / len);

                Point3D current = viewpoint;
                float stepSize = grid.voxelSize / 2.0f;
                float distance = 0.0f;

                while (distance < maxDistance) {
                    auto [cx, cy, cz] = grid.pointToGridIndex(current);
                    if (grid.isOccupied(cx, cy, cz)) {
                        visible_points.push_back(grid.gridIndexToPoint(cx, cy, cz));
                        break;
                    }
                    current = current + norm_dir * stepSize;
                    distance += stepSize;
                }
            }
        }
    }

    return visible_points;
}
