#pragma once
#include <cmath>
#include <tuple>
#include <vector>
#include <iostream>
#include "chamfer.h"

struct VoxelGrid {
    int sizeX, sizeY, sizeZ;
    float voxelSize;
    std::vector<bool> grid;

    VoxelGrid(int x, int y, int z, float size)
        : sizeX(x), sizeY(y), sizeZ(z), voxelSize(size), grid(x * y * z, false) {}

    int getIndex(int x, int y, int z) const {
        if (x >= 0 && x < sizeX && y >= 0 && y < sizeY && z >= 0 && z < sizeZ)
            return x + y * sizeX + z * sizeX * sizeY;
        return -1;
    }

    bool isOccupied(int x, int y, int z) const {
        int index = getIndex(x, y, z);
        return index != -1 && grid[index];
    }

    void setOccupied(int x, int y, int z) {
        int index = getIndex(x, y, z);
        if (index != -1) grid[index] = true;
    }

    std::tuple<int, int, int> pointToGridIndex(const Point3D& point) const {
        int x = static_cast<int>(point.x / voxelSize);
        int y = static_cast<int>(point.y / voxelSize);
        int z = static_cast<int>(point.z / voxelSize);
        return { x, y, z };
    }

    Point3D gridIndexToPoint(int x, int y, int z) const {
        return { (x + 0.5f) * voxelSize, (y + 0.5f) * voxelSize, (z + 0.5f) * voxelSize };
    }

    void insertPointCloud(const std::vector<Point3D>& points) {
        for (const auto& point : points) {
            if (point.x < 0 || point.x >= voxelSize * sizeX ||
                point.y < 0 || point.y >= voxelSize * sizeY ||
                point.z < 0 || point.z >= voxelSize * sizeZ) {
                std::cerr << "Point (" << point.x << ", " << point.y << ", " << point.z << ") is out of bounds. Skipping this point.\n";
                continue;
            }
            auto [x, y, z] = pointToGridIndex(point);
            setOccupied(x, y, z);
        }
    }

    std::vector<Point3D> extractPointCloud() const {
        std::vector<Point3D> pointCloud;
        for (int z = 0; z < sizeZ; ++z)
            for (int y = 0; y < sizeY; ++y)
                for (int x = 0; x < sizeX; ++x)
                    if (isOccupied(x, y, z))
                        pointCloud.push_back(gridIndexToPoint(x, y, z));
        return pointCloud;
    }
};
