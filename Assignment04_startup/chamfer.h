#pragma once
#include <vector>

struct Point3D {
    float x, y, z;

    Point3D operator+(const Point3D& other) const {
        return { x + other.x, y + other.y, z + other.z };
    }

    Point3D operator-(const Point3D& other) const {
        return { x - other.x, y - other.y, z - other.z };
    }

    Point3D operator*(float scalar) const {
        return { x * scalar, y * scalar, z * scalar };
    }
};

float chamfer_distance(const std::vector<Point3D>& cloud1, const std::vector<Point3D>& cloud2);
