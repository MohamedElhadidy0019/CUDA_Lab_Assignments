#include "chamfer.h"
#include <cmath>
#include <limits>

static float euclidean_distance(const Point3D& p1, const Point3D& p2) {
    return std::sqrt((p1.x - p2.x) * (p1.x - p2.x) +
        (p1.y - p2.y) * (p1.y - p2.y) +
        (p1.z - p2.z) * (p1.z - p2.z));
}

float chamfer_distance(const std::vector<Point3D>& cloud1, const std::vector<Point3D>& cloud2) {
    float total_distance1 = 0.0f, total_distance2 = 0.0f;

    for (const auto& p1 : cloud1) {
        float min_distance = std::numeric_limits<float>::max();
        for (const auto& p2 : cloud2) {
            float dist = euclidean_distance(p1, p2);
            if (dist < min_distance) min_distance = dist;
        }
        total_distance1 += min_distance;
    }

    for (const auto& p2 : cloud2) {
        float min_distance = std::numeric_limits<float>::max();
        for (const auto& p1 : cloud1) {
            float dist = euclidean_distance(p2, p1);
            if (dist < min_distance) min_distance = dist;
        }
        total_distance2 += min_distance;
    }

    return (total_distance1 / cloud1.size()) + (total_distance2 / cloud2.size());
}
