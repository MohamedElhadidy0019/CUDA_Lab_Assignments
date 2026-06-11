#include <iostream>
#include <vector>
#include <cmath>
#include <cfloat>
#include <limits>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>

// Structure to represent a 3D point
struct Point3D {
    float x, y, z;
};

// Function to compute the Euclidean distance between two 3D points
float euclidean_distance(const Point3D& p1, const Point3D& p2) {
    return std::sqrt((p1.x - p2.x) * (p1.x - p2.x) +
        (p1.y - p2.y) * (p1.y - p2.y) +
        (p1.z - p2.z) * (p1.z - p2.z));
}

// ------------------------------------ BRUTE FORCE (CPU) ------------------------------------
// Brute-force implementation of Chamfer Distance
float chamfer_distance_bruteforce(const std::vector<Point3D>& cloud1, const std::vector<Point3D>& cloud2) {
    float total_distance1 = 0.0f, total_distance2 = 0.0f;

    // Find the shortest distance from each point in cloud1 to cloud2
    for (const auto& p1 : cloud1) {
        float min_distance = std::numeric_limits<float>::max();
        for (const auto& p2 : cloud2) {
            float dist = euclidean_distance(p1, p2);
            if (dist < min_distance) min_distance = dist;
        }
        total_distance1 += min_distance;
    }

    // Find the shortest distance from each point in cloud2 to cloud1
    for (const auto& p2 : cloud2) {
        float min_distance = std::numeric_limits<float>::max();
        for (const auto& p1 : cloud1) {
            float dist = euclidean_distance(p2, p1);
            if (dist < min_distance) min_distance = dist;
        }
        total_distance2 += min_distance;
    }

    // Return the normalized Chamfer Distance
    return (total_distance1 / cloud1.size()) + (total_distance2 / cloud2.size());
}

// ------------------------------------ KD-TREE (CPU) ------------------------------------
// KD-Tree node structure
struct KDNode {
    Point3D point;
    KDNode* left;
    KDNode* right;
};

// KD-Tree implementation
class KDTree {
public:
    KDNode* root;

    KDTree() : root(nullptr) {}

    // Build the KD-Tree recursively
    KDNode* build(std::vector<Point3D>& points, int depth = 0) {
        if (points.empty()) return nullptr;

        int axis = depth % 3; // Current axis (x, y, z)
        auto comparator = [axis](const Point3D& a, const Point3D& b) {
            if (axis == 0) return a.x < b.x;
            if (axis == 1) return a.y < b.y;
            return a.z < b.z;
        };

        // Sort points and pick the median
        std::sort(points.begin(), points.end(), comparator);
        int mid = points.size() / 2;

        // Create node and recursively build subtrees
        KDNode* node = new KDNode{ points[mid], nullptr, nullptr };
        std::vector<Point3D> leftPoints(points.begin(), points.begin() + mid);
        std::vector<Point3D> rightPoints(points.begin() + mid + 1, points.end());

        node->left = build(leftPoints, depth + 1);
        node->right = build(rightPoints, depth + 1);

        return node;
    }

    // Find the nearest neighbor recursively
    void nearest_neighbor(KDNode* node, const Point3D& target, Point3D& best, float& bestDist, int depth = 0) const {
        if (!node) return;

        int axis = depth % 3;
        float dist = euclidean_distance(target, node->point);

        // Update the best neighbor
        if (dist < bestDist) {
            bestDist = dist;
            best = node->point;
        }

        // Determine which subtree to search
        KDNode* next = (axis == 0 && target.x < node->point.x) ||
            (axis == 1 && target.y < node->point.y) ||
            (axis == 2 && target.z < node->point.z)
            ? node->left
            : node->right;
        KDNode* other = next == node->left ? node->right : node->left;

        // Search the closer subtree
        nearest_neighbor(next, target, best, bestDist, depth + 1);

        // Check if the other subtree might contain a closer point
        float planeDist = (axis == 0 ? std::fabs(target.x - node->point.x)
            : axis == 1 ? std::fabs(target.y - node->point.y)
            : std::fabs(target.z - node->point.z));
        if (planeDist < bestDist) {
            nearest_neighbor(other, target, best, bestDist, depth + 1);
        }
    }
};

// KD-Tree optimized Chamfer Distance
float chamfer_distance_kdtree(const std::vector<Point3D>& cloud1, const std::vector<Point3D>& cloud2) {
    KDTree tree;
    std::vector<Point3D> cloud2Copy = cloud2; // Create a modifiable copy
    tree.root = tree.build(cloud2Copy);

    float total_distance1 = 0.0f;

    // Compute shortest distances from cloud1 to cloud2
    for (const auto& p1 : cloud1) {
        Point3D best;
        float bestDist = std::numeric_limits<float>::max();
        tree.nearest_neighbor(tree.root, p1, best, bestDist);
        total_distance1 += bestDist;
    }

    KDTree tree2;
    std::vector<Point3D> cloud1Copy = cloud1; // Create a modifiable copy
    tree2.root = tree2.build(cloud1Copy);

    float total_distance2 = 0.0f;

    // Compute shortest distances from cloud2 to cloud1
    for (const auto& p2 : cloud2) {
        Point3D best;
        float bestDist = std::numeric_limits<float>::max();
        tree2.nearest_neighbor(tree2.root, p2, best, bestDist);
        total_distance2 += bestDist;
    }

    // Return the normalized Chamfer Distance
    return (total_distance1 / cloud1.size()) + (total_distance2 / cloud2.size());
}

// ------------------------------------ CUDA BRUTE FORCE ------------------------------------
// CUDA kernel to compute the nearest distances for each point in cloud1 to cloud2
__global__ void chamfer_distance_kernel(const Point3D* cloud1, int n, const Point3D* cloud2, int m, float* distances) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float px = cloud1[tid].x;
    float py = cloud1[tid].y;
    float pz = cloud1[tid].z;

    float minDist2 = FLT_MAX;
    for (int j = 0; j < m; j++) {
        float dx = px - cloud2[j].x;
        float dy = py - cloud2[j].y;
        float dz = pz - cloud2[j].z;
        float dist2 = dx*dx + dy*dy + dz*dz;
        if (dist2 < minDist2) minDist2 = dist2;
    }

    distances[tid] = sqrtf(minDist2);
}

// Host function to compute Chamfer Distance using CUDA
float chamfer_distance_cuda(const std::vector<Point3D>& cloud1, const std::vector<Point3D>& cloud2) {
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
    // cloud1 -> cloud2: find nearest in cloud2 for each point in cloud1
    chamfer_distance_kernel<<<(n + blockSize-1)/blockSize, blockSize>>>(d_cloud1, n, d_cloud2, m, d_dist1);
    // cloud2 -> cloud1: find nearest in cloud1 for each point in cloud2
    chamfer_distance_kernel<<<(m + blockSize-1)/blockSize, blockSize>>>(d_cloud2, m, d_cloud1, n, d_dist2);

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

int main() {
    // Set a fixed random seed
    srand(42);
    // Test different points number
    std::vector<int> num_of_points_list = { 100, 1000, 10000, 100000, 1000000, 2000000};
    for (auto num_of_points : num_of_points_list) {
        // Generate two random point clouds
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<Point3D> cloud1, cloud2;
        for (int i = 0; i < num_of_points; i++) {
            cloud1.push_back({ static_cast<float>(rand()) / RAND_MAX,
                              static_cast<float>(rand()) / RAND_MAX,
                              static_cast<float>(rand()) / RAND_MAX });
            cloud2.push_back({ static_cast<float>(rand()) / RAND_MAX,
                              static_cast<float>(rand()) / RAND_MAX,
                              static_cast<float>(rand()) / RAND_MAX });
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Generate Two Random Clouds with Number of Points: " << num_of_points
            << ", Time: " << std::chrono::duration<double>(end - start).count() << " seconds\n";

        // Since brute-force is too slow in large size, only test of small size
        if (num_of_points <= 10000) {
            // Measure runtime for brute-force method
            start = std::chrono::high_resolution_clock::now();
            float result_bruteforce = chamfer_distance_bruteforce(cloud1, cloud2);
            end = std::chrono::high_resolution_clock::now();
            std::cout << "Chamfer Distance (Brute Force): " << result_bruteforce
                << ", Time: " << std::chrono::duration<double>(end - start).count() << " seconds\n";
        }

        // Measure runtime for KD-Tree method
        start = std::chrono::high_resolution_clock::now();
        float result_kdtree = chamfer_distance_kdtree(cloud1, cloud2);
        end = std::chrono::high_resolution_clock::now();
        std::cout << "Chamfer Distance (KD-Tree): " << result_kdtree
            << ", Time: " << std::chrono::duration<double>(end - start).count() << " seconds\n";

        // Measure runtime for Cuda method
        start = std::chrono::high_resolution_clock::now();
        float result_cuda = chamfer_distance_cuda(cloud1, cloud2);
        end = std::chrono::high_resolution_clock::now();
        std::cout << "Chamfer Distance (Cuda): " << result_cuda
            << ", Time: " << std::chrono::duration<double>(end - start).count() << " seconds\n";

        // Consistency check: compare CUDA against brute force (small N) or KD-Tree (large N)
        float ref = (num_of_points <= 10000) ? chamfer_distance_bruteforce(cloud1, cloud2) : result_kdtree;
        const char* ref_name = (num_of_points <= 10000) ? "Brute Force" : "KD-Tree";
        float rel_err = std::fabs(result_cuda - ref) / (std::fabs(ref) + 1e-8f);
        std::cout << "Consistency Check (CUDA vs " << ref_name << "): "
            << (rel_err < 1e-3f ? "PASSED" : "FAILED")
            << " (relative error = " << rel_err << ")\n";
        std::cout << "------------------------------------------------------------\n";
    }

    return 0;
}
