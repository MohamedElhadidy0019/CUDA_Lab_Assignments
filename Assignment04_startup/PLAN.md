# Assignment 04 — Implementation Plan

## Step 1: Get CPU code running
- [ ] Create `build/` and `Output/` directories inside `Assignment04_startup/`
- [ ] Build with CMake (`cmake .. && make`)
- [ ] Run on `_64` resolution only (comment out 128 and 256 in `main()` temporarily)
- [ ] Confirm `.pcd` files appear in `Output/` and chamfer distances are printed
- [ ] Note the runtime — this is your CPU baseline

## Step 2: Plug in CUDA Chamfer Distance (low-risk, already proven)
- [ ] Copy `Point3D` struct, `chamfer_distance_kernel_shared`, and `chamfer_distance_cuda_shared()` from `Assignment03_shared_mem.cu` into `Assignment04.cu`
- [ ] Replace both `chamfer_distance(...)` CPU calls in `main()` with `chamfer_distance_cuda_shared(...)`
- [ ] Rebuild and run on `_64` — chamfer values must match the CPU baseline exactly
- [ ] Add timing prints around both CD calls so you can see the speedup

## Step 3: Prepare voxel grid for GPU (prerequisite for ray casting kernel)
- [ ] After `object_grid.insertPointCloud(...)`, copy the grid to device:
  ```cpp
  // vector<bool> can't be cudaMemcpy'd — convert first
  std::vector<uint8_t> grid_host(object_grid.grid.begin(), object_grid.grid.end());
  uint8_t* d_grid;
  cudaMalloc(&d_grid, grid_host.size());
  cudaMemcpy(d_grid, grid_host.data(), grid_host.size(), cudaMemcpyHostToDevice);
  ```
- [ ] Pass `sizeX`, `sizeY`, `sizeZ`, `voxelSize` as kernel parameters (plain ints/float)

## Step 4: Write the ray casting CUDA kernel
- [ ] Kernel signature:
  ```cpp
  __global__ void rayCasting_kernel(
      const uint8_t* grid, int sizeX, int sizeY, int sizeZ, float voxelSize,
      float vpx, float vpy, float vpz,       // viewpoint
      float maxDistance, float stepSize,
      uint8_t* hit_flags,                    // output: 1 = hit, 0 = miss  [sizeX*sizeY*sizeZ]
      float3* hit_points                     // output: endpoint if hit     [sizeX*sizeY*sizeZ]
  )
  ```
- [ ] Thread ID → voxel index: `int idx = blockIdx.x * blockDim.x + threadIdx.x`
- [ ] Convert `idx` to `(x, y, z)`, skip if not occupied (`grid[idx] == 0`)
- [ ] Compute voxel center, ray direction (voxel_center - viewpoint), normalize
- [ ] March ray from viewpoint with `stepSize = voxelSize / 2.0f`, stop at first occupied hit or maxDistance
- [ ] Write result into `hit_flags[idx]` and `hit_points[idx]`
- [ ] Launch: `rayCasting_kernel<<<(N+255)/256, 256>>>(...)` where `N = sizeX*sizeY*sizeZ`

## Step 5: Integrate kernel into the viewpoint loop
- [ ] Allocate `d_hit_flags` and `d_hit_points` once before the viewpoint loop (reuse each iteration)
- [ ] For each viewpoint: launch kernel, `cudaDeviceSynchronize()`, copy results back
- [ ] Collect hits on CPU: loop over `hit_flags`, build `view_point_visible_points` from `hit_points`
- [ ] Deduplication (unordered_set) stays on CPU — no changes needed there
- [ ] Verify on `_64`: visible point counts per viewpoint must match Step 1 CPU baseline

## Step 6: Verify correctness across resolutions
- [ ] Re-enable `_128` and `_256` in the outer loop
- [ ] Run full pipeline, compare chamfer distances with CPU baseline
- [ ] Confirm all output `.pcd` files are produced correctly

## Step 7: Measure and report
- [ ] Time each section: grid upload, ray casting per viewpoint, chamfer distance, NBV scoring
- [ ] Print GPU vs CPU speedup for ray casting and chamfer distance
- [ ] Run all 6 models (Armadillo + Dragon x 64/128/256) and tabulate results

## Optional Step 8: Batch ray casting across viewpoints
- [ ] Launch one thread per `(viewpoint_idx, voxel_idx)` pair instead of looping viewpoints serially
- [ ] Output buffers become `[100 x sizeX*sizeY*sizeZ]` — check VRAM (at 256^3: 100 x 16M bytes = 1.6 GB, fits in 6 GB)
- [ ] Eliminates 100 serial kernel launches -> one big launch
