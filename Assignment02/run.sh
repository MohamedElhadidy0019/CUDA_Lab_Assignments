#!/bin/bash
set -e

echo "=========================================="
echo " Assignment 2 - CPU Final Reduction"
echo "=========================================="
nvcc -std=c++14 -arch=sm_61 assignment02.cu -lcurand -o a2 && ./a2

echo ""
echo "=========================================="
echo " Assignment 2 - Full GPU Reduction"
echo "=========================================="
nvcc -std=c++14 -arch=sm_61 assignment02_gpu_reduce.cu -lcurand -o a2_gpu && ./a2_gpu
