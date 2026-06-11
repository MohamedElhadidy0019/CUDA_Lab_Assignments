#!/bin/bash
set -e

echo "=========================================="
echo " Assignment 1 - Main"
echo "=========================================="
nvcc -std=c++14 -arch=sm_61 assignment01.cu -o a1 && ./a1

echo ""
echo "=========================================="
echo " Experiment 1 - Thread Counts"
echo "=========================================="
nvcc -std=c++14 -arch=sm_61 exp1_threads.cu -o exp1 && ./exp1

echo ""
echo "=========================================="
echo " Experiment 2 - Managed vs Manual Memory"
echo "=========================================="
nvcc -std=c++14 -arch=sm_61 exp2_managed.cu -o exp2 && ./exp2

echo ""
echo "=========================================="
echo " Experiment 3 - CPU vs GPU Initialization"
echo "=========================================="
nvcc -std=c++14 -arch=sm_61 exp3_gpu_init.cu -lcurand -o exp3 && ./exp3
