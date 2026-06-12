#!/bin/bash
# mkdir -p build Output
rm -f Output/*
rm -rf build/*
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./Assignment04
