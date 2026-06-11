#!/bin/bash
# mkdir -p build Output
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./Assignment04
