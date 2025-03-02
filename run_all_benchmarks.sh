#!/bin/bash

# Compile all benchmarks
make -C vector_add
make -C matrix_mul
make -C memory_bandwidth

# Run all benchmarks and append the output to a single file with separators
echo "Running vector_add benchmark..." > results.txt
vector_add/vector_add >> results.txt
echo -e "\nRunning matrix_mul benchmark..." >> results.txt
matrix_mul/matrix_mul >> results.txt
echo -e "\nRunning memory_bandwidth benchmark..." >> results.txt
memory_bandwidth/memory_bandwidth >> results.txt