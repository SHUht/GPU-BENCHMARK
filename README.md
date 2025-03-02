# GPU Benchmark

This repository contains a framework for benchmarking GPU performance using CUDA. It includes compilation commands, CUDA test cases, performance data collection, and scripts for compiling, running, and displaying performance levels across different GPU hardware platforms.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
    - [Compiling](#compiling)
    - [Running Benchmarks](#running-benchmarks)
    - [Displaying Results](#displaying-results)
- [Benchmark Cases](#benchmark-cases)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites
- CUDA Toolkit (https://developer.nvidia.com/cuda-toolkit)
- NVIDIA GPU with CUDA support
- CMake (https://cmake.org/)

## Installation
Clone the repository to your local machine:
```bash
git clone https://github.com/your_username/gpu-benchmark.git
cd gpu-benchmark
```

## Usage

### Compiling
Run the following commands to compile all benchmark cases:
```bash
make all
```

### Running Benchmarks
Execute the benchmark script to run all tests:
```bash
./run_all_benchmarks.sh
```

### Displaying Results
After running the benchmarks, use the provided script to display the performance results:
```bash
python display_results.py
```

## Benchmark Cases
The benchmark includes the following test cases:
- **Vector Addition:** Measures the performance of adding two vectors.
- **Matrix Multiplication:** Measures the performance of multiplying two matrices.
- **Memory Bandwidth:** Measures the memory bandwidth by copying data between the host and device.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.