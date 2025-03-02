CUDA_PATH ?= /usr/local/cuda
NVCC ?= $(CUDA_PATH)/bin/nvcc
TARGET ?= matrix_mul

.PHONY: all clean

all: $(TARGET)

$(TARGET): matrix_mul.cu
	$(NVCC) -o $@ $^

clean:
	rm -f $(TARGET)