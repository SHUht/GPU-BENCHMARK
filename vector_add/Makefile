CUDA_PATH ?= /usr/local/cuda
NVCC ?= $(CUDA_PATH)/bin/nvcc
TARGET ?= vector_add

.PHONY: all clean

all: $(TARGET)

$(TARGET): vector_add.cu
	$(NVCC) -o $@ $^

clean:
	rm -f $(TARGET)