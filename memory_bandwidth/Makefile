CUDA_PATH ?= /usr/local/cuda
NVCC ?= $(CUDA_PATH)/bin/nvcc
TARGET ?= memory_bandwidth

.PHONY: all clean

all: $(TARGET)

$(TARGET): memory_bandwidth.cu
	$(NVCC) -o $@ $^

clean:
	rm -f $(TARGET)