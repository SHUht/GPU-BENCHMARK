.PHONY: all clean

all:
	make -C vector_add
	make -C matrix_mul
	make -C memory_bandwidth

clean:
	make -C vector_add clean
	make -C matrix_mul clean
	make -C memory_bandwidth clean