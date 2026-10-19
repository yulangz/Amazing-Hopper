CUDA_PATH ?= /usr/local/cuda
CUTLASS_PATH ?= /opt/cutlass
NVCC := $(CUDA_PATH)/bin/nvcc
FLAGS := -std=c++17 -O3 -lineinfo -gencode arch=compute_90a,code=sm_90a --expt-relaxed-constexpr -I$(CUTLASS_PATH)/include

.PHONY: all
all: build/repro

build/repro: repro.cu ../tma_store_vs_vector/hopper.cuh
	mkdir -p build
	$(NVCC) $(FLAGS) $< -lcuda -o $@
