CUDA_PATH ?= /usr/local/cuda
CUTLASS_PATH ?= /opt/cutlass
NVCC := $(CUDA_PATH)/bin/nvcc
FLAGS := -std=c++17 -O3 -lineinfo -gencode arch=compute_90a,code=sm_90a --expt-relaxed-constexpr -I$(CUTLASS_PATH)/include

.PHONY: all
all: build/repro build/minimal

build/repro: repro.cu ../tma_store_vs_vector/hopper.cuh
	mkdir -p build
	$(NVCC) $(FLAGS) $< -lcuda -o $@

build/minimal: minimal.cu
	mkdir -p build
	$(NVCC) $(FLAGS) $< -o $@

build/state_probe: state_probe.cu
	mkdir -p build
	$(NVCC) $(FLAGS) $< -o $@

build/cache_probe: cache_probe.cu
	mkdir -p build
	$(NVCC) $(FLAGS) $< -o $@
