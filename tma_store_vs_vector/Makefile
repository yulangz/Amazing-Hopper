CUDA_PATH ?= /usr/local/cuda
CUTLASS_PATH ?= /opt/cutlass
NVCC := $(CUDA_PATH)/bin/nvcc
FLAGS := -std=c++17 -O3 -lineinfo -gencode arch=compute_90a,code=sm_90a --expt-relaxed-constexpr -I$(CUTLASS_PATH)/include

.PHONY: all smoke
all: build/store_repro build/ring_stage_repro

build:
	mkdir -p $@

build/store_repro: store_repro.cu | build
	$(NVCC) $(FLAGS) $< -lcuda -o $@

build/ring_stage_repro: ring_stage_repro.cu hopper.cuh | build
	$(NVCC) $(FLAGS) $< -lcuda -o $@

# Small correctness runs, not performance measurements.
smoke: all
	build/store_repro 64 64 2 8 2 0
	build/store_repro 64 4096 2 4 2 0
	build/ring_stage_repro copy 4096 lane0 32 1 2 0
	build/ring_stage_repro gemm 4096 lane0 32 1 2 0
	build/ring_stage_repro gemm 4096 elect 64 0 2 0
