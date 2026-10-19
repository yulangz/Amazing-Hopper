// SPDX-License-Identifier: Apache-2.0
// SM90a diagnostic for activation-barrier -> dW-output shared-memory reuse.
// No PyTorch, WeftMoE runtime, NCCL, optimizer, or model checkpoint.
#include "../tma_store_vs_vector/hopper.cuh"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cutlass/arch/reg_reconfig.h>
#include <stdexcept>
#include <string>

using BF16 = cutlass::bfloat16_t;
using Barrier = cutlass::arch::ClusterTransactionBarrier;
using Mma = hopper::BF16Mma<256>;
constexpr unsigned M = 128, N = 256, K = 64, Stages = 3, Threads = 384;
constexpr unsigned OutputOffset = Stages * (M + N) * K * 2;    // 147456
constexpr unsigned BarrierOffset = OutputOffset + 3 * 4096;    // 159744
constexpr unsigned DwBarrierOffset = OutputOffset + M * N * 2; // 212992
constexpr unsigned SharedBytes = DwBarrierOffset + 128;

struct Shared {
  alignas(1024) BF16 a[Stages][M * K];
  alignas(1024) BF16 b[Stages][N * K];
  alignas(1024) unsigned char pull[3][4096];
  Barrier full[Stages], empty[Stages], pull_full[3];
};
static_assert(offsetof(Shared, full) == BarrierOffset);
static_assert(offsetof(Shared, pull_full) + 3 * sizeof(Barrier) == 159816);

#define CK(call)                                                               \
  do {                                                                         \
    auto e = (call);                                                           \
    if (e != cudaSuccess)                                                      \
      throw std::runtime_error(std::string(#call) + ": " +                     \
                               cudaGetErrorString(e));                         \
  } while (0)

__host__ __device__ unsigned short value(unsigned row, unsigned col,
                                         unsigned block, unsigned iteration) {
  return 0x3e80U + ((row * 17U + col * 7U + block + iteration) & 127U);
}

__device__ __forceinline__ void stmatrix_x2(void *ptr, unsigned a, unsigned b) {
  const auto addr = cute::cast_smem_ptr_to_uint(ptr);
  asm volatile(
      "stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};" ::"r"(addr),
      "r"(a), "r"(b)
      : "memory");
}

__device__ __forceinline__ void advance(unsigned &stage, unsigned &phase) {
  stage = (stage + 1) % Stages;
  phase ^= stage == 0;
}

__global__ void fill(BF16 *a, BF16 *b) {
  for (unsigned i = threadIdx.x; i < N * K; i += blockDim.x) {
    b[i] = BF16(1.0f);
    if (i < M * K)
      a[i] = BF16(1.0f);
  }
}

__device__ __forceinline__ void write_tile(BF16 *out, const CUtensorMap *output,
                                           unsigned iteration) {
  const unsigned tid = threadIdx.x, lane = tid % 32;
  // Same M128/N256, 128-B swizzle, STSM.x2 -> fence -> TMA-store epilogue.
  if (tid < 256) {
    const unsigned warp = tid / 32, row = warp * 16 + lane / 4;
#pragma unroll
    for (unsigned i = 0; i < N / 8; ++i) {
      const unsigned col = i * 8 + (lane % 4) * 2;
      unsigned a = value(row, col, blockIdx.x, iteration) |
                   (unsigned(value(row, col + 1, blockIdx.x, iteration)) << 16);
      unsigned b =
          value(row + 8, col, blockIdx.x, iteration) |
          (unsigned(value(row + 8, col + 1, blockIdx.x, iteration)) << 16);
      const unsigned atom = i / 8, in_atom = i % 8;
      auto *ptr = reinterpret_cast<unsigned char *>(out) + atom * M * 128 +
                  (warp * 16 + lane) * 128 + (in_atom ^ (lane % 8)) * 16;
      stmatrix_x2(ptr, a, b);
    }
    cute::tma_store_fence();
  }
  asm volatile("bar.sync 1, 256;" ::: "memory");
  if (tid < N / 64) {
    cute::SM90_TMA_STORE_2D::copy(output, out + tid * M * 64, tid * 64,
                                  blockIdx.x * M);
    cute::tma_store_arrive();
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  }
  asm volatile("bar.sync 1, 256;" ::: "memory");
}

template <bool Gemm>
__global__ __launch_bounds__(Threads, 1) void reuse(
    const __grid_constant__ CUtensorMap input_a,
    const __grid_constant__ CUtensorMap input_b,
    const __grid_constant__ CUtensorMap output, unsigned *joined,
    const BF16 *pull_input, float *probe, unsigned steps, unsigned iteration,
    unsigned cluster, bool drain, bool invalidate) {
  extern __shared__ __align__(1024) unsigned char arena[];
  auto &s = *reinterpret_cast<Shared *>(arena);
  const unsigned tid = threadIdx.x, lane = tid % 32;
  const unsigned rank = cute::block_rank_in_cluster();
  if (tid == 256) {
    for (unsigned i = 0; i < Stages; ++i) {
      s.full[i].init(1);
      s.empty[i].init(rank == 0 ? 8 * cluster : 8);
      s.pull_full[i].init(1);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  cute::cluster_sync();
  if (tid < 256)
    cutlass::arch::warpgroup_reg_alloc<208>();
  else
    cutlass::arch::warpgroup_reg_dealloc<80>();

  // The dW phase uses twelve new barriers outside the output tile. Keeping
  // this second phase retains the production activation -> dW transition.
  auto *dw = reinterpret_cast<Barrier *>(arena + DwBarrierOffset);
  for (unsigned pass = 0; pass < 2; ++pass) {
    auto *full = pass == 0 ? s.full : dw + 6;
    auto *empty = pass == 0 ? s.empty : dw + 9;
    unsigned stage = 0, phase = 0;
    if (tid == 256) {
      for (unsigned i = 0; i < steps; ++i) {
        empty[stage].wait(phase ^ 1);
        hopper::tma_load<K, M, 128>(&input_a, &full[stage], s.a[stage], 0, 0,
                                    cluster);
        hopper::tma_load<K, N, 128>(&input_b, &full[stage], s.b[stage], 0, 0);
        full[stage].arrive_and_expect_tx((M + N) * K * 2);
        advance(stage, phase);
      }
      if (drain || pass == 1)
        for (unsigned i = 0; i < Stages; ++i) {
          empty[stage].wait(phase ^ 1);
          advance(stage, phase);
        }
    } else if (tid < 256) {
      float accum[Mma::kNumAccum];
      if constexpr (Gemm) {
#pragma unroll
        for (auto &v : accum)
          v = 0;
      }
      for (unsigned i = 0; i < steps; ++i) {
        full[stage].wait(phase);
        if constexpr (Gemm) {
#pragma unroll
          for (auto &v : accum)
            cute::warpgroup_fence_operand(v);
          cute::warpgroup_arrive();
#pragma unroll
          for (unsigned k = 0; k < K; k += Mma::K) {
            const auto da = hopper::make_gmma_desc<hopper::Major::K, M, K, 128>(
                s.a[stage], tid / 128 * 64, k);
            const auto db = hopper::make_gmma_desc<hopper::Major::K, N, K, 128>(
                s.b[stage], 0, k);
            Mma::wgmma(da, db, accum, i != 0 || k != 0);
          }
          cute::warpgroup_commit_batch();
          cute::warpgroup_wait<0>();
#pragma unroll
          for (auto &v : accum)
            cute::warpgroup_fence_operand(v);
        }
        __syncwarp();
        if (lane == 0)
          empty[stage].arrive();
        if (lane == 1 && rank != 0)
          empty[stage].arrive(0U, 1U);
        advance(stage, phase);
      }
      if constexpr (Gemm)
        probe[blockIdx.x * 256 + tid] = accum[0];
      if (pass == 1)
        write_tile(reinterpret_cast<BF16 *>(arena + OutputOffset), &output,
                   iteration);
    } else if (tid >= 288 && lane == 0 && pass == 0) {
      const unsigned warp = (tid - 288) / 32;
      for (unsigned i = 0; i < steps; ++i) {
        cute::SM90_BULK_COPY_G2S::copy(
            pull_input, reinterpret_cast<std::uint64_t *>(&s.pull_full[warp]),
            s.pull[warp], 4096);
        s.pull_full[warp].arrive_and_expect_tx(4096);
        s.pull_full[warp].wait(i & 1U);
      }
    }

    if (pass == 1)
      break;

    // Production's single-node path joins the activation grid before dW.
    // One resident CTA per SM is required, as for any software grid barrier.
    __syncthreads();
    if (tid == 0) {
      __threadfence();
      atomicAdd(joined, 1U);
      while (atomicAdd(joined, 0U) < gridDim.x) {
      }
    }
    __syncthreads();
    if (tid == 256 && invalidate) {
      for (auto &b : s.full)
        Barrier::invalidate(reinterpret_cast<std::uint64_t *>(&b));
      for (auto &b : s.empty)
        Barrier::invalidate(reinterpret_cast<std::uint64_t *>(&b));
      for (auto &b : s.pull_full)
        Barrier::invalidate(reinterpret_cast<std::uint64_t *>(&b));
    }
    if (tid == 256) {
      for (unsigned i = 0; i < 12; ++i)
        dw[i].init(i / 3 % 2 == 0 ? 1 : (rank == 0 ? 8 * cluster : 8));
      cutlass::arch::fence_barrier_init();
    }
    __syncthreads();
    cute::cluster_sync();
  }

  __syncthreads();
  cute::cluster_sync();
}

struct Result {
  unsigned long long mismatches, nonfinite, overlap;
  unsigned long long gemm_mismatches;
  unsigned first, observed, expected;
};

__global__ void check(const unsigned short *out, const float *probe,
                      unsigned size, unsigned iteration, unsigned steps,
                      bool gemm, Result *result) {
  if (gemm && probe[blockIdx.x * 256 + threadIdx.x] != float(steps * K))
    atomicAdd(&result->gemm_mismatches, 1ULL);
  for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    const unsigned block = i / (M * N), row = (i / N) % M, col = i % N;
    const unsigned short expected = value(row, col, block, iteration);
    const unsigned short observed = out[i];
    if (observed != expected) {
      atomicAdd(&result->mismatches, 1ULL);
      if ((observed & 0x7f80U) == 0x7f80U)
        atomicAdd(&result->nonfinite, 1ULL);
      // The nine old barriers occupy row 96, columns 0..35 in atom zero.
      if (row == 96 && col < 36)
        atomicAdd(&result->overlap, 1ULL);
      if (atomicCAS(&result->first, ~0U, i) == ~0U) {
        result->observed = observed;
        result->expected = expected;
      }
    }
  }
}

CUtensorMap tensor_map(BF16 *p, unsigned width, unsigned height, unsigned box) {
  CUtensorMap map{};
  const cuuint64_t dims[] = {width, height},
                   strides[] = {std::uint64_t(width) * 2};
  const cuuint32_t boxes[] = {64, box}, elements[] = {1, 1};
  if (cuTensorMapEncodeTiled(
          &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, p, dims, strides, boxes,
          elements, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
          CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
    throw std::runtime_error("cuTensorMapEncodeTiled failed");
  return map;
}

int main(int argc, char **argv) try {
  if (argc != 8 && argc != 9)
    throw std::runtime_error("usage: repro cluster ctas steps repeats drain "
                             "invalidate device [gemm]");
  const unsigned cluster = std::stoul(argv[1]),
                 requested_ctas = std::stoul(argv[2]);
  const unsigned steps = std::stoul(argv[3]), repeats = std::stoul(argv[4]);
  const bool drain = std::stoi(argv[5]), invalidate = std::stoi(argv[6]);
  const int device = std::stoi(argv[7]);
  const bool gemm = argc == 9 && std::stoi(argv[8]);
  CK(cudaSetDevice(device));
  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, device));
  if (cluster != 1 && cluster != 2)
    throw std::runtime_error("cluster must be 1 or 2");
  const unsigned ctas = requested_ctas
                            ? requested_ctas
                            : prop.multiProcessorCount / cluster * cluster;
  if (!ctas || ctas % cluster || ctas > unsigned(prop.multiProcessorCount) ||
      steps < Stages || !repeats)
    throw std::runtime_error(
        "invalid cluster, resident CTA count, steps, or repeats");
  BF16 *a, *b, *out;
  unsigned *joined;
  Result *result;
  float *probe;
  CK(cudaMalloc(&a, M * K * 2));
  CK(cudaMalloc(&b, N * K * 2));
  CK(cudaMalloc(&out, std::size_t(ctas) * M * N * 2));
  CK(cudaMalloc(&joined, sizeof(unsigned)));
  CK(cudaMalloc(&result, sizeof(Result)));
  CK(cudaMalloc(&probe, std::size_t(ctas) * 256 * sizeof(float)));
  fill<<<1, 256>>>(a, b);
  Result host{};
  host.first = ~0U;
  CK(cudaMemcpy(result, &host, sizeof(host), cudaMemcpyHostToDevice));
  const auto am = tensor_map(a, K, M, M), bm = tensor_map(b, K, N, N);
  const auto om = tensor_map(out, N, ctas * M, M);
  const auto kernel = gemm ? reuse<true> : reuse<false>;
  CK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                          SharedBytes));
  cudaFuncAttributes kernel_attr{};
  CK(cudaFuncGetAttributes(&kernel_attr, kernel));
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim = {cluster, 1, 1};
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(ctas);
  config.blockDim = dim3(Threads);
  config.dynamicSmemBytes = SharedBytes;
  config.attrs = &attr;
  config.numAttrs = 1;
  for (unsigned i = 0; i < repeats; ++i) {
    CK(cudaMemsetAsync(joined, 0, sizeof(unsigned)));
    CK(cudaMemsetAsync(out, 0, std::size_t(ctas) * M * N * 2));
    CK(cudaLaunchKernelEx(&config, kernel, am, bm, om, joined, a, probe, steps,
                          i, cluster, drain, invalidate));
    check<<<ctas, 256>>>(reinterpret_cast<unsigned short *>(out), probe,
                         ctas * M * N, i, steps, gemm, result);
    CK(cudaGetLastError());
  }
  CK(cudaMemcpy(&host, result, sizeof(host), cudaMemcpyDeviceToHost));
  std::printf(
      "{\"device\":\"%s\",\"cluster\":%u,\"ctas\":%u,\"steps\":%u,"
      "\"repeats\":%u,\"drain\":%d,\"invalidate\":%d,\"gemm\":%d,\"phases\":2,"
      "\"registers\":%d,\"local_bytes\":%zu,\"shared_bytes\":%u,"
      "\"mismatches\":%llu,\"nonfinite\":%llu,\"overlap\":%llu,\"gemm_"
      "mismatches\":%llu,"
      "\"first_index\":%u,\"observed\":%u,\"expected\":%u}\n",
      prop.name, cluster, ctas, steps, repeats, drain, invalidate, gemm,
      kernel_attr.numRegs, kernel_attr.localSizeBytes, SharedBytes,
      host.mismatches, host.nonfinite, host.overlap, host.gemm_mismatches,
      host.first, host.observed, host.expected);
  CK(cudaFree(probe));
  CK(cudaFree(result));
  CK(cudaFree(joined));
  CK(cudaFree(out));
  CK(cudaFree(b));
  CK(cudaFree(a));
  return host.mismatches || host.gemm_mismatches ? 2 : 0;
} catch (const std::exception &e) {
  std::fprintf(stderr, "%s\n", e.what());
  return 1;
}
