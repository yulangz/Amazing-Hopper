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
  unsigned x =
      row * 73856093U ^ col * 19349663U ^ block * 83492791U ^ iteration;
  x ^= x >> 13;
  x *= 0x85ebca6bU;
  return (0x3780U + (x & 1023U)) | ((x >> 16) & 0x8000U);
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
                                           unsigned iteration, unsigned tile) {
  const unsigned tid = threadIdx.x, lane = tid % 32;
  const unsigned block = tile * gridDim.x + blockIdx.x;
  if (tid < N / 64)
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  asm volatile("bar.sync 1, 256;" ::: "memory");
  // Same M128/N256, 128-B swizzle, STSM.x2 -> fence -> TMA-store epilogue.
  if (tid < 256) {
    const unsigned warp = tid / 32, row = warp * 16 + lane / 4;
#pragma unroll
    for (unsigned i = 0; i < N / 8; ++i) {
      const unsigned col = i * 8 + (lane % 4) * 2;
      unsigned a = value(row, col, block, iteration) |
                   (unsigned(value(row, col + 1, block, iteration)) << 16);
      unsigned b = value(row + 8, col, block, iteration) |
                   (unsigned(value(row + 8, col + 1, block, iteration)) << 16);
      const unsigned atom = i / 8, in_atom = i % 8;
      auto *ptr = reinterpret_cast<unsigned char *>(out) + atom * M * 128 +
                  (warp * 16 + lane) * 128 + (in_atom ^ (lane % 8)) * 16;
      stmatrix_x2(ptr, a, b);
    }
    cute::tma_store_fence();
  }
  asm volatile("bar.sync 2, 256;" ::: "memory");
  if (tid < N / 64) {
    cute::SM90_TMA_STORE_2D::copy(output, out + tid * M * 64, tid * 64,
                                  block * M);
    cute::tma_store_arrive();
  }
  __syncwarp();
}

__device__ __forceinline__ void
compute_join(std::uint64_t *word, unsigned &phase, unsigned cluster) {
  asm volatile("bar.sync 3, 288;" ::: "memory");
  if (threadIdx.x == 0) {
    const unsigned local = cute::cast_smem_ptr_to_uint(word);
    asm volatile(
        "mbarrier.arrive.release.cluster.shared::cta.b64 _, [%0];" ::"r"(local)
        : "memory");
    if (cluster == 2) {
      const unsigned peer = cute::block_rank_in_cluster() ^ 1U;
      asm volatile(
          "{ .reg .b32 remote; mapa.shared::cluster.u32 remote, %0, %1;"
          "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remote]; "
          "}" ::"r"(local),
          "r"(peer)
          : "memory");
    }
    asm volatile("{ .reg .pred ready; wait_phase:"
                 "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 "
                 "ready, [%0], %1;"
                 "@!ready bra wait_phase; }" ::"r"(local),
                 "r"(phase)
                 : "memory");
  }
  phase ^= 1U;
  asm volatile("bar.sync 3, 288;" ::: "memory");
}

template <bool Activation, bool Gemm>
__device__ __forceinline__ void
pipeline(Shared &s, unsigned char *arena, const CUtensorMap *input_a,
         const CUtensorMap *input_b, const CUtensorMap *output, float *probe,
         unsigned steps, unsigned tiles, unsigned iteration, unsigned cluster,
         bool drain, std::uint64_t *join) {
  constexpr auto MajorA = Activation ? hopper::Major::K : hopper::Major::MN;
  using Mma = hopper::BF16Mma<N, MajorA, hopper::Major::MN>;
  auto *dw = reinterpret_cast<Barrier *>(arena + DwBarrierOffset);
  auto *full = Activation ? s.full : dw + 6;
  auto *empty = Activation ? s.empty : dw + 9;
  const unsigned tid = threadIdx.x, lane = tid % 32;
  const unsigned rank = cute::block_rank_in_cluster();
  unsigned stage = 0, phase = 0, join_phase = 0;
  for (unsigned first = 0; first < tiles; first += 4) {
    const unsigned end = min(first + 4, tiles);
    const unsigned count =
        steps + (Activation ? 0 : (blockIdx.x / cluster + first / 4) % 5);
    if constexpr (!Activation) {
      compute_join(join, join_phase, cluster);
      compute_join(join, join_phase, cluster);
    }
    if (tid == 256) {
      for (unsigned tile = first; tile < end; ++tile)
        for (unsigned i = 0; i < count; ++i) {
          empty[stage].wait(phase ^ 1);
          BF16 *a = Activation
                        ? s.a[stage]
                        : reinterpret_cast<BF16 *>(arena) + stage * (M + N) * K;
          BF16 *b = Activation ? s.b[stage] : a + M * K;
          if constexpr (Activation)
            hopper::tma_load<K, M, 128>(input_a, &full[stage], a, 0, 0,
                                        cluster);
          else
            hopper::tma_load<M, K, 128>(input_a, &full[stage], a, 0, 0,
                                        cluster);
          hopper::tma_load<N, K, 128>(input_b, &full[stage], b, 0, 0);
          full[stage].arrive_and_expect_tx((M + N) * K * 2);
          advance(stage, phase);
        }
      if constexpr (Activation)
        if (drain)
          for (unsigned i = 0; i < Stages; ++i) {
            empty[stage].wait(phase ^ 1);
            advance(stage, phase);
          }
    } else if (tid < 256) {
      for (unsigned tile = first; tile < end; ++tile) {
        float accum[Mma::kNumAccum] = {};
        for (unsigned i = 0; i < count; ++i) {
          full[stage].wait(phase);
          if constexpr (Gemm) {
            BF16 *a = Activation ? s.a[stage]
                                 : reinterpret_cast<BF16 *>(arena) +
                                       stage * (M + N) * K;
            BF16 *b = Activation ? s.b[stage] : a + M * K;
#pragma unroll
            for (auto &v : accum)
              cute::warpgroup_fence_operand(v);
            cute::warpgroup_arrive();
#pragma unroll
            for (unsigned k = 0; k < K; k += Mma::K) {
              const auto da = hopper::make_gmma_desc<MajorA, M, K, 128>(
                  a, tid / 128 * 64, k);
              const auto db =
                  hopper::make_gmma_desc<hopper::Major::MN, N, K, 128>(b, 0, k);
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
        if constexpr (!Activation)
          write_tile(reinterpret_cast<BF16 *>(arena + OutputOffset), output,
                     iteration, tile);
      }
    }
    if constexpr (!Activation) {
      asm volatile("bar.sync 3, 288;" ::: "memory");
      if (tid < N / 64)
        asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
      asm volatile("bar.sync 3, 288;" ::: "memory");
    }
  }
  if constexpr (!Activation)
    compute_join(join, join_phase, cluster);
}

template <bool Gemm>
__global__ __launch_bounds__(Threads, 1) void reuse(
    const __grid_constant__ CUtensorMap input_a,
    const __grid_constant__ CUtensorMap input_b,
    const __grid_constant__ CUtensorMap dw_a,
    const __grid_constant__ CUtensorMap output, unsigned *joined,
    const BF16 *pull_input, float *probe, unsigned steps, unsigned iteration,
    unsigned cluster, bool drain, bool invalidate, unsigned tiles) {
  extern __shared__ __align__(1024) unsigned char arena[];
  __shared__ alignas(16) std::uint64_t task_join;
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
  if (tid < 288)
    pipeline<true, Gemm>(s, arena, &input_a, &input_b, &output, probe, steps, 1,
                         iteration, cluster, drain, &task_join);
  else if (lane == 0) {
    const unsigned warp = (tid - 288) / 32;
    for (unsigned i = 0; i < steps; ++i) {
      cute::SM90_BULK_COPY_G2S::copy(
          pull_input, reinterpret_cast<std::uint64_t *>(&s.pull_full[warp]),
          s.pull[warp], 4096);
      s.pull_full[warp].arrive_and_expect_tx(4096);
      s.pull_full[warp].wait(i & 1U);
    }
  }
  __threadfence_system();
  __syncthreads();
  if (tid == 0) {
    atomicAdd(joined, 1U);
    while (atomicAdd(joined, 0U) < gridDim.x) {
    }
  }
  __syncthreads();
  if (tid == 256) {
    if (invalidate) {
      for (auto &b : s.full)
        Barrier::invalidate(reinterpret_cast<std::uint64_t *>(&b));
      for (auto &b : s.empty)
        Barrier::invalidate(reinterpret_cast<std::uint64_t *>(&b));
      for (auto &b : s.pull_full)
        Barrier::invalidate(reinterpret_cast<std::uint64_t *>(&b));
    }
    auto *dw = reinterpret_cast<Barrier *>(arena + DwBarrierOffset);
    for (unsigned i = 0; i < 12; ++i)
      dw[i].init(i / 3 % 2 == 0 ? 1 : (rank == 0 ? 8 * cluster : 8));
    cutlass::arch::fence_barrier_init();
  }
  if (tid == 0) {
    reinterpret_cast<cutlass::arch::ClusterBarrier *>(&task_join)
        ->init(cluster);
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  cute::cluster_sync();
  if (tid < 288)
    pipeline<false, Gemm>(s, arena, &dw_a, &input_b, &output, probe, steps,
                          tiles, iteration, cluster, false, &task_join);
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
                      bool gemm, unsigned tiles, unsigned cluster,
                      Result *result) {
  if (gemm &&
      probe[blockIdx.x * 256 + threadIdx.x] !=
          float((steps + (blockIdx.x / cluster + (tiles - 1) / 4) % 5) * K))
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
  if (argc < 8 || argc > 10)
    throw std::runtime_error("usage: repro cluster ctas steps repeats drain "
                             "invalidate device [gemm] [tiles]");
  const unsigned cluster = std::stoul(argv[1]),
                 requested_ctas = std::stoul(argv[2]);
  const unsigned steps = std::stoul(argv[3]), repeats = std::stoul(argv[4]);
  const bool drain = std::stoi(argv[5]), invalidate = std::stoi(argv[6]);
  const int device = std::stoi(argv[7]);
  const bool gemm = argc >= 9 && std::stoi(argv[8]);
  const unsigned tiles = argc == 10 ? std::stoul(argv[9]) : 32;
  CK(cudaSetDevice(device));
  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, device));
  if (cluster != 1 && cluster != 2)
    throw std::runtime_error("cluster must be 1 or 2");
  const unsigned ctas = requested_ctas
                            ? requested_ctas
                            : prop.multiProcessorCount / cluster * cluster;
  if (!ctas || ctas % cluster || ctas > unsigned(prop.multiProcessorCount) ||
      steps < Stages || !repeats || !tiles)
    throw std::runtime_error(
        "invalid cluster, resident CTA count, steps, or repeats");
  BF16 *a, *b, *out;
  unsigned *joined;
  Result *result;
  float *probe;
  CK(cudaMalloc(&a, M * K * 2));
  CK(cudaMalloc(&b, N * K * 2));
  CK(cudaMalloc(&out, std::size_t(ctas) * tiles * M * N * 2));
  CK(cudaMalloc(&joined, sizeof(unsigned)));
  CK(cudaMalloc(&result, sizeof(Result)));
  CK(cudaMalloc(&probe, std::size_t(ctas) * 256 * sizeof(float)));
  fill<<<1, 256>>>(a, b);
  Result host{};
  host.first = ~0U;
  CK(cudaMemcpy(result, &host, sizeof(host), cudaMemcpyHostToDevice));
  const auto am = tensor_map(a, K, M, M), bm = tensor_map(b, N, K, K);
  const auto dm = tensor_map(a, M, K, K);
  const auto om = tensor_map(out, N, ctas * tiles * M, M);
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
    CK(cudaMemsetAsync(out, 0, std::size_t(ctas) * tiles * M * N * 2));
    CK(cudaLaunchKernelEx(&config, kernel, am, bm, dm, om, joined, a, probe,
                          steps, i, cluster, drain, invalidate, tiles));
    check<<<ctas, 256>>>(reinterpret_cast<unsigned short *>(out), probe,
                         ctas * tiles * M * N, i, steps, gemm, tiles, cluster,
                         result);
    CK(cudaGetLastError());
  }
  CK(cudaMemcpy(&host, result, sizeof(host), cudaMemcpyDeviceToHost));
  std::printf(
      "{\"device\":\"%s\",\"cluster\":%u,\"ctas\":%u,\"steps\":%u,"
      "\"repeats\":%u,\"drain\":%d,\"invalidate\":%d,\"gemm\":%d,\"phases\":2,"
      "\"tiles\":%u,\"registers\":%d,\"local_bytes\":%zu,\"shared_bytes\":%u,"
      "\"mismatches\":%llu,\"nonfinite\":%llu,\"overlap\":%llu,\"gemm_"
      "mismatches\":%llu,"
      "\"first_index\":%u,\"observed\":%u,\"expected\":%u}\n",
      prop.name, cluster, ctas, steps, repeats, drain, invalidate, gemm, tiles,
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
