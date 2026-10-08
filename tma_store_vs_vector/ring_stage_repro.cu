// Extracted from WeftMoE (Apache-2.0); local helper include/namespace only.
// Standalone SM90a Saved-X diagnostic; see README.md for commands.
// No Weft runtime, NCCL, route planner, FC2, or host allocations in the timer.
#include "hopper.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cutlass/arch/reg_reconfig.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using BF16 = cutlass::bfloat16_t;
using Barrier = cutlass::arch::ClusterTransactionBarrier;
using Mma = hopper::BF16Mma<256>;
constexpr unsigned M = 128, N = 256, K = 64, Hidden = 4096, Stages = 3;
constexpr unsigned MathThreads = 256, Threads = 384;
enum Mode { None, Vector, TmaFull, TmaRead, TmaPipelined, ModeCount };
constexpr const char *Names[] = {"none", "vector", "tma_full", "tma_read",
                                 "tma_pipelined"};

#define CK(call)                                                                       \
  do {                                                                                 \
    auto e = (call);                                                                   \
    if (e != cudaSuccess)                                                              \
      throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(e));     \
  } while (0)

struct Shared {
  alignas(1024) BF16 a[Stages][M * K];
  alignas(1024) BF16 b[Stages][N * K];
  // Preserve production occupancy/SMEM reservation, not its epilogue or Pull.
  alignas(1024) BF16 unused_output[M * N];
  alignas(1024) unsigned char unused_pull[3][4096];
  Barrier full[Stages], empty[Stages], unused_pull_full[3];
};
static_assert(sizeof(Shared) == 226304);

struct Args {
  BF16 *saved, *output;
  unsigned rows, experts, ntiles, tma_rows;
  bool fence;
};

__host__ __device__ float pattern(std::size_t i, unsigned salt) {
  unsigned x = unsigned(i) ^ salt;
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  return (int(x & 31U) - 16) / 32.0f;
}
__global__ void fill(BF16 *x, std::size_t size, unsigned salt) {
  for (auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < size;
       i += std::size_t(gridDim.x) * blockDim.x)
    x[i] = BF16(pattern(i, salt));
}
__global__ void compare(const BF16 *a, const BF16 *b, std::size_t size,
                        unsigned *errors) {
  for (auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < size;
       i += std::size_t(gridDim.x) * blockDim.x)
    if (reinterpret_cast<const unsigned short *>(a)[i] !=
        reinterpret_cast<const unsigned short *>(b)[i])
      atomicAdd(errors, 1U);
}

// All modes of each specialization use the SAME kernel binary. This removes
// per-mode register allocation/code-layout differences from the comparison.
template <bool Gemm, bool Elected>
__global__ __launch_bounds__(Threads, 1) void saved_x_repro(
    const __grid_constant__ CUtensorMap xm, const __grid_constant__ CUtensorMap wm,
    const __grid_constant__ CUtensorMap saved_map, Args args, Mode mode) {
  extern __shared__ __align__(1024) unsigned char storage[];
  auto &s = *reinterpret_cast<Shared *>(storage);
  const unsigned tid = threadIdx.x, lane = tid & 31U;
  if (tid == MathThreads) {
    for (unsigned stage = 0; stage < Stages; ++stage) {
      s.full[stage].init(1U);
      s.empty[stage].init(16U);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  cute::cluster_sync();
  if constexpr (Gemm) {
    if (tid < MathThreads)
      cutlass::arch::warpgroup_reg_alloc<208>();
    else
      cutlass::arch::warpgroup_reg_dealloc<80>();
  }
  unsigned stage = 0U, phase = 0U;
  const unsigned pairs = Gemm ? args.ntiles / 2U : 1U;
  const unsigned blocks = args.rows / M;
  for (unsigned work = blockIdx.x / 2U; work < blocks * pairs; work += gridDim.x / 2U) {
    const auto block = work / pairs, pair = work % pairs;
    const auto ntile = pair * 2U + (blockIdx.x & 1U);
    const auto row = block * M, expert = block * args.experts / blocks;
    const auto half = (ntile & 1U) * (M / 2U);
    const bool owner = mode != None && pair == block % pairs;
    if (tid == MathThreads) {
      for (unsigned k = 0; k < Hidden; k += K) {
        s.empty[stage].wait(phase ^ 1U);
        hopper::tma_load<K, M, 128U>(&xm, &s.full[stage], s.a[stage], k, row, 2U);
        if constexpr (Gemm)
          hopper::tma_load<K, N, 128U>(&wm, &s.full[stage], s.b[stage], k,
                                       (expert * args.ntiles + ntile) * N);
        s.full[stage].arrive_and_expect_tx((M + (Gemm ? N : 0U)) * K * 2U);
        stage = stage + 1U == Stages ? 0U : stage + 1U;
        phase ^= stage == 0U;
      }
    } else if (tid < MathThreads) {
      float accum[Mma::kNumAccum] = {};
      auto da = hopper::make_gmma_desc<hopper::Major::K, M, K, 128U>(
          s.a[0], tid / 128U * 64U, 0U);
      auto db = hopper::make_gmma_desc<hopper::Major::K, N, K, 128U>(s.b[0], 0U, 0U);
      const auto abase = __shfl_sync(0xffffffffU, da.reg32_[0], 0);
      const auto bbase = __shfl_sync(0xffffffffU, db.reg32_[0], 0);
      BF16 *destinations[2];
#pragma unroll
      for (unsigned i = 0; i < 2; ++i)
        destinations[i] =
            args.saved +
            std::size_t(row + half + (tid + i * MathThreads) / 8U) * Hidden;
      for (unsigned k = 0; k < Hidden; k += K) {
        s.full[stage].wait(phase);
        if constexpr (Gemm) {
#pragma unroll
          for (auto &value : accum)
            cute::warpgroup_fence_operand(value);
          cute::warpgroup_arrive();
#pragma unroll
          for (unsigned atom = 0; atom < K; atom += Mma::K) {
            da.reg32_[0] = abase + stage * (M * K / 8U) + atom / 8U;
            db.reg32_[0] = bbase + stage * (N * K / 8U) + atom / 8U;
            Mma::wgmma(da, db, accum, k != 0U || atom != 0U);
          }
          cute::warpgroup_commit_batch();
#pragma unroll
          for (auto &value : accum)
            cute::warpgroup_fence_operand(value);
        }
        if (owner) {
          if (mode == Vector) {
#pragma unroll
            for (unsigned i = 0; i < 2; ++i) {
              const auto r = half + (tid + i * MathThreads) / 8U;
              const auto column = (tid % 8U) * 8U;
              const auto offset = r * K + (column ^ ((r % 8U) * 8U));
              *reinterpret_cast<uint4 *>(destinations[i] + k + column) =
                  *reinterpret_cast<const uint4 *>(s.a[stage] + offset);
            }
          } else {
            const auto warp = __shfl_sync(0xffffffffU, tid / 32U, 0);
            if (warp == 0U && (Elected ? cute::elect_one_sync() : tid == 0U)) {
              if (args.fence)
                cute::tma_store_fence();
              cute::SM90_TMA_STORE_2D::copy(&saved_map, s.a[stage] + half * K, k,
                                            row + half);
              if (args.tma_rows == 32U)
                cute::SM90_TMA_STORE_2D::copy(&saved_map, s.a[stage] + (half + 32U) * K,
                                              k, row + half + 32U);
              cute::tma_store_arrive();
              if (mode == TmaFull)
                asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
              else if (mode == TmaRead)
                asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
              else if (k != 0U)
                asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");
            }
            if (tid < 32U)
              __syncwarp();
          }
        }
        if constexpr (Gemm)
          cute::warpgroup_wait<0>();
        const bool deferred = owner && mode == TmaPipelined;
        if (lane < 2U && (!deferred || k != 0U)) {
          const auto retired = deferred ? (stage + Stages - 1U) % Stages : stage;
          s.empty[retired].arrive(lane);
        }
        stage = stage + 1U == Stages ? 0U : stage + 1U;
        phase ^= stage == 0U;
      }
      if constexpr (Gemm) {
        const auto row0 = tid / 32U * 16U + lane / 4U;
#pragma unroll
        for (unsigned i = 0; i < N / 8U; ++i) {
          const auto col = ntile * N + i * 8U + (lane % 4U) * 2U;
          *reinterpret_cast<__nv_bfloat162 *>(
              args.output + std::size_t(row + row0) * args.ntiles * N + col) =
              __float22bfloat162_rn(make_float2(accum[i * 4U], accum[i * 4U + 1U]));
          *reinterpret_cast<__nv_bfloat162 *>(
              args.output + std::size_t(row + row0 + 8U) * args.ntiles * N + col) =
              __float22bfloat162_rn(
                  make_float2(accum[i * 4U + 2U], accum[i * 4U + 3U]));
        }
      }
      // Drain every possible issuer: elect does not promise the same lane
      // across iterations. Threads with no outstanding groups are harmless.
      if (owner && (mode == TmaRead || mode == TmaPipelined) && tid < 32U)
        asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
      if (owner && mode == TmaPipelined) {
        if (tid < 32U)
          __syncwarp();
        if (lane < 2U)
          s.empty[(stage + Stages - 1U) % Stages].arrive(lane);
      }
    }
  }
  // The partner's multicast producer must finish before shared memory exits.
  __syncthreads();
  cute::cluster_sync();
}

CUtensorMap tensor_map(BF16 *p, unsigned width, unsigned height, unsigned box) {
  CUtensorMap map{};
  const cuuint64_t dims[] = {width, height}, strides[] = {std::uint64_t(width) * 2U};
  const cuuint32_t boxes[] = {64U, box}, elements[] = {1U, 1U};
  if (cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, p, dims,
                             strides, boxes, elements, CU_TENSOR_MAP_INTERLEAVE_NONE,
                             CU_TENSOR_MAP_SWIZZLE_128B,
                             CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
    throw std::runtime_error("cuTensorMapEncodeTiled failed");
  return map;
}

int main(int argc, char **argv) try {
  // Positional args deliberately keep the repro dependency-free.
  // copy|gemm rows lane0|elect tma_rows fence samples device [single_mode]
  if (argc < 8 || argc > 9)
    throw std::runtime_error(
        "usage: repro copy|gemm rows lane0|elect 32|64 fence samples device "
        "[none|vector|tma_full|tma_read|tma_pipelined]");
  const bool gemm = std::string(argv[1]) == "gemm",
             elected = std::string(argv[3]) == "elect";
  const unsigned rows = std::stoul(argv[2]), tma_rows = std::stoul(argv[4]);
  const bool fence = std::stoi(argv[5]) != 0;
  const unsigned samples = std::stoul(argv[6]);
  if ((std::string(argv[1]) != "copy" && !gemm) ||
      (std::string(argv[3]) != "lane0" && !elected) || !rows || rows % M ||
      (tma_rows != 32 && tma_rows != 64) || !samples)
    throw std::runtime_error("invalid diagnostic arguments");
  CK(cudaSetDevice(std::stoi(argv[7])));
  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, std::stoi(argv[7])));
  const unsigned ctas = unsigned(prop.multiProcessorCount) / 2U * 2U;
  constexpr unsigned experts = 32U, ntiles = 16U;
  const auto nx = std::size_t(rows) * Hidden;
  const auto nw = std::size_t(experts) * ntiles * N * Hidden;
  const auto nc = gemm ? std::size_t(rows) * ntiles * N : 1U;
  BF16 *x, *w, *saved, *out, *reference;
  unsigned *errors;
  CK(cudaMalloc(&x, nx * 2));
  CK(cudaMalloc(&w, nw * 2));
  CK(cudaMalloc(&saved, nx * 2));
  CK(cudaMalloc(&out, nc * 2));
  CK(cudaMalloc(&reference, nc * 2));
  CK(cudaMalloc(&errors, sizeof(unsigned)));
  fill<<<256, 256>>>(x, nx, 123U);
  fill<<<256, 256>>>(w, nw, 987U);
  auto xm = tensor_map(x, Hidden, rows, M);
  auto wm = tensor_map(w, Hidden, experts * ntiles * N, N);
  auto sm = tensor_map(saved, Hidden, rows, tma_rows);
  const Args args{saved, out, rows, experts, ntiles, tma_rows, fence};
  const auto kernel =
      gemm ? (elected ? saved_x_repro<true, true> : saved_x_repro<true, false>)
           : (elected ? saved_x_repro<false, true> : saved_x_repro<false, false>);
  CK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                          sizeof(Shared)));
  cudaFuncAttributes attr{};
  CK(cudaFuncGetAttributes(&attr, kernel));
  cudaLaunchAttribute cluster{};
  cluster.id = cudaLaunchAttributeClusterDimension;
  cluster.val.clusterDim = {2, 1, 1};
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(ctas);
  config.blockDim = dim3(Threads);
  config.dynamicSmemBytes = sizeof(Shared);
  config.attrs = &cluster;
  config.numAttrs = 1;
  CK(cudaStreamCreateWithFlags(&config.stream, cudaStreamNonBlocking));
  CK(cudaDeviceSynchronize());
  auto launch = [&](Mode mode) {
    CK(cudaLaunchKernelEx(&config, kernel, xm, wm, sm, args, mode));
  };
  std::vector<Mode> modes;
  for (unsigned i = 0; i < ModeCount; ++i)
    // wait_group.read 1 needs a stable issuer across K iterations.
    if (!(elected && i == TmaPipelined) &&
        (argc == 8 || std::string(argv[8]) == Names[i]))
      modes.push_back(Mode(i));
  if (modes.empty())
    throw std::runtime_error("invalid single_mode");
  // Numerical validation is outside timing. A no-copy GEMM is the exact
  // reference, with independent CPU dot products checking its element layout.
  if (gemm) {
    launch(None);
    CK(cudaStreamSynchronize(config.stream));
    CK(cudaMemcpy(reference, out, nc * 2, cudaMemcpyDeviceToDevice));
    for (unsigned sample = 0; sample < 32; ++sample) {
      const auto row = unsigned(std::size_t(sample) * (rows - 1U) / 31U);
      const auto col = (sample * 137U) % (ntiles * N);
      const auto expert = (row / M) * experts / (rows / M);
      float expected = 0.0f;
      for (unsigned k = 0; k < Hidden; ++k)
        expected +=
            pattern(std::size_t(row) * Hidden + k, 123U) *
            pattern((std::size_t(expert) * ntiles * N + col) * Hidden + k, 987U);
      BF16 actual;
      CK(cudaMemcpy(&actual, reference + std::size_t(row) * ntiles * N + col, 2,
                    cudaMemcpyDeviceToHost));
      if (float(actual) != float(BF16(expected)))
        throw std::runtime_error("CPU GEMM reference mismatch");
    }
  }
  for (auto mode : modes) {
    CK(cudaMemsetAsync(saved, 0xff, nx * 2, config.stream));
    CK(cudaMemsetAsync(errors, 0, sizeof(unsigned), config.stream));
    launch(mode);
    if (mode != None)
      compare<<<256, 256, 0, config.stream>>>(x, saved, nx, errors);
    if (gemm)
      compare<<<256, 256, 0, config.stream>>>(reference, out, nc, errors);
    CK(cudaStreamSynchronize(config.stream));
    unsigned count;
    CK(cudaMemcpy(&count, errors, sizeof(count), cudaMemcpyDeviceToHost));
    if (count)
      throw std::runtime_error("checkpoint/GEMM mismatch: " + std::string(Names[mode]));
  }
  cudaEvent_t begin, end;
  CK(cudaEventCreate(&begin));
  CK(cudaEventCreate(&end));
  std::vector<float> elapsed[ModeCount];
  for (unsigned round = 0; round < samples + 8U; ++round) {
    auto order = modes;
    if (round & 1U)
      std::reverse(order.begin(), order.end());
    for (auto mode : order) {
      // Pure FC1 is more power-dense than PF. Pace outside the CUDA events
      // so repeated launches do not hit the board power cap at 1620 MHz.
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      CK(cudaEventRecord(begin, config.stream));
      launch(mode);
      CK(cudaEventRecord(end, config.stream));
      CK(cudaEventSynchronize(end));
      float ms;
      CK(cudaEventElapsedTime(&ms, begin, end));
      if (round >= 8U)
        elapsed[mode].push_back(ms);
    }
  }
  for (auto mode : modes) {
    auto sorted = elapsed[mode];
    std::sort(sorted.begin(), sorted.end());
    const auto median =
        (sorted[(sorted.size() - 1) / 2] + sorted[sorted.size() / 2]) * 0.5f;
    std::printf("SAVED_X_REPRO "
                "{\"op\":\"%s\",\"rows\":%u,\"issuer\":\"%s\",\"tma_rows\":%u,"
                "\"fence\":%s,\"mode\":\"%s\",\"ctas\":%u,\"registers\":%d,\"local_"
                "bytes\":%zu,\"shared_bytes\":%zu,\"samples\":%u,\"median_ms\":%.6f,"
                "\"pause_us\":2000,\"verified\":true,\"samples_ms\":[",
                gemm ? "gemm" : "copy", rows, elected ? "elect" : "lane0", tma_rows,
                fence ? "true" : "false", Names[mode], ctas, attr.numRegs,
                attr.localSizeBytes, sizeof(Shared), samples, median);
    for (unsigned i = 0; i < elapsed[mode].size(); ++i)
      std::printf("%s%.6f", i ? "," : "", elapsed[mode][i]);
    std::printf("]}\n");
  }
  CK(cudaEventDestroy(begin));
  CK(cudaEventDestroy(end));
  CK(cudaStreamDestroy(config.stream));
  for (auto p : {x, w, saved, out, reference})
    CK(cudaFree(p));
  CK(cudaFree(errors));
  return 0;
} catch (const std::exception &e) {
  std::fprintf(stderr, "%s\n", e.what());
  return 1;
}
