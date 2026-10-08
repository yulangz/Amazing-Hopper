// Immutable shared -> global store diagnostic; no GEMM, loads, or ring reuse.
// Extracted from WeftMoE (Apache-2.0); documentation reference only changed.
// See README.md. Dimensions are in BF16 elements; payload is raw bits.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#define CK(call)                                                                       \
  do {                                                                                 \
    const auto e = (call);                                                             \
    if (e != cudaSuccess)                                                              \
      throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(e));     \
  } while (0)

constexpr unsigned K = 64, Threads = 256;
enum Mode { None, Vector, TmaSerial, TmaQueued, ModeCount };
constexpr const char *Names[] = {"none", "vector", "tma_serial", "tma_queued"};

__device__ unsigned pattern(unsigned word, unsigned cta) {
  unsigned x = word ^ (cta * 0x9e3779b9U + 123U);
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  return x ^ (x >> 16);
}

template <unsigned Rows, unsigned Width>
__global__
__launch_bounds__(Threads) void store_repro(const __grid_constant__ CUtensorMap map,
                                            uint4 *output, unsigned tasks, Mode mode) {
  constexpr auto width = Width;
  __shared__ __align__(1024) unsigned source[Rows * K / 2];
  for (unsigned i = threadIdx.x; i < Rows * K / 2; i += Threads) {
    const auto r = i / (K / 2), col = i % (K / 2);
    source[r * (K / 2) + (col ^ ((r % 8U) * 4U))] = pattern(i, blockIdx.x);
  }
  // Every writer orders its generic SMEM stores before the async proxy.
  // This preparation is identical in every mode and paid once per CTA.
  cute::tma_store_fence();
  __syncthreads();
  if (mode == Vector) {
    for (unsigned task = blockIdx.x; task < tasks; task += gridDim.x) {
      const auto row = task / (width / K) * Rows;
      const auto col = task % (width / K) * K;
#pragma unroll
      for (unsigned j = 0; j < Rows * K / (8 * Threads); ++j) {
        const auto i = threadIdx.x + j * Threads;
        const auto r = i / (K / 8), c = i % (K / 8) * 8;
        const auto address = cute::cast_smem_ptr_to_uint(source + r * (K / 2) +
                                                         ((c / 2) ^ ((r % 8U) * 4U)));
        uint4 value;
        // PTX volatile is required too: asm volatile alone still lets ptxas
        // hoist these immutable shared loads out of the task loop.
        asm volatile("ld.volatile.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                     : "r"(address)
                     : "memory");
        output[(std::size_t(row + r) * width + col + c) / 8] = value;
      }
    }
  } else if (mode != None) {
    // Elect once around the whole loop: issue and all waits have one issuer,
    // while ptxas can avoid a per-copy lane-uniformization/peeling loop.
    const auto warp = __shfl_sync(0xffffffffU, threadIdx.x / 32U, 0);
    if (warp == 0U && cute::elect_one_sync()) {
      for (unsigned task = blockIdx.x; task < tasks; task += gridDim.x) {
        cute::SM90_TMA_STORE_2D::copy(&map, source, task % (width / K) * K,
                                      task / (width / K) * Rows);
        cute::tma_store_arrive();
        if (mode == TmaSerial)
          asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
        else
          asm volatile("cp.async.bulk.wait_group 7;" ::: "memory");
      }
      asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    }
  }
  __syncthreads();
}

__global__ void verify(const unsigned *output, std::size_t words, unsigned width,
                       unsigned rows, unsigned ctas, unsigned *errors) {
  for (auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < words;
       i += std::size_t(gridDim.x) * blockDim.x) {
    const auto row = i / (width / 2U), col = i % (width / 2U);
    const auto task = row / rows * (width / K) + col / (K / 2U);
    const auto word = row % rows * (K / 2U) + col % (K / 2U);
    if (output[i] != pattern(unsigned(word), unsigned(task % ctas)))
      atomicAdd(errors, 1U);
  }
}

int main(int argc, char **argv) try {
  // rows width output_MiB CTAs_per_SM samples device [mode]
  if (argc < 7 || argc > 8)
    throw std::runtime_error("usage: store_repro 32|64|256 64|4096 MiB ctas_per_sm "
                             "samples device [none|vector|tma_serial|tma_queued]");
  const unsigned rows = std::stoul(argv[1]), width = std::stoul(argv[2]);
  const auto bytes = std::stoull(argv[3]) * 1024U * 1024U;
  const unsigned blocks_per_sm = std::stoul(argv[4]), samples = std::stoul(argv[5]);
  if ((rows != 32 && rows != 64 && rows != 256) || (width != 64 && width != 4096) ||
      !bytes || !blocks_per_sm || !samples || bytes % (rows * width * 2U))
    throw std::runtime_error("invalid diagnostic dimensions");
  const auto tasks = unsigned(bytes / (rows * K * 2U));
  CK(cudaSetDevice(std::stoi(argv[6])));
  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, std::stoi(argv[6])));
  const unsigned ctas = prop.multiProcessorCount * blocks_per_sm;
  uint4 *output;
  unsigned *errors;
  CK(cudaMalloc(&output, bytes));
  CK(cudaMalloc(&errors, sizeof(unsigned)));
  CUtensorMap map{};
  const cuuint64_t dims[] = {width, bytes / (width * 2U)};
  const cuuint64_t strides[] = {std::uint64_t(width) * 2U};
  const cuuint32_t boxes[] = {K, rows}, elements[] = {1U, 1U};
  if (cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, output, dims,
                             strides, boxes, elements, CU_TENSOR_MAP_INTERLEAVE_NONE,
                             CU_TENSOR_MAP_SWIZZLE_128B,
                             CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
    throw std::runtime_error("cuTensorMapEncodeTiled failed");
  const auto kernel = width == 64 ? (rows == 32   ? store_repro<32, 64>
                                     : rows == 64 ? store_repro<64, 64>
                                                  : store_repro<256, 64>)
                                  : (rows == 32   ? store_repro<32, 4096>
                                     : rows == 64 ? store_repro<64, 4096>
                                                  : store_repro<256, 4096>);
  cudaFuncAttributes attr{};
  CK(cudaFuncGetAttributes(&attr, kernel));
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(ctas);
  config.blockDim = dim3(Threads);
  CK(cudaStreamCreateWithFlags(&config.stream, cudaStreamNonBlocking));
  const auto launch = [&](Mode mode) {
    CK(cudaLaunchKernelEx(&config, kernel, map, output, tasks, mode));
  };
  std::vector<Mode> modes;
  for (unsigned i = 0; i < ModeCount; ++i)
    if (argc == 7 || std::string(argv[7]) == Names[i])
      modes.push_back(Mode(i));
  if (modes.empty())
    throw std::runtime_error("invalid mode");
  for (auto mode : modes) {
    CK(cudaMemsetAsync(output, 0, bytes, config.stream));
    CK(cudaMemsetAsync(errors, 0, sizeof(unsigned), config.stream));
    launch(mode);
    if (mode != None)
      verify<<<256, 256, 0, config.stream>>>(reinterpret_cast<unsigned *>(output),
                                             bytes / 4, width, rows, ctas, errors);
    CK(cudaStreamSynchronize(config.stream));
    unsigned count;
    CK(cudaMemcpy(&count, errors, sizeof(count), cudaMemcpyDeviceToHost));
    if (count)
      throw std::runtime_error(std::string(Names[mode]) +
                               " mismatches: " + std::to_string(count));
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
        (sorted[(sorted.size() - 1) / 2] + sorted[sorted.size() / 2]) / 2;
    std::printf("STORE_REPRO {\"tile_bytes\":%u,\"width\":%u,\"bytes\":%llu,"
                "\"ctas\":%u,\"threads\":%u,\"registers\":%d,\"local_bytes\":%zu,"
                "\"shared_bytes\":%zu,\"mode\":\"%s\",\"samples\":%u,"
                "\"median_ms\":%.6f,\"store_GBps\":%.3f,\"pause_us\":2000,"
                "\"verified\":true,\"samples_ms\":[",
                rows * K * 2U, width, bytes, ctas, Threads, attr.numRegs,
                attr.localSizeBytes, attr.sharedSizeBytes, Names[mode], samples, median,
                mode == None ? 0.0 : bytes / (median * 1e6));
    for (unsigned i = 0; i < elapsed[mode].size(); ++i)
      std::printf("%s%.6f", i ? "," : "", elapsed[mode][i]);
    std::printf("]}\n");
  }
  CK(cudaEventDestroy(begin));
  CK(cudaEventDestroy(end));
  CK(cudaStreamDestroy(config.stream));
  CK(cudaFree(output));
  CK(cudaFree(errors));
  return 0;
} catch (const std::exception &e) {
  std::fprintf(stderr, "%s\n", e.what());
  return 1;
}
