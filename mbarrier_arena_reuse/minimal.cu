// SPDX-License-Identifier: Apache-2.0
// A completed but non-invalidated mbarrier aliases ordinary shared data.
// Run two processes on the same GPU; no TMA, WGMMA, cluster, or remote writes.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

#define CK(call)                                                               \
  do {                                                                         \
    auto e = (call);                                                           \
    if (e != cudaSuccess)                                                      \
      throw std::runtime_error(std::string(#call) + ": " +                     \
                               cudaGetErrorString(e));                         \
  } while (0)

constexpr unsigned Barriers = 9, Offset = 159744, SharedBytes = 213120;
constexpr unsigned long long Pattern = 0x3c003b803b003a80ULL;

__device__ __forceinline__ unsigned long long timer() {
  unsigned long long v;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v));
  return v;
}

__global__ void reuse(bool invalidate, unsigned long long duration_ns,
                      unsigned long long *bad) {
  extern __shared__ __align__(16) unsigned char smem[];
  const unsigned base = unsigned(__cvta_generic_to_shared(smem + Offset));
  if (threadIdx.x == 0) {
    for (unsigned i = 0; i < Barriers; ++i) {
      unsigned addr = base + i * 8;
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(addr)
                   : "memory");
      asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(addr)
                   : "memory");
      asm volatile("{ .reg .pred ready; wait_phase:"
                   "mbarrier.try_wait.parity.shared::cta.b64 ready, [%0], 0;"
                   "@!ready bra wait_phase; }" ::"r"(addr)
                   : "memory");
      if (invalidate)
        asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(addr)
                     : "memory");
    }
  }
  __syncthreads();
  if (threadIdx.x < Barriers) {
    const unsigned addr = base + threadIdx.x * 8;
    const auto value = Pattern + threadIdx.x;
    asm volatile("st.volatile.shared.b64 [%0], %1;" ::"r"(addr), "l"(value)
                 : "memory");
  }
  __syncthreads();
  const auto end = timer() + duration_ns;
  if (threadIdx.x < Barriers) {
    const unsigned addr = base + threadIdx.x * 8;
    const auto expected = Pattern + threadIdx.x;
    do {
      unsigned long long observed;
      asm volatile("ld.volatile.shared.b64 %0, [%1];"
                   : "=l"(observed)
                   : "r"(addr)
                   : "memory");
      if (observed != expected) {
        atomicCAS(bad + blockIdx.x * Barriers + threadIdx.x, 0ULL, observed);
        break;
      }
    } while (timer() < end);
  }
}

int main(int argc, char **argv) try {
  if (argc != 5)
    throw std::runtime_error(
        "usage: minimal device invalidate repeats hold_ms");
  const int device = std::stoi(argv[1]);
  const bool invalidate = std::stoi(argv[2]);
  const unsigned repeats = std::stoul(argv[3]), hold_ms = std::stoul(argv[4]);
  CK(cudaSetDevice(device));
  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, device));
  const unsigned ctas = prop.multiProcessorCount;
  unsigned long long *bad;
  const auto bytes = ctas * Barriers * sizeof(unsigned long long);
  CK(cudaMalloc(&bad, bytes));
  CK(cudaMemset(bad, 0, bytes));
  CK(cudaFuncSetAttribute(reuse, cudaFuncAttributeMaxDynamicSharedMemorySize,
                          SharedBytes));
  for (unsigned i = 0; i < repeats; ++i)
    reuse<<<ctas, 32, SharedBytes>>>(invalidate, hold_ms * 1000000ULL, bad);
  CK(cudaGetLastError());
  std::vector<unsigned long long> host(ctas * Barriers);
  CK(cudaMemcpy(host.data(), bad, bytes, cudaMemcpyDeviceToHost));
  unsigned errors = 0;
  for (unsigned i = 0; i < host.size(); ++i) {
    if (!host[i])
      continue;
    if (errors < 4)
      std::printf("{\"cta\":%u,\"barrier\":%u,\"observed\":\"0x%016llx\","
                  "\"expected\":\"0x%016llx\"}\n",
                  i / Barriers, i % Barriers, host[i], Pattern + i % Barriers);
    ++errors;
  }
  std::printf("{\"device\":%d,\"ctas\":%u,\"invalidate\":%d,\"repeats\":%u,"
              "\"hold_ms\":%u,\"corrupted_slots\":%u}\n",
              device, ctas, invalidate, repeats, hold_ms, errors);
  CK(cudaFree(bad));
  return errors ? 2 : 0;
} catch (const std::exception &e) {
  std::fprintf(stderr, "%s\n", e.what());
  return 1;
}
