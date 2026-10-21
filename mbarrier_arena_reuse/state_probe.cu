// SPDX-License-Identifier: Apache-2.0
// Diagnostic: does corruption track the retired barrier's count and phase?
// One CTA, one barrier, two ordinary-data guards; no raw reads of a live barrier.
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#define CK(call) do { auto e = (call); if (e != cudaSuccess) \
  throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(e)); \
} while (0)

using U64 = unsigned long long;
constexpr U64 Pattern = 0x3c003b803b003a80ULL;
struct Sample {
  U64 observed, gap_ns, elapsed_ns, good_reads;
  unsigned guards_bad;
};

__device__ __forceinline__ U64 timer() {
  U64 v;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v));
  return v;
}
__device__ __forceinline__ U64 read_shared(unsigned addr) {
  U64 v;
  asm volatile("ld.volatile.shared.b64 %0, [%1];" : "=l"(v) : "r"(addr) : "memory");
  return v;
}

// mode: 0 = ordinary data only, 1 = old protocol, 2 = inval, 3 = fence only.
__global__ void probe(unsigned mode, unsigned count, unsigned phases,
                      U64 duration, Sample *out) {
  __shared__ __align__(8) U64 words[3];
  if (threadIdx.x != 0) return;
  unsigned addr = unsigned(__cvta_generic_to_shared(words + 1));
  if (mode) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :: "r"(addr), "r"(count) : "memory");
    for (unsigned p = 0; p < phases; ++p) {
      for (unsigned i = 0; i < count; ++i)
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                     :: "r"(addr) : "memory");
      asm volatile("{ .reg .pred ready; wait_phase: "
                   "mbarrier.try_wait.parity.shared::cta.b64 ready, [%0], %1;"
                   "@!ready bra wait_phase; }"
                   :: "r"(addr), "r"(p & 1) : "memory");
    }
    if (mode == 2)
      asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(addr) : "memory");
    if (mode == 3)
      asm volatile("fence.sc.cta;" ::: "memory");
  }
  for (unsigned i = 0; i < 3; ++i) {
    const unsigned a = addr - 8 + i * 8;
    const U64 v = Pattern + i;
    asm volatile("st.volatile.shared.b64 [%0], %1;" :: "r"(a), "l"(v) : "memory");
  }
  Sample result{};
  const U64 start = timer();
  U64 prev = start;
  do {
    U64 v = read_shared(addr);
    U64 now = timer();
    if (v != Pattern + 1) {
      result.observed = v;
      result.gap_ns = now - prev;
      result.elapsed_ns = now - start;
      break;
    }
    ++result.good_reads;
    prev = now;
  } while (prev - start < duration);
  result.guards_bad = (read_shared(addr - 8) != Pattern) +
                      (read_shared(addr + 8) != Pattern + 2);
  *out = result;
}

int main(int argc, char **argv) try {
  if (argc != 7)
    throw std::runtime_error("usage: state_probe device mode count phases repeats hold_ms");
  int device = std::stoi(argv[1]);
  unsigned mode = std::stoul(argv[2]), count = std::stoul(argv[3]);
  unsigned phases = std::stoul(argv[4]), repeats = std::stoul(argv[5]);
  unsigned hold = std::stoul(argv[6]);
  if (mode > 3 || !count || count >= (1U << 20) || !phases || !repeats || !hold)
    throw std::runtime_error("invalid arguments");
  CK(cudaSetDevice(device));
  Sample *gpu;
  CK(cudaMalloc(&gpu, repeats * sizeof(Sample)));
  for (unsigned i = 0; i < repeats; ++i)
    probe<<<1, 32>>>(mode, count, phases, hold * 1000000ULL, gpu + i);
  CK(cudaGetLastError());
  std::vector<Sample> samples(repeats);
  CK(cudaMemcpy(samples.data(), gpu, repeats * sizeof(Sample), cudaMemcpyDeviceToHost));
  unsigned bad = 0, guards = 0;
  for (unsigned i = 0; i < repeats; ++i) {
    const auto &s = samples[i];
    guards += s.guards_bad;
    if (!s.observed) continue;
    if (bad < 4)
      std::printf("{\"iteration\":%u,\"observed\":\"0x%016llx\",\"gap_ns\":%llu,"
                  "\"elapsed_ns\":%llu,\"good_reads\":%llu}\n",
                  i, s.observed, s.gap_ns, s.elapsed_ns, s.good_reads);
    ++bad;
  }
  std::printf("{\"device\":%d,\"mode\":%u,\"count\":%u,\"phases\":%u,"
              "\"ctas\":1,\"shared_bytes\":24,\"repeats\":%u,\"hold_ms\":%u,"
              "\"corrupted_kernels\":%u,\"corrupted_guards\":%u}\n",
              device, mode, count, phases, repeats, hold, bad, guards);
  CK(cudaFree(gpu));
  return bad || guards ? 2 : 0;
} catch (const std::exception &e) {
  std::fprintf(stderr, "%s\n", e.what());
  return 1;
}
