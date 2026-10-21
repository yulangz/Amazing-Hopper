// SPDX-License-Identifier: Apache-2.0
// Single-context experiment: touch other barriers after reusing one address.
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
struct Result { U64 before, after; unsigned first_change, guards_bad; };

__device__ __forceinline__ U64 read_shared(unsigned addr) {
  U64 v;
  asm volatile("ld.volatile.shared.b64 %0, [%1];" : "=l"(v) : "r"(addr) : "memory");
  return v;
}
__device__ __forceinline__ void complete_barrier(unsigned addr) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(addr) : "memory");
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(addr) : "memory");
  asm volatile("{ .reg .pred p; again: "
               "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;"
               "@!p bra again; }" :: "r"(addr) : "memory");
}
// mode 0: never initialize target; 1: reuse without inval; 2: inval before reuse.
// mode 3: ordinary stores as pressure; 4: invalidate each pressure barrier.
__global__ void probe(unsigned mode, unsigned n, unsigned stride, Result *out) {
  extern __shared__ __align__(16) unsigned char shared[];
  if (threadIdx.x != 0) return;
  unsigned base = unsigned(__cvta_generic_to_shared(shared));
  unsigned target = base + 8;
  if (mode) complete_barrier(target);
  if (mode == 2)
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(target) : "memory");
  for (unsigned i = 0; i < 3; ++i) {
    unsigned addr = base + i * 8;
    U64 v = Pattern + i;
    asm volatile("st.volatile.shared.b64 [%0], %1;" :: "r"(addr), "l"(v) : "memory");
  }
  Result r{read_shared(target), Pattern + 1, 0, 0};
  for (unsigned i = 0; i < n; ++i) {
    unsigned other = base + 32 + i * stride;
    if (mode == 3) {
      U64 data = Pattern + i;
      asm volatile("st.volatile.shared.b64 [%0], %1;" :: "r"(other), "l"(data) : "memory");
    } else {
      complete_barrier(other);
      if (mode == 4)
        asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(other) : "memory");
    }
    U64 v = read_shared(target);
    if (v != Pattern + 1 && !r.first_change) r.first_change = i + 1;
  }
  r.after = read_shared(target);
  r.guards_bad = (read_shared(base) != Pattern) + (read_shared(base + 16) != Pattern + 2);
  *out = r;
}

int main(int argc, char **argv) try {
  if (argc != 2) throw std::runtime_error("usage: cache_probe device");
  CK(cudaSetDevice(std::stoi(argv[1])));
  constexpr unsigned Repeats = 20, MaxShared = 32 + 1024 * 128;
  CK(cudaFuncSetAttribute(probe, cudaFuncAttributeMaxDynamicSharedMemorySize, MaxShared));
  Result *gpu;
  CK(cudaMalloc(&gpu, Repeats * sizeof(Result)));
  std::vector<Result> host(Repeats);
  for (unsigned mode : {0U, 1U, 2U, 3U, 4U})
    for (unsigned stride : {8U, 16U, 32U, 128U})
      for (unsigned n : {0U, 1U, 2U, 4U, 8U, 15U, 16U, 30U, 31U, 32U, 64U, 128U, 256U, 1024U}) {
        for (unsigned i = 0; i < Repeats; ++i)
          probe<<<1, 32, 32 + n * stride>>>(mode, n, stride, gpu + i);
        CK(cudaGetLastError());
        CK(cudaMemcpy(host.data(), gpu, Repeats * sizeof(Result), cudaMemcpyDeviceToHost));
        unsigned bad = 0, initial_bad = 0, guards = 0, first = 0, last = 0;
        U64 observed = Pattern + 1;
        for (const auto &r : host) {
          initial_bad += r.before != Pattern + 1;
          guards += r.guards_bad;
          if (r.after != Pattern + 1 || r.first_change) {
            ++bad;
            observed = r.after;
            if (!first || (r.first_change && r.first_change < first)) first = r.first_change;
            if (r.first_change > last) last = r.first_change;
          }
        }
        std::printf("{\"mode\":%u,\"stride\":%u,\"other_barriers\":%u,\"repeats\":%u,"
                    "\"initial_bad\":%u,\"corrupted_kernels\":%u,\"guards_bad\":%u,"
                    "\"first_change_min\":%u,\"first_change_max\":%u,\"observed\":\"0x%016llx\"}\n",
                    mode, stride, n, Repeats, initial_bad, bad, guards, first, last, observed);
      }
  CK(cudaFree(gpu));
  return 0; // This diagnostic reports observations, including negative results.
} catch (const std::exception &e) {
  std::fprintf(stderr, "%s\n", e.what());
  return 1;
}
