#pragma once

// Extracted from WeftMoE (Apache-2.0), trimmed to this diagnostic.
// Descriptor/address and array adapters derived from DeepGEMM (MIT).
// See THIRD_PARTY_NOTICES.md.
// Instructions are provided by CuTe/CUTLASS.
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm90_desc.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cutlass/arch/barrier.h>

namespace hopper {
using Major = cute::SM90::GMMA::Major;

template <int Columns, Major MajorA = Major::K, Major MajorB = Major::K,
          uint32_t StageK = 64, uint32_t SwizzleBytes = 128>
struct BF16Mma {
  static constexpr int M = 64, N = Columns, K = 16;
  static constexpr int kNumAccum = Columns / 2;
  static constexpr auto operation() {
    using namespace cute::SM90::GMMA;
    static_assert(Columns == 64 || Columns == 128 || Columns == 256);
    if constexpr (Columns == 64)
      return MMA_64x64x16_F32BF16BF16_SS<MajorA, MajorB>{};
    else if constexpr (Columns == 128)
      return MMA_64x128x16_F32BF16BF16_SS<MajorA, MajorB>{};
    else
      return MMA_64x256x16_F32BF16BF16_SS<MajorA, MajorB>{};
  }
  template <size_t... I>
  CUTLASS_DEVICE static void apply(uint64_t const& a, uint64_t const& b,
                                  float* d, bool accumulate,
                                  cute::index_sequence<I...>) {
    decltype(operation())::fma(a, b, d[I]...,
        accumulate ? cute::SM90::GMMA::ScaleOut::One
                   : cute::SM90::GMMA::ScaleOut::Zero);
  }
  CUTLASS_DEVICE static void wgmma(uint64_t const& a, uint64_t const& b,
                                  float* d, bool accumulate) {
    apply(a, b, d, accumulate, cute::make_index_sequence<kNumAccum>{});
  }
};

template <class Pointer>
CUTLASS_DEVICE cute::GmmaDescriptor make_smem_desc(
    Pointer pointer, const int& layout, const uint32_t& leading = 0,
    const uint32_t& stride = 1024) {
  cute::GmmaDescriptor desc;
  desc.bitfield.start_address_ = cute::cast_smem_ptr_to_uint(pointer) >> 4;
  desc.bitfield.layout_type_ = layout;
  desc.bitfield.leading_byte_offset_ = leading >> 4;
  desc.bitfield.stride_byte_offset_ = stride >> 4;
  desc.bitfield.base_offset_ = 0;
  return desc;
}

template <Major Mode, uint32_t MN, uint32_t K, uint32_t Swizzle, class T>
CUTLASS_DEVICE cute::GmmaDescriptor make_gmma_desc(
    T* base, uint32_t mn, uint32_t k) {
  static_assert(Swizzle == 128);
  constexpr auto atom = Swizzle / sizeof(T);
  constexpr auto layout = static_cast<int>(cute::SM90::GMMA::LayoutType::B128);
  if constexpr (Mode == Major::K) {
    static_assert(K % atom == 0);
    return make_smem_desc(base + (k / atom) * MN * atom + mn * atom + k % atom,
                          layout, 0, 8 * Swizzle);
  } else {
    return make_smem_desc(base + mn * K + k * atom, layout,
                          K * atom * sizeof(T), 8 * atom * sizeof(T));
  }
}

template <uint32_t Inner, uint32_t Outer, uint32_t Swizzle, class T>
CUTLASS_DEVICE void tma_load(
    void const* map, cutlass::arch::ClusterTransactionBarrier* barrier,
    T* shared, const uint32_t& inner, const uint32_t& outer,
    const uint32_t& multicast = 1,
    uint64_t hint = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL)) {
  static_assert(Swizzle == 128);
  constexpr auto atom = Swizzle / sizeof(T);
  if (multicast == 1) {
#pragma unroll
    for (uint32_t i = 0; i < Inner / atom; ++i)
      cute::SM90_TMA_LOAD_2D::copy(map, reinterpret_cast<uint64_t*>(barrier),
          hint, shared + i * Outer * atom, inner + i * atom, outer);
  } else if (cute::block_rank_in_cluster() == 0) {
#pragma unroll
    for (uint32_t i = 0; i < Inner / atom; ++i)
      cute::SM90_TMA_LOAD_MULTICAST_2D::copy(map,
          reinterpret_cast<uint64_t*>(barrier), (1 << multicast) - 1, hint,
          shared + i * Outer * atom, inner + i * atom, outer);
  }
}

} // namespace hopper
