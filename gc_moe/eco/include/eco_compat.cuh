#pragma once

// Minimal stand-ins for the DeepGEMM / CUTLASS helpers the ECO plan uses, so the plan kernel compiles with
// plain nvcc (no CUTLASS headers)

#include <cstdint>

#ifndef CUTLASS_DEVICE
#define CUTLASS_DEVICE __forceinline__ __device__
#endif
#ifndef CUTLASS_HOST_DEVICE
#define CUTLASS_HOST_DEVICE __forceinline__ __host__ __device__
#endif

#define DG_STATIC_ASSERT(cond, ...) static_assert(cond, __VA_ARGS__)
#ifdef __CUDA_ARCH__
#define DG_DEVICE_ASSERT(cond) do { if (not (cond)) { printf("ECO plan assertion failed: %s\n", #cond); asm volatile("trap;"); } } while (0)
#else
#define DG_DEVICE_ASSERT(cond)
#endif

namespace deep_gemm {

namespace math {
template <typename T>
CUTLASS_HOST_DEVICE constexpr T constexpr_ceil_div(T a, T b) { return (a + b - 1) / b; }
template <typename T>
CUTLASS_HOST_DEVICE T ceil_div(T a, T b) { return (a + b - 1) / b; }
} // namespace math

namespace ptx {
CUTLASS_DEVICE uint32_t get_lane_idx() {
    uint32_t lane;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
    return lane;
}
CUTLASS_DEVICE void red_add(const int* ptr, const int& value) {
    asm volatile("red.gpu.global.add.s32 [%0], %1;" :: "l"(ptr), "r"(value));
}
} // namespace ptx

} // namespace deep_gemm

namespace cute {
template <typename T>
CUTLASS_HOST_DEVICE constexpr T min(const T& a, const T& b) { return a < b ? a : b; }
template <typename T>
CUTLASS_HOST_DEVICE constexpr T max(const T& a, const T& b) { return a < b ? b : a; }
CUTLASS_DEVICE bool elect_one_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n"
        ".reg .b32 %%rx;\n"
        ".reg .pred %%px;\n"
        "     elect.sync %%rx|%%px, %1;\n"
        "@%%px mov.s32 %0, 1;\n"
        "}\n"
        : "+r"(pred) : "r"(0xffffffffu));
    return pred != 0;
}
} // namespace cute
