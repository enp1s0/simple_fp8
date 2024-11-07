#pragma once
#include <cstdint>

#ifdef __CUDA_ARCH__
#define CUDA_DEVICE_FUNC __device__
#else
#define CUDA_DEVICE_FUNC
#endif

namespace mtk {
template <std::uint32_t exp_len>
struct f8 {
  static_assert(exp_len <= 4);
  std::uint8_t data;
};

template <class T>
struct numerical_limits;
template <std::uint32_t exp_len>
struct numerical_limits<mtk::f8<exp_len>> {
  static constexpr mtk::f8<exp_len> min{.data = 0x01};
  static constexpr mtk::f8<exp_len> max{.data = 0x7f};
};


template <std::uint32_t exp_len>
CUDA_DEVICE_FUNC inline float to_f32(const mtk::f8<exp_len> v) {
  constexpr std::uint32_t mantissa_len = 7 - exp_len;
  const auto data_u32 = static_cast<std::uint32_t>(v.data);
  const std::uint32_t sign = (data_u32 & 0x80) << 24;
  const std::uint32_t mantissa = data_u32 & ((1u << mantissa_len) - 1);
  const std::uint32_t exponent = (data_u32 & 0x7f) >> mantissa_len;

  if ((v.data & 0x7f) == 0) {
    return 0;
  }

  const std::uint32_t bs_f32_exponent = exponent + 127 - ((1u << (exp_len - 1)) - 1);
  const std::uint32_t bs_f32_mantissa = ((mantissa << 1) | 0x1) << (22 - mantissa_len);
  const std::uint32_t bs_f32 = sign | bs_f32_mantissa | (bs_f32_exponent << 23);

  return *reinterpret_cast<const float*>(&bs_f32);
}

template <std::uint32_t exp_len>
CUDA_DEVICE_FUNC inline mtk::f8<exp_len> to_f8(const float v) {
  constexpr std::uint32_t mantissa_len = 7 - exp_len;
  const auto bs_f32 = *reinterpret_cast<const std::uint32_t*>(&v);
  const std::uint8_t sign_u8 = (bs_f32 & 0x80000000u) >> 24;
  const std::uint8_t mantissa_u8 = ((bs_f32 & 0x7fffffu) >> (23 - mantissa_len));
  const std::uint32_t exponent_u32 = (bs_f32 & 0x7fffffffu) >> 23;

  if (std::abs(v) < mtk::to_f32(mtk::numerical_limits<mtk::f8<exp_len>>::min)) {
    return mtk::f8<exp_len>{0};
  }

  const std::uint8_t exponent_u8 = exponent_u32 - 127 + ((1u << (exp_len - 1)) - 1);
  const std::uint8_t bs_f8 = sign_u8 | (exponent_u8 << mantissa_len) | mantissa_u8;

  return mtk::f8<exp_len>{bs_f8};
}
} // namespace mtk
