#include <iostream>
#include <cmath>
#include <simple_fp8.hpp>

template <std::uint32_t exp_len>
int eval(const float v) {
  if (std::abs(v) > mtk::to_f32(mtk::numerical_limits<mtk::f8<exp_len>>::max)) {
    return 0;
  }

  const auto f8 = mtk::to_f8<exp_len>(v);
  const auto r = mtk::to_f32(f8);

  const auto error_threshold = std::log10(7. - exp_len);
  const float relative_error = v == 0 ? 0 : std::abs((v - r) / v);

  //std::printf("org = %+7e, rec = %+7e, relative_error = %e\n", v, r, relative_error);
  if (error_threshold < relative_error && v != 0 && r != 0) {
    std::printf("[exp=%u] Error for fp32=%e (relative error=%e) is larger than the threshold (%e), fp8=%e\n", exp_len, v, relative_error, error_threshold, r);
    return 1;
  }
  return 0;
}

template <std::uint32_t exp_len>
int equiv_eval(const std::uint8_t v) {
  if (v == 0x80) {
    return 0;
  }
  const auto f32 = mtk::to_f32(mtk::f8<exp_len>{v});
  const auto w = mtk::to_f8<exp_len>(f32).data;
  if (v != w) {
    std::printf("v = 0x%02x (%e) is not the same as the result of f32(f8(v))\n", v, f32);
    return 1;
  }
  return 0;
}

int main() {
  std::uint32_t num_tests = 0;
  std::uint32_t num_passed = 0;
  for (int i = -120; i <= 120; i++) {
    num_tests += 3;
    if (!eval<2>(i / 10.f)) {num_passed++;}
    if (!eval<3>(i / 10.f)) {num_passed++;}
    if (!eval<4>(i / 10.f)) {num_passed++;}
  }

  for (std::uint32_t i = 0; i <= 0xff; i++) {
    num_tests += 3;
    if (!equiv_eval<2>(i)) {num_passed++;}
    if (!equiv_eval<3>(i)) {num_passed++;}
    if (!equiv_eval<4>(i)) {num_passed++;}
  }
  std::printf("[TEST] %5u / %5u passed\n", num_passed, num_tests);
}
