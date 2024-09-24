#include <iostream>
#include <cmath>
#include <simple_fp8.hpp>

template <std::uint32_t exp_len>
int eval(const float v) {
  const auto f8 = mtk::to_f8<exp_len>(v);
  const auto r = mtk::to_f32(f8);

  const auto error_threshold = std::log10(7. - exp_len);
  const float relative_error = v == 0 ? 0 : std::abs((v - r) / v);

  //std::printf("org = %+7e, rec = %+7e, relative_error = %e\n", v, r, relative_error);
  if (error_threshold < relative_error && v != 0 && r != 0) {
    std::printf("Error for %e (%e) is larger than the threshold (%e), fp8=%e\n", v, relative_error, error_threshold, r);
    return 1;
  }
  return 0;
}

int main() {
  std::uint32_t num_tests = 0;
  std::uint32_t num_passed = 0;
  for (int i = -120; i <= 120; i++) {
    num_tests += 4;
    if (!eval<2>(i / 10.f)) {num_passed++;}
    if (!eval<3>(i / 10.f)) {num_passed++;}
    if (!eval<4>(i / 10.f)) {num_passed++;}
    if (!eval<5>(i / 10.f)) {num_passed++;}
  }
  std::printf("[TEST] %5u / %5u passed\n", num_passed, num_tests);
}
