#include <iostream>
#include <uni20/core/math.hpp>
#include <uni20/core/scalar_io.hpp>

int main()
{
  using namespace uni20;
  std::cout << "Upper incomplete Gamma integrates t^(a-1)*exp(-t) from x to infinity.\n"
               "These are unregularized values: no division by Gamma(a).\n"
               "Negative shape is valid when the lower endpoint is positive:\n";
  for (double a : {-3., -.5, 0., .5, 2.})
    std::cout << "Gamma(" << a << ", 1) = " << format_scalar(math::upper_gamma(a, 1.)) << '\n';
  std::cout << "\nAt a=1/2, x=1, the answer is sqrt(pi)*erfc(1). Precision remains native:\n";
  std::cout << "float32: " << format_scalar(math::upper_gamma(.5f, 1.f)) << '\n';
  std::cout << "float64: " << format_scalar(math::upper_gamma(.5, 1.)) << '\n';
#if UNI20_HAS_FLOAT80
  std::cout << "float80: " << format_scalar(math::upper_gamma(float80(.5), float80(1))) << '\n';
#endif
#if UNI20_HAS_FLOAT128
  std::cout << "float128: " << format_scalar(math::upper_gamma(float128(.5), float128(1))) << '\n';
#endif
  std::cout << "\nA finite tail remains representable even when complete Gamma overflows:\n"
            << "Gamma(200, 1000) = " << format_scalar(math::upper_gamma(200., 1000.)) << '\n';
  std::cout << "At x=0 the API uses analytic Gamma(a), not the divergent-integral interpretation\n"
               "for nonpositive shape. All calculations above use native arithmetic.\n";
}
