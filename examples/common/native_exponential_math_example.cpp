#include <iostream>
#include <uni20/core/math.hpp>
#include <uni20/core/scalar_io.hpp>

int main()
{
  using namespace uni20;
  double x = 1e-20;
  std::cout << "Stable exponential/logarithmic changes for x = 1e-20 (double).\n"
               "At this scale, forming 1+x or subtracting 1 from an exponential loses the change.\n";
  std::cout << "naive log2(1+x) = " << format_scalar(math::log2(1 + x)) << '\n';
  std::cout << "stable log2p1(x) = " << format_scalar(math::log2p1(x)) << '\n';
  std::cout << "naive exp2(x)-1 = " << format_scalar(math::exp2(x) - 1) << '\n';
  std::cout << "stable exp2m1(x) = " << format_scalar(math::exp2m1(x)) << '\n';
  std::cout << "stable exp10m1(x) = " << format_scalar(math::exp10m1(x)) << '\n';
  std::cout << "stable log10p1(x) = " << format_scalar(math::log10p1(x)) << '\n';
  std::cout << "All calls retain the native type; MPFR is not used for native evaluation.\n";
#if UNI20_HAS_FLOAT80
  std::cout << "Extended exponent range: exp10(400) in native float80 = " << format_scalar(math::exp10(float80(400)))
            << '\n';
#endif
#if UNI20_HAS_FLOAT128
  std::cout << "Extended exponent range: exp10(4000) in native float128 = "
            << format_scalar(math::exp10(float128(4000))) << '\n';
#endif
}
