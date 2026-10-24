#include <iostream>
#include <uni20/core/math.hpp>
#include <uni20/core/scalar_io.hpp>

int main()
{
  using namespace uni20;
  std::cout << "Arithmetic-geometric means use repeated arithmetic and geometric averaging.\n"
               "The shared math interface retains each native result type:\n";
  std::cout << "AGM(1, 1/2), float32 = " << format_scalar(math::agm(1.0f, 0.5f)) << '\n';
  std::cout << "AGM(1, 1/2), float64 = " << format_scalar(math::agm(1.0, 0.5)) << '\n';
#if UNI20_HAS_FLOAT80
  std::cout << "AGM(1, 1/2), float80 = " << format_scalar(math::agm(float80(1), float80(0.5))) << '\n';
#endif
#if UNI20_HAS_FLOAT128
  std::cout << "AGM(1, 1/2), float128 = " << format_scalar(math::agm(float128(1), float128(0.5))) << '\n';
#endif
  auto tiny = numeric_limits<double>::denorm_min(), big = numeric_limits<double>::max();
  std::cout << "\nExtreme input ratio: the smallest positive double and the largest finite double.\n"
               "Scaling both by the larger exponent would erase the smaller input.\n"
               "AGM keeps separate scales until it is safe to combine them:\n";
  std::cout << "AGM(minimum-subnormal, maximum) = " << format_scalar(math::agm(tiny, big)) << '\n';
  std::cout << "All calculations above use native arithmetic without MPFR.\n";
}
