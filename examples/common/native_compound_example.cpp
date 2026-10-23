#include <cstdint>
#include <iostream>
#include <uni20/core/math.hpp>
#include <uni20/core/scalar_io.hpp>

int main()
{
  using namespace uni20;
  double x = 0x1p-60;
  std::int64_t n = std::int64_t(1) << 60;
  std::cout << "Compound growth from a tiny double-precision increment.\n"
               "x = 2^-60 and n = 2^60: (1+x)^n is close to e, but double(1+x) is exactly 1.\n";
  std::cout << "pown(1+x,n) = " << format_scalar(math::pown(1 + x, n)) << '\n';
  std::cout << "compound(x,n) = " << format_scalar(math::compound(x, n)) << '\n';
  std::cout << "compound(x,-n) = " << format_scalar(math::compound(x, -n)) << '\n';
  std::cout << "\nNegative powers also preserve representable reciprocal tails.\n"
               "Although 2^1024 overflows double, 2^-1024 is a nonzero subnormal.\n";
  std::cout << "1 / pown(2,1024) = " << format_scalar(1 / math::pown(2.0, 1024)) << '\n';
  std::cout << "compound(1,-1024) = " << format_scalar(math::compound(1.0, -1024)) << '\n';
  std::cout << "Native inputs use native arithmetic; MPFR is not part of these evaluations.\n";
}
