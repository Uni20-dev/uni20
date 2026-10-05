#include <iostream>
#include <uni20/core/math.hpp>

int main()
{
  using namespace uni20;
  namespace m = uni20::math;
  auto exact = Precision::exact();
  auto low = Precision::bits(3), p = Precision::bits(128);
  mpreal third("1/3", exact);
  std::cout << "Numerical utilities preserve exact operands until the result is rounded.\n"
            << "At three bits, 1/3 is not representable, but (1/3)*3-1 is exactly zero.\n"
            << "  fma(exact third, 3, -1): " << m::fma(third, mpreal(3, low), mpreal(-1, low)) << '\n'
            << "  fma(rounded third, 3, -1): " << m::fma(third.at(low), mpreal(3, low), mpreal(-1, low)) << '\n';

  mpreal near_one("1023/1024", exact);
  std::cout << "\nExplicit precision rounds the utility result, not its inputs.\n"
            << "  floor(1023/1024, three bits): " << m::floor(near_one, low) << '\n'
            << "  floor(after rounding 1023/1024 to three bits): " << m::floor(near_one.at(low)) << '\n';

  auto binary = m::frexp(mpreal("7/3", exact));
  auto fraction = m::modf(mpreal("7/3", exact));
  auto remainder = m::remquo(mpreal("7/3", exact), mpreal("2/3", exact));
  std::cout << "\nDecompositions return named, owning values, with exact rationals retained.\n"
            << "  7/3 = " << binary.fraction << " * 2^" << binary.exponent << '\n'
            << "  7/3 = " << fraction.integer << " + " << fraction.fraction << '\n'
            << "  nearest-even remainder of (7/3)/(2/3): " << remainder.remainder
            << "; signed low three quotient bits: " << remainder.quotient << '\n';

  auto one = mpreal(1, p), direction = mpreal("1.000000000000000000000000000000000000000001", Precision::bits(256));
  std::cout << "\nAdjacent values need a finite grid; the direction may have greater precision.\n"
            << "  nextafter(128-bit one, a larger 256-bit value) - one: " << m::nextafter(one, direction) - one << '\n'
            << "  exact cube root of -8/27: " << m::rootn(mpreal("-8/27", exact), 3) << '\n'
            << "  cube root of 2 at 128 bits: " << m::rootn(mpreal{2}, 3, p) << '\n';
}
