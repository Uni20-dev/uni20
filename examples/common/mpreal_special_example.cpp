#include <iostream>
#include <uni20/core/math.hpp>

int main()
{
  using namespace uni20;
  namespace m = uni20::math;
  auto p = Precision::bits(128), exact = Precision::exact();
  std::cout << "MPFR special functions use the value's precision, or an explicit override.\n"
            << "The same scalar keeps recognized rational results exact.\n"
            << "  Gamma(7), exact: " << m::tgamma(mpreal{7}) << '\n'
            << "  zeta(-1), exact: " << m::zeta(mpreal{-1}) << '\n'
            << "  sinpi(1/6), exact: " << m::sinpi(mpreal("1/6", exact)) << '\n'
            << "  Gamma(1/2), 128 bits: " << m::tgamma(mpreal("1/2", exact), p) << '\n';

  auto gamma = m::lgamma_sign(mpreal("-1/2", p));
  std::cout << "\nLog-Gamma reports log(abs(Gamma(x))) and its sign separately.\n"
            << "  x=-1/2: value=" << gamma.value << ", sign=" << gamma.sign << '\n'
            << "  upper Gamma(3,1): " << m::upper_gamma(mpreal(3, p), mpreal(1, p)) << '\n'
            << "  real part of Li2(2): " << m::dilog_real(mpreal(2, p)) << '\n';

  auto tiny = m::ldexp(mpreal(1, p), -160);
  auto large = m::ldexp(mpreal(1, p), 120);
  std::cout << "\nDedicated routines avoid cancellation and rounded-pi argument errors.\n"
            << "  log2(1+2^-160), rounded sum: " << m::log2(1 + tiny) << '\n'
            << "  log2p1(2^-160): " << m::log2p1(tiny) << '\n'
            << "  sin(rounded pi * 2^120): " << m::sin(pi<mpreal>.at(p) * large) << '\n'
            << "  sinpi(2^120): " << m::sinpi(large) << '\n';

  auto low = Precision::bits(3);
  auto third = mpreal("2/3", exact);
  std::cout << "\nCompound is an arithmetic utility: an override rounds the result.\n"
            << "  (1+2/3)^2, rounded to three bits: " << m::compound(third, 2, low) << '\n'
            << "  compound after rounding 2/3 first: " << m::compound(third.at(low), 2) << '\n'
            << "\nConstants are evaluated at the requested precision.\n"
            << "  log(2): " << log_two<mpreal>.at(p) << '\n'
            << "  Euler's constant: " << euler_gamma<mpreal>.at(p) << '\n'
            << "  Catalan's constant: " << catalan<mpreal>.at(p) << '\n';
}
