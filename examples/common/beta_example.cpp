#include <iostream>
#include <uni20/core/math.hpp>
#include <uni20/core/scalar_io.hpp>

int main()
{
  using namespace uni20;
  std::cout << "Beta(a,b) integrates t^(a-1)*(1-t)^(b-1) over [0,1], for positive parameters.\n"
               "B(1/2,1/2)=pi. Each call below preserves its native result type:\n";
  std::cout << "float32: " << format_scalar(math::beta(.5f, .5f)) << '\n';
  std::cout << "float64: " << format_scalar(math::beta(.5, .5)) << '\n';
#if UNI20_HAS_FLOAT80
  std::cout << "float80: " << format_scalar(math::beta(float80(.5), float80(.5))) << '\n';
#endif
#if UNI20_HAS_FLOAT128
  std::cout << "float128: " << format_scalar(math::beta(float128(.5), float128(.5))) << '\n';
  auto large = numeric_limits<float128>::max();
  std::cout << "\nThe binary128 fallback avoids enormous intermediate Gamma values.\n"
               "B(1/2,maximum float128) remains positive and representable:\n";
  std::cout << format_scalar(math::beta(float128(.5), large)) << '\n';
  std::cout << "B(maximum float128,maximum float128) underflows to zero without forming their sum:\n";
  std::cout << format_scalar(math::beta(large, large)) << '\n';
#endif
  std::cout << "\nExisting float/double/long-double standard providers retain their own accuracy and range limits.\n"
               "Native arguments are never evaluated through MPFR.\n";
}
