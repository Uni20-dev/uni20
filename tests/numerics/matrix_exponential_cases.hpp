#pragma once

#include "precision_cases.hpp"

#include <uni20/linalg/backends/cpu/matrix_exponential.hpp>

#include <array>

namespace uni20::test
{

// The analytical rotation is an independent oracle for the general Padé
// implementation. A zero matrix passed the old finite-and-bounded regression.
template <class C> void expect_large_planar_rotations()
{
  using Real = typename C::real_type;
  using Scalar = typename C::scalar_type;
  Real const huge = C::digits() <= 24 ? static_cast<Real>(1.0e20F) : static_cast<Real>(1.0e80);
  // The fractional part is representable in Real but lost if an extended
  // precision angle is narrowed to double (or binary128 to long double).
  Real const precision_sensitive = math::ldexp(C::real(1), C::digits() - 4) + C::real(1) / C::real(4);
  Real const near_limit = math::ldexp(C::real(1), numeric_limits<Real>::max_exponent - 2);
  Real const tolerance = C::real(8) * C::epsilon();
  std::array<Real, 3> const angles{huge, precision_sensitive, near_limit};
  for (std::size_t angle_index = 0; angle_index < angles.size(); ++angle_index)
  {
    for (int sign : {-1, 1})
    {
      SCOPED_TRACE(::testing::Message() << "angle case=" << angle_index << " sign=" << sign);
      Real const theta = C::real(sign) * angles[angle_index];
      auto matrix = C::matrix(2, 2);
      matrix[0, 1] = Scalar(-theta);
      matrix[1, 0] = Scalar(theta);
      auto const result = linalg::backends::cpu::matrix_exponential(matrix, C::real(1));
      Real const cosine = math::cos(theta);
      Real const sine = math::sin(theta);
      std::array<Scalar, 4> const expected{Scalar(cosine), Scalar(-sine), Scalar(sine), Scalar(cosine)};
      for (index_type row = 0; row < 2; ++row)
      {
        for (index_type col = 0; col < 2; ++col)
        {
          EXPECT_TRUE(math::abs(result[row, col] - expected[2 * row + col]) <= tolerance)
              << "entry (" << row << ", " << col << ")";
        }
      }
    }
  }
}

} // namespace uni20::test
