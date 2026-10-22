#pragma once

#include <uni20/core/runtime_precision.hpp>
#include <uni20/krylov/matrix_free.hpp>

namespace uni20::krylov
{
/// \brief An opaque vector space with one explicit arithmetic precision.
/// \details The operations and prototype are borrowed and must outlive the
///          solve. Allocation must preserve the prototype's vector-space
///          metadata. Fixed scalar types require their native precision;
///          runtime scalars require finite precision. Operations are responsible
///          for performing their arithmetic at that precision.
template <RealOrComplex S, class Ops, class V>
  requires KrylovVectorOps<Ops, V, S>
struct KrylovVectorSpace
{
    using scalar_type = S;
    using real_type = make_real_t<S>;
    using vector_type = V;
    Ops& ops;
    V const& prototype;
    Precision precision;

    void validate() const
    {
      require_precision<S>(precision);
      (void)precision.bit_count();
      if (ops.vector_dimension(prototype) != ops.problem_dimension())
        throw std::invalid_argument("Krylov prototype does not belong to the vector space");
    }
};

/// \brief Bind vector operations and an allocation prototype to finite precision.
template <RealOrComplex S, class Ops, class V> auto krylov_space(Ops& ops, V const& prototype, Precision p)
{
  KrylovVectorSpace<S, Ops, V> space{ops, prototype, p};
  space.validate();
  return space;
}

/// \brief Bind fixed-scalar vector operations at native precision.
template <RealOrComplex S, class Ops, class V>
  requires(!has_runtime_precision_v<S>)
auto krylov_space(Ops& ops, V const& prototype)
{
  return krylov_space<S>(ops, prototype, native_precision<S>());
}

namespace gmres_detail
{
// Scalar conversion for explicit workspace boundaries, including complex
// components and native extension types not covered by std::complex constructors.
template <RealOrComplex To, RealOrComplex From>
  requires(Complex<To> == Complex<From>)
To convert_scalar(From const& value, Precision p)
{
  require_precision<To>(p);
  if constexpr (Complex<To>)
    return To(convert_scalar<make_real_t<To>>(uni20::real(value), p),
              convert_scalar<make_real_t<To>>(uni20::imag(value), p));
  else if constexpr (has_runtime_precision_v<To>)
    return scalar_at_precision<To>(value, p);
  else
    return static_cast<To>(value);
}
} // namespace gmres_detail
} // namespace uni20::krylov
