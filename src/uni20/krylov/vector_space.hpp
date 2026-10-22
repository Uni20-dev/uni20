#pragma once

#include <uni20/core/runtime_precision.hpp>
#include <uni20/krylov/matrix_free.hpp>

#include <type_traits>
#include <utility>

namespace uni20::krylov
{
/// \brief An opaque vector space with one explicit arithmetic precision.
/// \details Owns an allocation descriptor, not a numerical vector. The
///          operations and any storage/context borrowed by the descriptor must
///          outlive the solve. Allocation returns an independent owning vector
///          preserving the descriptor's metadata and placement.
///          Fixed scalar types require their native precision;
///          runtime scalars require finite precision. Operations are responsible
///          for performing their arithmetic at that precision.
template <RealOrComplex S, class Ops, class Prototype>
  requires(!std::is_reference_v<Prototype>) &&
          KrylovVectorOps<Ops, decltype(std::declval<Ops&>().allocate_like(std::declval<Prototype const&>())), S>
struct KrylovVectorSpace
{
    using scalar_type = S;
    using real_type = make_real_t<S>;
    using prototype_type = Prototype;
    using vector_type = decltype(std::declval<Ops&>().allocate_like(std::declval<Prototype const&>()));
    Ops& ops;
    Prototype prototype;
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
/// \details `ops.prototype_view(source)` produces a descriptor retained by value.
///          The adapter accepts temporary borrowed views, rejects temporary
///          owners, and allocates owning work vectors from the descriptor.
template <RealOrComplex S, class Ops, class V>
  requires requires(Ops& ops, V&& source) { ops.prototype_view(std::forward<V>(source)); }
auto krylov_space(Ops& ops, V&& source, Precision p)
{
  using prototype_type = decltype(ops.prototype_view(std::forward<V>(source)));
  KrylovVectorSpace<S, Ops, prototype_type> space{ops, ops.prototype_view(std::forward<V>(source)), p};
  space.validate();
  return space;
}

/// \brief Bind fixed-scalar vector operations at native precision.
template <RealOrComplex S, class Ops, class V>
  requires(!has_runtime_precision_v<S>) &&
          requires(Ops& ops, V&& source) { ops.prototype_view(std::forward<V>(source)); }
auto krylov_space(Ops& ops, V&& source)
{
  return krylov_space<S>(ops, std::forward<V>(source), native_precision<S>());
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
