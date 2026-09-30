#pragma once

#include "types.hpp"
#if UNI20_ENABLE_MPFR
#include "precision.hpp"
#include <optional>
#endif

namespace uni20
{
/// \brief Whether numerical values of this scalar family carry runtime precision.
template <typename T> inline constexpr bool has_runtime_precision_v = false;
#if UNI20_ENABLE_MPFR
template <> inline constexpr bool has_runtime_precision_v<mpreal> = true;
#endif
#if UNI20_ENABLE_MPC
template <> inline constexpr bool has_runtime_precision_v<mpcomplex> = true;
#endif

namespace detail
{
// Empty for fixed-precision values; owners and views need no extra runtime state.
template <typename T, bool = has_runtime_precision_v<std::remove_cv_t<T>>> struct precision_default
{};
#if UNI20_ENABLE_MPFR
template <typename T> struct precision_default<T, true>
{
    std::optional<Precision> value = {};

    Precision get() const
    {
      if (!value) throw std::logic_error("tensor: no default working precision was supplied");
      return *value;
    }
    Precision default_precision() const { return this->get(); }
    void default_precision(Precision p) { value = p; }
    std::optional<Precision> default_precision_if_set() const noexcept { return value; }
    template <class Source> void copy_default_precision(Source const& source)
    {
      if constexpr (requires { source.default_precision_if_set(); }) value = source.default_precision_if_set();
    }
};
#endif
template <typename Target, typename Source> constexpr Target with_precision_default(Target target, Source const& source)
{
  if constexpr (requires { target.copy_default_precision(source); }) target.copy_default_precision(source);
  return target;
}
} // namespace detail
} // namespace uni20
