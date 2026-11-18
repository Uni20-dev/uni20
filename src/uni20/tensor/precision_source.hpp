#pragma once

#include <uni20/core/runtime_precision.hpp>

namespace uni20::detail
{
// Descriptor views copy this binding to the ultimate owner, never to an
// intermediate view. Fixed scalar types need no metadata or extra storage.
template <class T, bool = has_runtime_precision_v<T>> class precision_source {};

#if UNI20_HAS_MPREAL
template <class T> class precision_source<T, true> {
  public:
    constexpr precision_source() = default;
    explicit constexpr precision_source(precision_default<T> const& owner) : value_(&owner.value) {}

    [[nodiscard]] std::optional<Precision> default_precision_if_set() const noexcept
    {
      return value_ ? *value_ : std::nullopt;
    }

    [[nodiscard]] Precision default_precision() const
    {
      auto p = this->default_precision_if_set();
      if (!p) throw std::logic_error("tensor: no default working precision was supplied");
      return *p;
    }

  private:
    std::optional<Precision> const* value_ = nullptr;
};
#endif

template <class Tensor>
[[nodiscard]] auto descriptor_precision_source(Tensor const& tensor) -> precision_source<typename Tensor::value_type>
{
  if constexpr (requires { tensor.precision_source(); })
    return tensor.precision_source();
  else
    return {};
}
} // namespace uni20::detail
