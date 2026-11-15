#pragma once

#include <uni20/core/runtime_precision.hpp>
#include <uni20/tensor/concepts.hpp>

#include <memory>

namespace uni20::detail
{
// Descriptor copies retain a live precision source, rather than a snapshot of
// the default. Fixed scalar types need no binding or extra storage.
template <class T, bool = has_runtime_precision_v<T>> class precision_source {};

#if UNI20_HAS_MPREAL
template <class T> class precision_source<T, true> {
  public:
    constexpr precision_source() = default;

    template <class Owner>
      requires requires(Owner const& owner) { owner.default_precision_if_set(); }
    explicit constexpr precision_source(Owner const& owner)
        : owner_(std::addressof(owner)), read_([](void const* pointer) -> std::optional<Precision> {
            return static_cast<Owner const*>(pointer)->default_precision_if_set();
          })
    {}

    [[nodiscard]] std::optional<Precision> default_precision_if_set() const
    {
      return read_ ? read_(owner_) : std::nullopt;
    }

    [[nodiscard]] Precision default_precision() const
    {
      auto p = this->default_precision_if_set();
      if (!p) throw std::logic_error("tensor: no default working precision was supplied");
      return *p;
    }

  private:
    void const* owner_ = nullptr;
    std::optional<Precision> (*read_)(void const*) = nullptr;
};
#endif

template <class Tensor>
[[nodiscard]] constexpr auto
descriptor_precision_source(Tensor const& tensor) -> precision_source<tensor_element_t<Tensor>>
{
  using source_type = precision_source<tensor_element_t<Tensor>>;
  if constexpr (!has_runtime_precision_v<tensor_element_t<Tensor>>)
    return {};
  else if constexpr (requires { tensor.precision_source(); })
    return tensor.precision_source();
  else if constexpr (requires {
                       requires TensorView<decltype(tensor.base())> &&
                                    std::is_lvalue_reference_v<decltype(tensor.base())>;
                     })
    return descriptor_precision_source(tensor.base());
  else if constexpr (requires { tensor.default_precision_if_set(); })
    return source_type{tensor};
  else
    return {};
}
} // namespace uni20::detail
