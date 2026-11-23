#pragma once

#include <uni20/core/compiler_attributes.hpp>

#include <cstddef>
#include <type_traits>

namespace uni20
{

/// \brief Access elements through a pointer and a signed element offset.
template <class T> struct default_accessor
{
    using element_type = T;
    using reference = T&;
    using data_handle_type = T*;
    using offset_type = std::ptrdiff_t;
    using offset_policy = default_accessor;

    UNI20_HOST_DEVICE constexpr default_accessor() noexcept {}

    template <class U>
      requires std::is_convertible_v<U (*)[], T (*)[]>
    UNI20_HOST_DEVICE constexpr default_accessor(default_accessor<U>) noexcept
    {}

    /// \brief Read or write an element relative to the handle.
    [[nodiscard]] UNI20_HOST_DEVICE constexpr reference access(data_handle_type handle,
                                                               offset_type offset) const noexcept
    {
      return handle[offset];
    }

    /// \brief Advance the handle by an element offset.
    [[nodiscard]] UNI20_HOST_DEVICE constexpr data_handle_type offset(data_handle_type handle,
                                                                      offset_type delta) const noexcept
    {
      return handle + delta;
    }
};

} // namespace uni20
