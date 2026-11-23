#pragma once

#include <uni20/core/compiler_attributes.hpp>

#include <cstddef>

namespace uni20
{

/// \brief Inclusive bounds on handle-relative element offsets.
struct offset_range
{
    std::ptrdiff_t minimum = 0;
    std::ptrdiff_t maximum = 0;

    /// \brief Report whether the range contains no offsets.
    [[nodiscard]] UNI20_HOST_DEVICE constexpr bool empty() const noexcept { return minimum > maximum; }
};

} // namespace uni20
