#pragma once

#include <uni20/random/identity.hpp>
#include <uni20/random/transforms.hpp>

namespace uni20::random
{
/// \brief A fixed-width transform declares the number of consecutive blocks consumed and values produced.
template <class T>
concept WordTransform = requires(T const& transform, typename T::input_type const& group) {
  requires T::input_type::size > 0 && T::result_type::size > 0;
  { transform(group) } -> std::same_as<typename T::result_type>;
};

template <WordTransform T> using transformed_value_t = typename T::result_type::value_type;

namespace detail
{
template <WordTransform T> inline constexpr std::uint64_t group_count = value_domain / T::input_type::size;

// The last complete group or the last representable element offset, whichever comes first.
template <WordTransform T>
inline constexpr std::uint64_t last_element =
    group_count<T> > UINT64_MAX / T::result_type::size ? UINT64_MAX : group_count<T>* T::result_type::size - 1;
} // namespace detail

/// \brief Evaluate one complete group of consecutive Philox blocks into its ordered output values.
template <WordTransform Transform>
[[nodiscard]] UNI20_HOST_DEVICE auto value_group(Identity leaf, std::uint64_t group_index, Transform transform)
{
  if (group_index >= detail::group_count<Transform>) detail::address_out_of_range();
  typename Transform::input_type input;
  auto const first = group_index * input.size;
  for (unsigned i = 0; i < input.size; ++i)
    input.blocks[i] = value_block(leaf, first + i);
  return transform(input);
}

/// \brief Evaluate one element at its allocation offset, in elements, under a supplied leaf.
template <WordTransform Transform>
[[nodiscard]] UNI20_HOST_DEVICE auto value_at(Identity leaf, std::uint64_t offset, Transform transform)
{
  constexpr auto g = Transform::result_type::size;
  return value_group(leaf, offset / g, transform).values[offset % g];
}
} // namespace uni20::random
