#pragma once

#include <array>
#include <uni20/mdspan/reverse.hpp>
#include <vector>

// Convenience typedefs
using index_t = std::ptrdiff_t;

/// \brief  Helper to construct a mapping (extents + strides)
template <std::size_t N, typename index_type>
auto make_mapping(std::array<std::size_t, N> const& extents, std::array<index_type, N> const& strides)
{
  using extents_t = stdex::dextents<index_type, N>;
  using mapping_t = uni20::layout_signed_stride::mapping<extents_t>;

  return mapping_t(extents_t(extents), strides);
}

/// \brief  A helper to build a 1D mdspan over a std::vector using layout_stride
inline auto make_mdspan_1d(std::vector<double>& v)
{
  using extents_t = stdex::dextents<index_t, 1>;
  std::array<std::ptrdiff_t, 1> strides{1};
  auto mapping = uni20::layout_stride::mapping<extents_t>(extents_t{v.size()}, strides);
  return uni20::mdspan<double, extents_t, uni20::layout_stride>(v.data(), mapping);
}

/// \brief  A helper to build a 2D row‐major mdspan using layout_stride
inline auto make_mdspan_2d(std::vector<double>& v, std::size_t R, std::size_t C)
{
  using extents_t = stdex::dextents<index_t, 2>;
  std::array<std::ptrdiff_t, 2> strides{static_cast<index_t>(C), 1}; // Row-major
  auto mapping = uni20::layout_stride::mapping<extents_t>(extents_t{R, C}, strides);
  return uni20::mdspan<double, extents_t, uni20::layout_stride>(v.data(), mapping);
}

inline auto make_mdspan_2d(std::vector<double>& v, std::size_t R, std::size_t C, std::array<std::ptrdiff_t, 2> strides)
{
  using extents_t = stdex::dextents<index_t, 2>;
  auto mapping = uni20::layout_stride::mapping<extents_t>(extents_t{R, C}, strides);
  return uni20::mdspan<double, extents_t, uni20::layout_stride>(v.data(), mapping);
}

/// \brief  A helper to build a reversed‐1D mdspan (negative stride)
inline auto make_reversed_1d(std::vector<double>& v)
{
  using extents_t = stdex::dextents<index_t, 1>;
  return uni20::reverse<0>(uni20::mdspan{v.data(), extents_t{v.size()}});
}

/// \brief  Build a 3D row-major mdspan with layout_stride
inline auto make_mdspan_3d(std::vector<double>& v, std::size_t R, std::size_t K, std::size_t L)
{
  using extents_t = stdex::dextents<index_t, 3>;
  // row-major: stride[0]=K*L, stride[1]=L, stride[2]=1
  std::array<index_t, 3> strides{static_cast<index_t>(K * L), static_cast<index_t>(L), 1};
  auto m = uni20::layout_stride::mapping<extents_t>(extents_t{R, K, L}, strides);
  return uni20::mdspan<double, extents_t, uni20::layout_stride>(v.data(), m);
}

/// \brief Compute the minimal span size needed for the supplied extents/strides.
template <std::size_t Rank>
constexpr std::size_t span_size_for(std::array<std::size_t, Rank> const& extents,
                                    std::array<index_t, Rank> const& strides)
{
  std::ptrdiff_t min_offset = 0;
  std::ptrdiff_t max_offset = 0;

  auto update = [&](std::ptrdiff_t stride, std::size_t extent) {
    if (extent == 0) return;
    if (stride >= 0)
    {
      max_offset += stride * static_cast<std::ptrdiff_t>(extent - 1);
    }
    else
    {
      min_offset += stride * static_cast<std::ptrdiff_t>(extent - 1);
    }
  };

  for (std::size_t i = 0; i < Rank; ++i)
  {
    update(strides[i], extents[i]);
  }

  return static_cast<std::size_t>(max_offset - min_offset + 1);
}

/// \brief Build an mdspan with arbitrary extents and strides.
template <std::size_t Rank>
inline auto make_mdspan_strided(std::vector<double>& v, std::array<std::size_t, Rank> const& extents,
                                std::array<index_t, Rank> const& strides, std::ptrdiff_t offset = 0)
{
  using extents_t = stdex::dextents<index_t, Rank>;
  auto mapping = uni20::layout_signed_stride::mapping<extents_t>(extents_t(extents), strides);
  return uni20::mdspan<double, extents_t, uni20::layout_signed_stride>(v.data() + offset, mapping);
}
