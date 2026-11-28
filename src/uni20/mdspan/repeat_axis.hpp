#pragma once

#include <uni20/mdspan/mdspec.hpp>

namespace uni20::detail
{
// Read one source slice repeatedly along a new leading axis. The source mapping still
// describes every storage offset, including the holes in a strided vector.
template <class SourceMapping> struct RepeatFirstAxis
{
    template <class Extents> struct mapping
    {
        using extents_type = Extents;
        using index_type = typename Extents::index_type;
        using size_type = typename Extents::size_type;
        using rank_type = typename Extents::rank_type;
        using layout_type = RepeatFirstAxis;
        static_assert(Extents::rank() == SourceMapping::extents_type::rank() + 1);

        Extents shape;
        SourceMapping source;

        [[nodiscard]] UNI20_HOST_DEVICE constexpr auto const& extents() const { return shape; }
        template <class... Indices>
        [[nodiscard]] UNI20_HOST_DEVICE constexpr decltype(auto) operator()(index_type, Indices... indices) const
        {
          return source(indices...);
        }

        [[nodiscard]] UNI20_HOST_DEVICE constexpr auto stride(rank_type axis) const
          requires(SourceMapping::is_always_strided())
        {
          if constexpr (Extents::rank() == 1)
            return index_type{0};
          else
            return axis == 0 ? index_type{0} : static_cast<index_type>(source.stride(axis - 1));
        }
        [[nodiscard]] UNI20_HOST_DEVICE constexpr auto required_span_size() const
          requires StorageSpanMapping<SourceMapping>
        {
          return shape.extent(0) == 0 ? decltype(source.required_span_size()){0} : source.required_span_size();
        }
        [[nodiscard]] UNI20_HOST_DEVICE static constexpr bool is_always_strided()
        {
          return SourceMapping::is_always_strided();
        }
        [[nodiscard]] UNI20_HOST_DEVICE static constexpr bool is_always_unique() { return false; }
        [[nodiscard]] UNI20_HOST_DEVICE static constexpr bool is_always_exhaustive()
        {
          return SourceMapping::is_always_exhaustive();
        }
        [[nodiscard]] UNI20_HOST_DEVICE constexpr bool is_strided() const { return source.is_strided(); }
        [[nodiscard]] UNI20_HOST_DEVICE constexpr bool is_unique() const
        {
          return mdspan_empty(shape) || (shape.extent(0) == 1 && source.is_unique());
        }
        [[nodiscard]] UNI20_HOST_DEVICE constexpr bool is_exhaustive() const
        {
          return shape.extent(0) == 0 || source.is_exhaustive();
        }
        friend UNI20_HOST_DEVICE constexpr bool operator==(mapping const&, mapping const&) = default;
    };
};

template <class Mapping> inline constexpr bool forward_layout<RepeatFirstAxis<Mapping>> = ForwardMapping<Mapping>;

// The caller supplies [repetitions, source extents...]. Only const descriptors are broadcast.
template <MdspecLike Span, class Extents>
  requires std::is_const_v<typename Span::element_type>
[[nodiscard]] constexpr auto repeat_first_axis(Span const& span, Extents shape)
{
  using Layout = RepeatFirstAxis<typename Span::mapping_type>;
  using A = typename Span::accessor_type;
  using T = typename Span::element_type;
  // The vector's physical stride can be wider than the matrix's logical extents.
  using Index = std::common_type_t<std::ptrdiff_t, std::make_signed_t<typename Extents::index_type>,
                                   std::make_signed_t<typename Span::index_type>>;
  using Shape = dextents<Index, Extents::rank()>;
  typename Layout::template mapping<Shape> map{Shape(shape), span.mapping()};
  if constexpr (MdspanLike<Span>)
    return mdspan<T, Shape, Layout, A>{span.data_handle(), map, span.accessor()};
  else
    return mdspec<T, Shape, Layout, A, typename Span::data_descriptor_type>{span.data_descriptor(), map,
                                                                            span.accessor()};
}
// Broadcast a single entry of a vector across its original shape using a read-only offset descriptor.
template <RankedStridedMdspecLike<1> S>
  requires std::is_const_v<typename S::element_type>
[[nodiscard]] constexpr auto repeat_vector_element(S const& in, typename S::index_type index)
{
  using A = typename S::accessor_type;
  using T = typename S::element_type;
  using E = extents<typename S::index_type>;
  layout_left::mapping<E> scalar_map{E{}};
  auto const offset = in.mapping()(index);
  auto scalar = [&] {
    if constexpr (MdspanLike<S>)
    {
      using OffsetAccessor = typename A::offset_policy;
      return mdspan<T, E, layout_left, OffsetAccessor>{in.accessor().offset(in.data_handle(), offset), scalar_map,
                                                       OffsetAccessor(in.accessor())};
    }
    else
      return mdspec<T, E, layout_left, A, typename S::data_descriptor_type>{in.data_descriptor().offset_by(offset),
                                                                            scalar_map, in.accessor()};
  }();
  return repeat_first_axis(scalar, in.extents());
}
} // namespace uni20::detail
