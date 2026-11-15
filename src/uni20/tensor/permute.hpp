#pragma once

/**
 * \file permute.hpp
 * \ingroup tensor
 * \brief Borrowed axis permutations of strided dense tensors.
 */

#include <uni20/mdspan/permute.hpp>
#include <uni20/tensor/mdspec_tensor_view.hpp>

namespace uni20
{
/// \brief Reorder dense tensor axes without moving numerical elements.
/// \details Output axis j is source axis Axis[j]. The returned descriptor
///          preserves the accessor, placement, backend selector, and live
///          default precision. Const sources produce read-only views. Use
///          make_tensor(result) for an owning materialization.
///          The source storage must outlive the result; moving or structurally
///          modifying its owner invalidates the view. Owning temporaries are
///          rejected; temporary descriptor views may be composed safely.
/// \tparam Axis A bijection of the source axes.
/// \param tensor Strided tensor lvalue or borrowed descriptor temporary.
template <std::size_t... Axis, TensorView Tensor>
  requires((std::is_lvalue_reference_v<Tensor &&> ||
            requires { typename std::remove_cvref_t<Tensor>::borrowed_tensor_view_tag; }) &&
           requires(Tensor& tensor) {
             uni20::permute<Axis...>(mdspec_of(tensor));
             uni20::permute<Axis...>(mdspec_of(std::as_const(tensor)));
           })
[[nodiscard]] constexpr auto permute(Tensor&& tensor)
{
  auto mutable_mdspec = uni20::permute<Axis...>(mdspec_of(tensor));
  auto const_mdspec = uni20::permute<Axis...>(mdspec_of(std::as_const(tensor)));
  using selector_type = std::remove_cvref_t<decltype(tensor.backend_selector())>;
  return MdspecTensorView<decltype(mutable_mdspec), decltype(const_mdspec), tensor_storage_policy_t<Tensor>,
                          selector_type>{std::move(mutable_mdspec), std::move(const_mdspec),
                                         detail::descriptor_precision_source(tensor), tensor.backend_selector()};
}
} // namespace uni20
