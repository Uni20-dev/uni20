#pragma once

/**
 * \file lq.hpp
 * \ingroup linalg
 * \brief Async Tensor wrappers for preserving and consuming reduced real or complex LQ factorization.
 * \details Precision may be inferred from the input matrix or supplied explicitly.
 *          Fixed results require their native precision; runtime decompositions
 *          require finite precision. Owning results record the selected default.
 *          Inference occurs only after the input epoch is readable.
 */

#include <uni20/async/async.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/linalg/ops/lq.hpp>

#include <type_traits>
#include <utility>

namespace uni20::linalg
{
namespace detail
{
template <class MatrixTensor>
using preserving_lq_result_t = std::remove_cvref_t<decltype(uni20::linalg::lq(std::declval<MatrixTensor const&>()))>;

template <class MatrixTensor>
using consuming_lq_result_t = std::remove_cvref_t<decltype(uni20::linalg::lq(std::declval<MatrixTensor&&>()))>;

template <class Result>
using async_lq_result_t =
    LqResult<async::Async<typename Result::l_tensor_type>, async::Async<typename Result::q_tensor_type>>;

template <KernelBackendSelector BackendSelector, class LTensor, class QTensor, uni20::RankedTensorView<2> MatrixTensor>
async::AsyncTask co_preserving_lq(BackendSelector const selector, async::WriteBuffer<LTensor> l,
                                  async::WriteBuffer<QTensor> q, async::ReadBuffer<MatrixTensor> matrix,
                                  std::optional<Precision> requested = {})
{
  auto l_storage_awaiter = l.storage();
  auto q_storage_awaiter = q.storage();
  auto& l_storage = co_await l_storage_awaiter;
  auto& q_storage = co_await q_storage_awaiter;
  auto const& matrix_value = co_await matrix;

  auto result = uni20::linalg::lq(selector, matrix_value, requested);
  l_storage.emplace(std::move(result.l));
  q_storage.emplace(std::move(result.q));
  co_return;
}

template <KernelBackendSelector BackendSelector, class LTensor, class QTensor,
          uni20::MutableRankedTensorView<2> MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor>
async::AsyncTask co_consuming_lq(BackendSelector const selector, async::WriteBuffer<LTensor> l,
                                 async::WriteBuffer<QTensor> q, async::WriteBuffer<MatrixTensor> matrix,
                                 std::optional<Precision> requested = {})
{
  auto l_storage_awaiter = l.storage();
  auto q_storage_awaiter = q.storage();
  auto& l_storage = co_await l_storage_awaiter;
  auto& q_storage = co_await q_storage_awaiter;
  // Retain the consumed writer gate until both outputs commit, so a failure is
  // published to the input and both independent result epochs.
  auto matrix_value = co_await matrix.take();

  auto result = uni20::linalg::lq(selector, std::move(matrix_value), requested);
  l_storage.emplace(std::move(result.l));
  q_storage.emplace(std::move(result.q));
  co_return;
}

template <KernelBackendSelector BackendSelector, uni20::RankedTensorView<2> MatrixTensor>
[[nodiscard]] auto schedule_preserving_lq(BackendSelector selector, async::Async<MatrixTensor> const& matrix,
                                          std::optional<Precision> requested = {})
{
  using result_type = preserving_lq_result_t<MatrixTensor>;
  async_lq_result_t<result_type> outputs;
  outputs.l.debug_name("lq.l");
  outputs.q.debug_name("lq.q");

  auto task = co_preserving_lq(std::move(selector), outputs.l.write(), outputs.q.write(), matrix.read(), requested);
  task.debug_name("lq");
  async::schedule(std::move(task));
  return outputs;
}

template <KernelBackendSelector BackendSelector, uni20::MutableRankedTensorView<2> MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor>
[[nodiscard]] auto schedule_consuming_lq(BackendSelector selector, async::Async<MatrixTensor>& matrix,
                                         std::optional<Precision> requested = {})
{
  using result_type = consuming_lq_result_t<MatrixTensor>;
  async_lq_result_t<result_type> outputs;
  outputs.l.debug_name("lq.l");
  outputs.q.debug_name("lq.q");

  auto task = co_consuming_lq(std::move(selector), outputs.l.write(), outputs.q.write(), matrix.write(), requested);
  task.debug_name("lq");
  async::schedule(std::move(task));
  return outputs;
}
} // namespace detail

/// \brief Schedule a preserving reduced real or complex LQ factorization through an explicit selector.
/// \details The returned `L` and `Q` handles have independent output epochs and
///          both receive any unhandled task failure.
template <KernelBackendSelector BackendSelector, uni20::RankedTensorView<2> MatrixTensor>
  requires detail::DecompositionScalar<uni20::tensor_element_t<MatrixTensor>>
[[nodiscard]] auto lq(BackendSelector selector, async::Async<MatrixTensor> const& matrix,
                      std::optional<Precision> requested = {})
{
  return detail::schedule_preserving_lq(std::move(selector), matrix, requested);
}

/// \brief Schedule a preserving reduced real or complex LQ factorization using the static Tensor selector.
/// \details Structured binding yields two independent `Async<Tensor>` values:
///          `auto [l, q] = lq(matrix);`.
template <uni20::RankedTensorView<2> MatrixTensor>
  requires detail::DecompositionScalar<uni20::tensor_element_t<MatrixTensor>>
[[nodiscard]] auto lq(async::Async<MatrixTensor> const& matrix, std::optional<Precision> requested = {})
{
  using work_type = detail::lq_matrix_tensor_t<MatrixTensor>;
  auto selector = detail::select_lq_backend<work_type>();
  return detail::schedule_preserving_lq(std::move(selector), matrix, requested);
}

/// \brief Schedule a consuming reduced real or complex LQ factorization through an explicit selector.
/// \details Passing the async matrix as an rvalue grants permission to remove
///          its stored owning value. The input writer and both output writers
///          receive any failure that occurs after enrollment.
template <KernelBackendSelector BackendSelector, class MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor> && uni20::MutableRankedTensorView<MatrixTensor, 2> &&
           detail::DecompositionScalar<uni20::tensor_element_t<MatrixTensor>>
[[nodiscard]] auto lq(BackendSelector selector, async::Async<MatrixTensor>&& matrix,
                      std::optional<Precision> requested = {})
{
  return detail::schedule_consuming_lq(std::move(selector), matrix, requested);
}

/// \brief Schedule a consuming reduced real or complex LQ factorization using the static Tensor selector.
template <class MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor> && uni20::MutableRankedTensorView<MatrixTensor, 2> &&
           detail::DecompositionScalar<uni20::tensor_element_t<MatrixTensor>>
[[nodiscard]] auto lq(async::Async<MatrixTensor>&& matrix, std::optional<Precision> requested = {})
{
  using work_type = std::conditional_t<detail::can_use_lq_storage_as_workspace<MatrixTensor>(),
                                       std::remove_cvref_t<MatrixTensor>, detail::lq_matrix_tensor_t<MatrixTensor>>;
  auto selector = detail::select_lq_backend<work_type>();
  return detail::schedule_consuming_lq(std::move(selector), matrix, requested);
}

} // namespace uni20::linalg
