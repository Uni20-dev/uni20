#pragma once

/**
 * \file qr.hpp
 * \ingroup linalg
 * \brief Async Tensor wrappers for preserving and consuming reduced real or complex QR factorization.
 * \details Precision may be inferred from the input matrix or supplied explicitly.
 *          Fixed results require their native precision; runtime decompositions
 *          require finite precision. Owning results record the selected default.
 *          Inference occurs only after the input epoch is readable.
 */

#include <uni20/async/async.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/linalg/ops/qr.hpp>

#include <type_traits>
#include <utility>

namespace uni20::linalg
{
namespace detail
{
template <class MatrixTensor>
using preserving_qr_result_t = std::remove_cvref_t<decltype(uni20::linalg::qr(std::declval<MatrixTensor const&>()))>;

template <class MatrixTensor>
using consuming_qr_result_t = std::remove_cvref_t<decltype(uni20::linalg::qr(std::declval<MatrixTensor&&>()))>;

template <class Result>
using async_qr_result_t =
    QrResult<async::Async<typename Result::q_tensor_type>, async::Async<typename Result::r_tensor_type>>;

template <KernelBackendSelector BackendSelector, class QTensor, class RTensor, uni20::RankedTensorView<2> MatrixTensor>
async::AsyncTask co_preserving_qr(BackendSelector const selector, async::WriteBuffer<QTensor> q,
                                  async::WriteBuffer<RTensor> r, async::ReadBuffer<MatrixTensor> matrix,
                                  std::optional<Precision> requested = {})
{
  auto q_storage_awaiter = q.storage();
  auto r_storage_awaiter = r.storage();
  auto& q_storage = co_await q_storage_awaiter;
  auto& r_storage = co_await r_storage_awaiter;
  auto const& matrix_value = co_await matrix;

  auto result = uni20::linalg::qr(selector, matrix_value, requested);
  q_storage.emplace(std::move(result.q));
  r_storage.emplace(std::move(result.r));
  co_return;
}

template <KernelBackendSelector BackendSelector, class QTensor, class RTensor,
          uni20::MutableRankedTensorView<2> MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor>
async::AsyncTask co_consuming_qr(BackendSelector const selector, async::WriteBuffer<QTensor> q,
                                 async::WriteBuffer<RTensor> r, async::WriteBuffer<MatrixTensor> matrix,
                                 std::optional<Precision> requested = {})
{
  auto q_storage_awaiter = q.storage();
  auto r_storage_awaiter = r.storage();
  auto& q_storage = co_await q_storage_awaiter;
  auto& r_storage = co_await r_storage_awaiter;
  // Retain the consumed writer gate until both outputs commit, so a failure is
  // published to the input and both independent result epochs.
  auto matrix_value = co_await matrix.take();

  auto result = uni20::linalg::qr(selector, std::move(matrix_value), requested);
  q_storage.emplace(std::move(result.q));
  r_storage.emplace(std::move(result.r));
  co_return;
}

template <KernelBackendSelector BackendSelector, uni20::RankedTensorView<2> MatrixTensor>
[[nodiscard]] auto schedule_preserving_qr(BackendSelector selector, async::Async<MatrixTensor> const& matrix,
                                          std::optional<Precision> requested = {})
{
  using result_type = preserving_qr_result_t<MatrixTensor>;
  async_qr_result_t<result_type> outputs;
  outputs.q.debug_name("qr.q");
  outputs.r.debug_name("qr.r");

  auto task = co_preserving_qr(std::move(selector), outputs.q.write(), outputs.r.write(), matrix.read(), requested);
  task.debug_name("qr");
  async::schedule(std::move(task));
  return outputs;
}

template <KernelBackendSelector BackendSelector, uni20::MutableRankedTensorView<2> MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor>
[[nodiscard]] auto schedule_consuming_qr(BackendSelector selector, async::Async<MatrixTensor>& matrix,
                                         std::optional<Precision> requested = {})
{
  using result_type = consuming_qr_result_t<MatrixTensor>;
  async_qr_result_t<result_type> outputs;
  outputs.q.debug_name("qr.q");
  outputs.r.debug_name("qr.r");

  auto task = co_consuming_qr(std::move(selector), outputs.q.write(), outputs.r.write(), matrix.write(), requested);
  task.debug_name("qr");
  async::schedule(std::move(task));
  return outputs;
}
} // namespace detail

/// \brief Schedule a preserving reduced real or complex QR factorization through an explicit selector.
/// \details The returned `Q` and `R` handles have independent output epochs and
///          both receive any unhandled task failure.
template <KernelBackendSelector BackendSelector, uni20::RankedTensorView<2> MatrixTensor>
  requires detail::DecompositionScalar<uni20::tensor_element_t<MatrixTensor>>
[[nodiscard]] auto qr(BackendSelector selector, async::Async<MatrixTensor> const& matrix,
                      std::optional<Precision> requested = {})
{
  return detail::schedule_preserving_qr(std::move(selector), matrix, requested);
}

/// \brief Schedule a preserving reduced real or complex QR factorization using the static Tensor selector.
/// \details Structured binding yields two independent `Async<Tensor>` values:
///          `auto [q, r] = qr(matrix);`.
template <uni20::RankedTensorView<2> MatrixTensor>
  requires detail::DecompositionScalar<uni20::tensor_element_t<MatrixTensor>>
[[nodiscard]] auto qr(async::Async<MatrixTensor> const& matrix, std::optional<Precision> requested = {})
{
  using work_type = detail::qr_matrix_tensor_t<MatrixTensor>;
  auto selector = detail::select_qr_backend<work_type>();
  return detail::schedule_preserving_qr(std::move(selector), matrix, requested);
}

/// \brief Schedule a consuming reduced real or complex QR factorization through an explicit selector.
/// \details Passing the async matrix as an rvalue grants permission to remove
///          its stored owning value. The input writer and both output writers
///          receive any failure that occurs after enrollment.
template <KernelBackendSelector BackendSelector, class MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor> && uni20::MutableRankedTensorView<MatrixTensor, 2> &&
           detail::DecompositionScalar<uni20::tensor_element_t<MatrixTensor>>
[[nodiscard]] auto qr(BackendSelector selector, async::Async<MatrixTensor>&& matrix,
                      std::optional<Precision> requested = {})
{
  return detail::schedule_consuming_qr(std::move(selector), matrix, requested);
}

/// \brief Schedule a consuming reduced real or complex QR factorization using the static Tensor selector.
template <class MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor> && uni20::MutableRankedTensorView<MatrixTensor, 2> &&
           detail::DecompositionScalar<uni20::tensor_element_t<MatrixTensor>>
[[nodiscard]] auto qr(async::Async<MatrixTensor>&& matrix, std::optional<Precision> requested = {})
{
  using work_type = std::conditional_t<detail::can_use_qr_storage_as_workspace<MatrixTensor>(),
                                       std::remove_cvref_t<MatrixTensor>, detail::qr_matrix_tensor_t<MatrixTensor>>;
  auto selector = detail::select_qr_backend<work_type>();
  return detail::schedule_consuming_qr(std::move(selector), matrix, requested);
}

} // namespace uni20::linalg
