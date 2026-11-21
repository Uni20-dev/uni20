#pragma once

/**
 * \file staging_backends.hpp
 * \ingroup linalg
 * \brief Backend values for CUDA copies that stage through a buffer and dispatch each step.
 */

#include <uni20/linalg/backend_selector.hpp>

#include <string_view>
#include <type_traits>

namespace uni20::linalg
{

/// \brief Copy between two CUDA devices through a staging buffer on the destination device.
/// \details Accepts device-to-device copies whose operands are on different
///          devices. It peer-copies the source's storage range into a buffer on
///          the destination device, then copies that buffer into the
///          destination there. Both steps dispatch through
///          `device_copy_selector`, which never contains this backend: the
///          `CudaStorage` copy default appends it to the storage selector and
///          retains that selector unchanged. A direct kernel for the same
///          instances belongs ahead of it in the outer list.
/// \tparam DeviceCopySelector Selector for the two same-storage copy steps.
template <KernelBackendSelector DeviceCopySelector> struct CudaPeerStagingBackend
{
    static constexpr std::string_view name = "cuda_peer_staging";

    DeviceCopySelector device_copy_selector;

    friend constexpr bool operator==(CudaPeerStagingBackend const&, CudaPeerStagingBackend const&) = default;
};

template <class DeviceCopySelector>
CudaPeerStagingBackend(DeviceCopySelector) -> CudaPeerStagingBackend<std::remove_cvref_t<DeviceCopySelector>>;

/// \brief Copy between host and CUDA storage through one contiguous transfer.
/// \details Accepts host-to-device and device-to-host copies in any layout. The
///          transfer moves one contiguous range in the host operand's element
///          order when that is contiguous, otherwise the device operand's,
///          otherwise column-major. A host copy packs or unpacks the host side
///          through `host_copy_selector`, `transfer_selector` moves the range,
///          and a device copy changes the device side's layout through
///          `device_copy_selector`. An operand that already has the transfer
///          order takes part in the transfer directly. Only the last step
///          writes the output, so a decline by any step leaves it unchanged.
/// \tparam DeviceCopySelector Selector for the CUDA-to-CUDA layout step.
/// \tparam HostCopySelector Selector for the host-to-host layout step.
/// \tparam TransferSelector Selector for the contiguous host-device step.
template <KernelBackendSelector DeviceCopySelector, KernelBackendSelector HostCopySelector,
          KernelBackendSelector TransferSelector>
struct CudaHostStagingBackend
{
    static constexpr std::string_view name = "cuda_host_staging";

    DeviceCopySelector device_copy_selector;
    HostCopySelector host_copy_selector;
    TransferSelector transfer_selector;

    friend constexpr bool operator==(CudaHostStagingBackend const&, CudaHostStagingBackend const&) = default;
};

template <class DeviceCopySelector, class HostCopySelector, class TransferSelector>
CudaHostStagingBackend(DeviceCopySelector, HostCopySelector, TransferSelector)
    -> CudaHostStagingBackend<std::remove_cvref_t<DeviceCopySelector>, std::remove_cvref_t<HostCopySelector>,
                              std::remove_cvref_t<TransferSelector>>;

} // namespace uni20::linalg
