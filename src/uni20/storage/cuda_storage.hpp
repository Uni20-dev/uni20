#pragma once

/**
 * \file cuda_storage.hpp
 * \ingroup tensor
 * \brief CUDA Tensor storage with deferred device-memory data descriptors.
 */

#include <uni20/backend/cuda/buffer.hpp>
#include <uni20/backend/cuda/initialization.hpp>
#include <uni20/common/initialization.hpp>
#include <uni20/linalg/backend_selector.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/storage/cuda_accessor.hpp>

#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace uni20::cuda
{

/// \brief Non-owning opaque view into one logical CUDA buffer.
/// \details The view carries completion-domain identity and an element offset, but
///          deliberately exposes no raw device pointer or host-side value
///          conversion. CUDA lowering must first acquire synchronized access
///          to `buffer()` on an operation stream.
template <class ElementType> class CudaBufferView {
  public:
    using element_type = ElementType;
    using value_type = std::remove_const_t<element_type>;
    using buffer_type = CudaBuffer<value_type>;
    using buffer_reference = std::conditional_t<std::is_const_v<element_type>, buffer_type const&, buffer_type&>;

    constexpr CudaBufferView() noexcept = default;

    /// \brief Construct a view at the beginning of a CUDA buffer.
    explicit constexpr CudaBufferView(buffer_reference buffer) noexcept : buffer_(&buffer) {}

    /// \brief Convert a mutable view into a read-only view.
    template <class OtherElement>
      requires(std::is_const_v<element_type> && !std::is_const_v<OtherElement> &&
               std::same_as<std::remove_const_t<OtherElement>, value_type>)
    constexpr CudaBufferView(CudaBufferView<OtherElement> other) noexcept
        : buffer_(other.buffer_), offset_(other.offset_)
    {}

    /// \brief Return whether this view identifies a CUDA buffer.
    [[nodiscard]] explicit constexpr operator bool() const noexcept { return buffer_ != nullptr; }

    /// \brief Return the CUDA buffer whose completion ledger governs access.
    [[nodiscard]] auto buffer() const -> buffer_reference
    {
      CHECK(buffer_ != nullptr, "cannot resolve an empty CUDA buffer view");
      return *buffer_;
    }

    /// \brief Return this view's element offset from the logical buffer start.
    [[nodiscard]] constexpr std::size_t element_offset() const noexcept { return offset_; }

    /// \brief Return a view advanced by an element offset.
    /// \pre The resulting element offset is representable by `std::size_t`.
    [[nodiscard]] constexpr CudaBufferView offset_by(std::size_t offset) const noexcept
    {
      CudaBufferView result = *this;
      result.offset_ += offset;
      return result;
    }

    friend constexpr bool operator==(CudaBufferView const&, CudaBufferView const&) = default;

  private:
    using buffer_pointer = std::add_pointer_t<std::remove_reference_t<buffer_reference>>;

    buffer_pointer buffer_ = nullptr;
    std::size_t offset_ = 0;

    template <class> friend class CudaBufferView;
};

} // namespace uni20::cuda

namespace uni20::cuda
{

namespace detail
{

// These representations have a numerical zero consisting entirely of zero
// bytes. Do not infer that property from a user-specialized scalar trait.
template <class T>
inline constexpr bool has_zero_object_representation_v = [] {
#if UNI20_HAS_FLOAT128
  if constexpr (std::same_as<T, uni20::float128>) return true;
#endif
  return std::is_arithmetic_v<T>;
}();

template <class Real>
inline constexpr bool has_zero_object_representation_v<uni20::complex<Real>> = has_zero_object_representation_v<Real>;

template <class Descriptor> struct IsCudaBufferView : std::false_type
{};

template <class ElementType> struct IsCudaBufferView<CudaBufferView<ElementType>> : std::true_type
{};

} // namespace detail

/// \brief CUDA-accessible mdspec backed by a `CudaBufferView` descriptor.
template <class Span>
concept BufferMdspec = uni20::CudaAccessibleMdspec<Span> && requires {
  typename std::remove_cvref_t<Span>::data_descriptor_type;
} && detail::IsCudaBufferView<typename std::remove_cvref_t<Span>::data_descriptor_type>::value;

} // namespace uni20::cuda

namespace uni20::cuda
{

/// \brief Factory that resolves CUDA accessors for Tensor descriptors.
struct CudaAccessorFactory
{
    template <class ElementType> using accessor_t = CudaPointerAccessor<ElementType>;
    template <class ElementType> using device_accessor_t = CudaPointerAccessor<ElementType>;

    template <class ElementType, class Storage>
    [[nodiscard]] constexpr auto make_accessor(Storage const&) const noexcept -> accessor_t<ElementType>
    {
      return accessor_t<ElementType>{};
    }

    template <class ElementType, class Storage>
    [[nodiscard]] constexpr auto make_device_accessor(Storage const&) const noexcept -> device_accessor_t<ElementType>
    {
      return device_accessor_t<ElementType>{};
    }
};

} // namespace uni20::cuda

namespace uni20
{

/// \brief Tensor storage policy for device-resident CUDA values.
/// \details Ordinary allocation uses the installed CUDA runtime's default
///          device. An explicit `cuda::DeviceResources` selects another enrolled
///          device or an isolated resource set used by tests. The policy selects
///          CUDA backends and exposes deferred `CudaBufferView` descriptors with
///          eventual pointer accessors. Stream and provider-resource admission
///          remains operation-local. Direct Tensor operations may block during
///          admission. `Async<Tensor>` operations use coroutine-aware dispatch
///          when a backend provides it, while retaining the same storage and
///          mdspec representation.
struct CudaStorage
{
    /// \brief Byte alignment guaranteed by CUDA runtime device allocations.
    static constexpr std::size_t allocation_alignment = 256;

    using context_type = cuda::DeviceResources;
    using accessor_factory_type = cuda::CudaAccessorFactory;
#if UNI20_BACKEND_CUSOLVER && UNI20_BACKEND_CUBLAS
    using backend_selector_type =
        linalg::backend_list<linalg::CusolverBackend, linalg::CublasBackend, linalg::CudaReferenceBackend>;
#elif UNI20_BACKEND_CUSOLVER
    using backend_selector_type = linalg::backend_list<linalg::CusolverBackend, linalg::CudaReferenceBackend>;
#elif UNI20_BACKEND_CUBLAS
    using backend_selector_type = linalg::backend_list<linalg::CublasBackend, linalg::CudaReferenceBackend>;
#else
    using backend_selector_type = linalg::backend_list<linalg::CudaReferenceBackend>;
#endif

    template <class ElementType> using storage_t = cuda::CudaBuffer<ElementType>;
    template <class ElementType> using packed_storage_t = cuda::PartitionedCudaBuffer<ElementType>;

    template <class ElementType>
    [[nodiscard]] static auto
    make_storage(std::size_t size,
                 StorageInitialization initialization = StorageInitialization::Uninitialized) -> storage_t<ElementType>
    {
      return make_storage<ElementType>(cuda::device_resources(), size, initialization);
    }

    template <class ElementType>
    [[nodiscard]] static auto
    make_storage(context_type& context, std::size_t size,
                 StorageInitialization initialization = StorageInitialization::Uninitialized) -> storage_t<ElementType>
    {
      storage_t<ElementType> result{context, size};
      initialize_storage(result, initialization);
      return result;
    }

    template <class ElementType>
    [[nodiscard]] static auto
    make_storage(cuda::Device device, std::size_t size,
                 StorageInitialization initialization = StorageInitialization::Uninitialized) -> storage_t<ElementType>
    {
      return make_storage<ElementType>(cuda::device_resources(device.ordinal()), size, initialization);
    }

    template <class ElementType>
    [[nodiscard]] static auto make_storage_like(
        storage_t<ElementType> const& storage, std::size_t size,
        StorageInitialization initialization = StorageInitialization::Uninitialized) -> storage_t<ElementType>
    {
      return make_storage<ElementType>(storage.resources(), size, initialization);
    }

    template <class ElementType>
    [[nodiscard]] static auto make_packed_storage(
        std::size_t size, std::span<std::size_t const> offsets,
        StorageInitialization initialization = StorageInitialization::Uninitialized) -> packed_storage_t<ElementType>
    {
      return make_packed_storage<ElementType>(cuda::device_resources(), size, offsets, initialization);
    }

    template <class ElementType>
    [[nodiscard]] static auto make_packed_storage(
        cuda::Device device, std::size_t size, std::span<std::size_t const> offsets,
        StorageInitialization initialization = StorageInitialization::Uninitialized) -> packed_storage_t<ElementType>
    {
      return make_packed_storage<ElementType>(cuda::device_resources(device.ordinal()), size, offsets, initialization);
    }

    template <class ElementType>
    [[nodiscard]] static auto make_packed_storage(
        context_type& context, std::size_t size, std::span<std::size_t const> offsets,
        StorageInitialization initialization = StorageInitialization::Uninitialized) -> packed_storage_t<ElementType>
    {
      std::vector<cuda::CudaBufferRange> ranges;
      if (!offsets.empty())
      {
        CHECK_EQUAL(offsets.back(), size);
        ranges.reserve(offsets.size() - 1);
        for (std::size_t ordinal = 0; ordinal + 1 < offsets.size(); ++ordinal)
          ranges.push_back(
              cuda::CudaBufferRange{.offset = offsets[ordinal], .size = offsets[ordinal + 1] - offsets[ordinal]});
      }
      packed_storage_t<ElementType> result{context, size, ranges};
      initialize_storage(result, initialization);
      return result;
    }

    template <class ElementType>
    [[nodiscard]] static auto make_packed_storage_like(
        packed_storage_t<ElementType> const& storage, std::size_t size, std::span<std::size_t const> offsets,
        StorageInitialization initialization = StorageInitialization::Uninitialized) -> packed_storage_t<ElementType>
    {
      return make_packed_storage<ElementType>(storage.resources(), size, offsets, initialization);
    }

    /// \brief Return the allocation context retained by a packed CUDA buffer.
    template <class ElementType>
    [[nodiscard]] static auto allocation_context(packed_storage_t<ElementType> const& storage) noexcept -> context_type&
    {
      return storage.resources();
    }

    /// \brief Initialize only the alignment gaps in a packed CUDA allocation.
    /// \details Uni20's real and complex scalar zeros have an all-zero CUDA
    ///          object representation. Block payloads remain uninitialized.
    template <RealOrComplex ElementType>
    static void initialize_packed_padding(packed_storage_t<ElementType>& storage, std::span<std::size_t const> offsets,
                                          std::span<std::size_t const> block_ends)
    {
      CHECK_EQUAL(offsets.size(), block_ends.size() + 1);
      bool has_padding = false;
      for (std::size_t ordinal = 0; ordinal < block_ends.size(); ++ordinal)
      {
        CHECK(block_ends[ordinal] <= offsets[ordinal + 1], block_ends[ordinal], offsets[ordinal + 1]);
        has_padding = has_padding || block_ends[ordinal] != offsets[ordinal + 1];
      }
      if (!has_padding) return;

      auto stream = storage.resources().streams().acquire();
      auto access = storage.write_synchronized_with(stream);
      cuda::ScopedDevice guard(storage.device().ordinal());
      for (std::size_t ordinal = 0; ordinal < block_ends.size(); ++ordinal)
      {
        auto const padding_size = offsets[ordinal + 1] - block_ends[ordinal];
        if (padding_size == 0) continue;
        cuda::check(cudaMemsetAsync(access.data() + block_ends[ordinal], 0, padding_size * sizeof(ElementType),
                                    stream.native_handle()),
                    "zero packed CUDA BlockTensor padding", stream.device());
      }
    }

    template <class ElementType>
    [[nodiscard]] static bool storage_is_compatible(storage_t<ElementType> const& storage, cuda::Device device)
    {
      return storage.device() == device;
    }

    template <class ElementType>
    [[nodiscard]] static auto
    make_data_descriptor(storage_t<ElementType>& storage) noexcept -> cuda::CudaBufferView<ElementType>
    {
      return cuda::CudaBufferView<ElementType>{storage};
    }

    template <class ElementType>
    [[nodiscard]] static auto
    make_data_descriptor(storage_t<ElementType> const& storage) noexcept -> cuda::CudaBufferView<ElementType const>
    {
      return cuda::CudaBufferView<ElementType const>{storage};
    }

    template <class ElementType>
    [[nodiscard]] static auto
    make_packed_data_descriptor(packed_storage_t<ElementType>& storage,
                                std::size_t ordinal) noexcept -> cuda::CudaBufferView<ElementType>
    {
      return cuda::CudaBufferView<ElementType>{storage.buffer(ordinal)};
    }

    template <class ElementType>
    [[nodiscard]] static auto
    make_packed_data_descriptor(packed_storage_t<ElementType> const& storage,
                                std::size_t ordinal) noexcept -> cuda::CudaBufferView<ElementType const>
    {
      return cuda::CudaBufferView<ElementType const>{storage.buffer(ordinal)};
    }

    /// \brief Return the ordered backend list for CUDA device storage.
    [[nodiscard]] static constexpr auto backend_selector() noexcept -> backend_selector_type
    {
#if UNI20_BACKEND_CUSOLVER && UNI20_BACKEND_CUBLAS
      return backend_selector_type{linalg::CusolverBackend{}, linalg::CublasBackend{}, linalg::CudaReferenceBackend{}};
#elif UNI20_BACKEND_CUSOLVER
      return backend_selector_type{linalg::CusolverBackend{}, linalg::CudaReferenceBackend{}};
#elif UNI20_BACKEND_CUBLAS
      return backend_selector_type{linalg::CublasBackend{}, linalg::CudaReferenceBackend{}};
#else
      return backend_selector_type{linalg::CudaReferenceBackend{}};
#endif
    }

  private:
    template <class Buffer> static void initialize_storage(Buffer& storage, StorageInitialization initialization)
    {
      using element_type = typename Buffer::element_type;
      if (storage.size() == 0) return;

      if (initialization == StorageInitialization::Zero)
      {
        if constexpr (cuda::detail::has_zero_object_representation_v<element_type>)
        {
          auto stream = storage.resources().streams().acquire();
          auto access = storage.write_synchronized_with(stream);
          cuda::ScopedDevice guard(storage.device().ordinal());
          cuda::check(cudaMemsetAsync(access.data(), 0, storage.size() * sizeof(element_type), stream.native_handle()),
                      "zero CUDA tensor storage", stream.device());
        }
        else
        {
          throw std::invalid_argument("CUDA zero initialization requires a supported numerical scalar representation");
        }
        return;
      }

#if UNI20_FILL_UNINITIALIZED_SNAN
      if constexpr (RealOrComplex<element_type>)
      {
        using real_type = make_real_t<element_type>;
        // These are the real precisions supported by the CUDA numerical
        // backends. Repeat the representation for both complex components.
        if constexpr (std::same_as<real_type, float> || std::same_as<real_type, double>)
        {
          using word_type = std::conditional_t<sizeof(real_type) == 4, std::uint32_t, std::uint64_t>;
          auto const pattern = std::bit_cast<word_type>(detail::signaling_nan_value<real_type>());
          auto stream = storage.resources().streams().acquire();
          auto access = storage.write_synchronized_with(stream);
          cuda::ScopedDevice guard(storage.device().ordinal());
          cuda::detail::enqueue_initialization_pattern(access.data(), storage.size() * sizeof(element_type), pattern,
                                                       sizeof(real_type), stream.native_handle(), stream.device());
        }
      }
#endif
    }
};

} // namespace uni20
