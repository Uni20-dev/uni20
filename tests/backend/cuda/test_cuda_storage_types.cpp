#include <uni20/storage/cuda_storage.hpp>
#include <uni20/tensor/tensor.hpp>

#if UNI20_ENABLE_MPFR
#include <uni20/core/mpreal.hpp>
#endif
#if UNI20_ENABLE_MPC
#include <uni20/core/mpcomplex.hpp>
#endif

#include <concepts>
#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>

namespace
{
// Exercise each public boundary separately: combining these requirements would
// let one constrained interface hide an unconstrained buffer or borrowed view.
template <class T> consteval bool rejects_cuda_storage()
{
  static_assert(!requires { typename uni20::cuda::CudaBuffer<T>; });
  static_assert(!requires { typename uni20::cuda::PartitionedCudaBuffer<T>; });
  static_assert(!requires { typename uni20::CudaStorage::storage_t<T>; });
  static_assert(!requires { typename uni20::CudaStorage::packed_storage_t<T>; });
  static_assert(!requires { typename uni20::cuda::CudaPointerAccessor<T>; });
  static_assert(!requires { typename uni20::cuda::CudaPointerAccessor<T const>; });
  static_assert(!requires { typename uni20::CudaTensor<T, 2>; });
  static_assert(!requires { typename uni20::CudaMatrix<T>; });
  return true;
}

static_assert(rejects_cuda_storage<std::unique_ptr<double>>());
static_assert(rejects_cuda_storage<std::string>());
#if UNI20_ENABLE_MPFR
static_assert(rejects_cuda_storage<uni20::mpreal>());
#endif
#if UNI20_ENABLE_MPC
static_assert(rejects_cuda_storage<uni20::complex<uni20::mpreal>>());
template <class R>
concept CanMakeComplexCudaProxy = requires { typename uni20::cuda::CudaComplexReference<R>; };
template <class R>
concept CanMakeConjugatingCudaAccessor = requires { typename uni20::cuda::CudaConjugatingPointerAccessor<R>; };
static_assert(!CanMakeComplexCudaProxy<uni20::mpreal>);
static_assert(!CanMakeConjugatingCudaAccessor<uni20::mpreal>);
#endif

struct TrivialRecord
{
    int index;
    double value;
};

// Storage can hold trivial records as well as scalars. Kernel arithmetic support
// is a separate question; avoid replacing the representation rule with a list
// of scalar types supported by a particular provider.
template <class T> consteval bool accepts_cuda_storage()
{
  static_assert(std::is_constructible_v<uni20::cuda::CudaBuffer<T>, std::size_t>);
  static_assert(std::is_move_constructible_v<uni20::cuda::PartitionedCudaBuffer<T>>);
  static_assert(std::is_constructible_v<uni20::CudaMatrix<T>, uni20::uninitialized_t, int, int>);
  static_assert(std::same_as<typename uni20::cuda::CudaPointerAccessor<T>::element_type, T>);
  static_assert(std::same_as<typename uni20::cuda::CudaPointerAccessor<T const>::element_type, T const>);
  return true;
}

static_assert(accepts_cuda_storage<float>());
static_assert(accepts_cuda_storage<double>());
static_assert(accepts_cuda_storage<uni20::complex<float>>());
static_assert(accepts_cuda_storage<uni20::complex<double>>());
static_assert(accepts_cuda_storage<TrivialRecord>());
} // namespace
