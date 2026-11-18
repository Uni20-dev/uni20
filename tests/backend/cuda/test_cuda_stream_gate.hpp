#pragma once

#include <uni20/backend/cuda/cuda_error.hpp>
#include <uni20/backend/cuda/runtime.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace uni20::test
{

/// \brief Holds a CUDA stream at a chosen point until the test opens the gate.
/// \details `close()` enqueues a write of an "entered" word followed by a
/// `cuStreamWaitValue32` on an "open" word. Both words are mapped pinned host
/// memory, so the stream waits on the device and the gate occupies neither
/// CUDA's host-function thread nor any GPU threads; `open()` is a plain host
/// store.
///
/// A host function must not serve as the gate. CUDA may run the host functions
/// of independent streams one at a time, so a host function that waits for the
/// test thread can hold back the stream-pool return callbacks that the test
/// thread itself is waiting for.
///
/// The words live in pinned slabs that are never freed, and a slot is never
/// reused, so a stream still reading a gate after its object is destroyed reads
/// valid memory. The destructor therefore only opens the gate. A full slab is
/// followed by a new one, so repeating tests in one process (`--gtest_repeat`)
/// costs 8 KiB of pinned memory per 1024 gates and never runs out.
///
/// The stream functions are obtained through the runtime's driver entry-point
/// query, so test executables do not link the driver library.
class CudaStreamGate {
  public:
    CudaStreamGate() : words_(allocate_slot()) {}
    ~CudaStreamGate() { this->open(); }

    CudaStreamGate(CudaStreamGate const&) = delete;
    CudaStreamGate& operator=(CudaStreamGate const&) = delete;

    /// \brief Make `stream` mark the gate entered and then wait until `open()`.
    void close(cudaStream_t stream, int device)
    {
      uni20::cuda::ScopedDevice guard(device);
      auto const& api = driver();
      check_driver(api.write_value(stream, device_address(entered_word(), device), 1, CU_STREAM_WRITE_VALUE_DEFAULT),
                   "cuStreamWriteValue32 stream gate entry", device);
      check_driver(api.wait_value(stream, device_address(open_word(), device), 1, CU_STREAM_WAIT_VALUE_GEQ),
                   "cuStreamWaitValue32 stream gate", device);
    }

    /// \brief Whether a stream closed on this gate has reached it.
    [[nodiscard]] bool entered() const noexcept
    {
      return std::atomic_ref(entered_word()).load(std::memory_order_acquire) != 0;
    }

    /// \brief Wait up to five seconds for a stream closed on this gate to reach it.
    [[nodiscard]] bool wait_entered() const
    {
      using namespace std::chrono_literals;
      auto const deadline = std::chrono::steady_clock::now() + 5s;
      while (!this->entered() && std::chrono::steady_clock::now() < deadline)
      {
        std::this_thread::yield();
      }
      return this->entered();
    }

    /// \brief Release every stream waiting on this gate.
    void open() noexcept { std::atomic_ref(open_word()).store(1, std::memory_order_release); }

  private:
    using stream_value_function = CUresult(CUDAAPI*)(CUstream, CUdeviceptr, cuuint32_t, unsigned int);

    struct DriverApi
    {
        stream_value_function wait_value = nullptr;
        stream_value_function write_value = nullptr;
    };

    static constexpr std::size_t slab_slots = 1024;

    [[nodiscard]] std::uint32_t& open_word() const noexcept { return words_[0]; }
    [[nodiscard]] std::uint32_t& entered_word() const noexcept { return words_[1]; }

    [[nodiscard]] static std::uint32_t* allocate_slot()
    {
      static std::mutex mutex;
      static std::uint32_t* slab = nullptr;
      static std::size_t next = slab_slots;

      std::lock_guard lock(mutex);
      if (next == slab_slots)
      {
        // Earlier slabs stay allocated: a stream may still read one of their gates.
        void* raw = nullptr;
        uni20::cuda::check(
            cudaHostAlloc(&raw, 2 * slab_slots * sizeof(std::uint32_t), cudaHostAllocMapped | cudaHostAllocPortable),
            "cudaHostAlloc stream gate words");
        slab = static_cast<std::uint32_t*>(raw);
        std::fill_n(slab, 2 * slab_slots, std::uint32_t{0});
        next = 0;
      }
      return slab + 2 * next++;
    }

    [[nodiscard]] static CUdeviceptr device_address(std::uint32_t& word, int device)
    {
      void* address = nullptr;
      uni20::cuda::check(cudaHostGetDevicePointer(&address, &word, 0), "cudaHostGetDevicePointer stream gate", device);
      return static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(address));
    }

    static void check_driver(CUresult status, char const* operation, int device)
    {
      if (status != CUDA_SUCCESS)
      {
        throw std::runtime_error(std::string(operation) + " failed on CUDA device " + std::to_string(device) +
                                 " with CUresult " + std::to_string(static_cast<int>(status)));
      }
    }

    template <class Function> [[nodiscard]] static Function load(char const* symbol)
    {
      void* function = nullptr;
      cudaDriverEntryPointQueryResult status = cudaDriverEntryPointSymbolNotFound;
#if CUDART_VERSION >= 12050
      uni20::cuda::check(cudaGetDriverEntryPointByVersion(symbol, &function, 12000, cudaEnableDefault, &status),
                         symbol);
#else
      uni20::cuda::check(cudaGetDriverEntryPoint(symbol, &function, cudaEnableDefault, &status), symbol);
#endif
      if (status != cudaDriverEntryPointSuccess || function == nullptr)
      {
        throw std::runtime_error(std::string("CUDA driver entry point unavailable: ") + symbol);
      }
      return reinterpret_cast<Function>(function);
    }

    [[nodiscard]] static DriverApi const& driver()
    {
      static DriverApi const api{load<stream_value_function>("cuStreamWaitValue32"),
                                 load<stream_value_function>("cuStreamWriteValue32")};
      return api;
    }

    std::uint32_t* words_;
};

} // namespace uni20::test
