#pragma once

#include <uni20/config.hpp>

#include <cstddef>

// Use the active translation unit's instrumentation, independently of build type.
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
#define UNI20_HAS_MEMORY_SANITIZER 1
#endif
#endif
#ifndef UNI20_HAS_MEMORY_SANITIZER
#define UNI20_HAS_MEMORY_SANITIZER 0
#endif

#if UNI20_HAS_MEMORY_SANITIZER
#include <sanitizer/msan_interface.h>
#endif
#if UNI20_ENABLE_VALGRIND
#include <valgrind/memcheck.h>
#endif

namespace uni20::detail::memory_diagnostics
{

/// \brief Mark writable storage undefined without changing its bytes.
/// \details Use when raw storage starts a new logical lifetime, after any diagnostic fill.
///          MSan records a new allocation origin when origin tracking is enabled.
inline void mark_uninitialized(void* data, std::size_t bytes) noexcept
{
  if (bytes == 0) return;
#if UNI20_HAS_MEMORY_SANITIZER
  __msan_allocated_memory(data, bytes);
#endif
#if UNI20_ENABLE_VALGRIND
  (void)VALGRIND_MAKE_MEM_UNDEFINED(data, bytes);
#endif
  (void)data;
}

/// \brief Diagnose undefined bytes before passing inputs to uninstrumented code.
/// \details Call only for the elements that the operation actually reads, excluding unused padding.
inline void check_initialized(void const* data, std::size_t bytes) noexcept
{
  if (bytes == 0) return;
#if UNI20_HAS_MEMORY_SANITIZER
  __msan_check_mem_is_initialized(data, bytes);
#endif
#if UNI20_ENABLE_VALGRIND
  (void)VALGRIND_CHECK_MEM_IS_DEFINED(data, bytes);
#endif
  (void)data;
}

/// \brief Acknowledge bytes defined by a completed external operation.
/// \details Do not apply to instrumented kernels or Memcheck-observed CPU routines: tracking their
///          actual writes detects missing initialization that this annotation would conceal.
inline void mark_initialized(void* data, std::size_t bytes) noexcept
{
  if (bytes == 0) return;
#if UNI20_HAS_MEMORY_SANITIZER
  __msan_unpoison(data, bytes);
#endif
#if UNI20_ENABLE_VALGRIND
  (void)VALGRIND_MAKE_MEM_DEFINED(data, bytes);
#endif
  (void)data;
}

} // namespace uni20::detail::memory_diagnostics
