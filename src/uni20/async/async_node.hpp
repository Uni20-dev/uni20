/// \file async_node.hpp
/// \brief Debug metadata nodes used for async DAG visualization.

#pragma once

#include "shared_storage.hpp"
#include <uni20/common/demangle.hpp>
#include <uni20/config.hpp>
#include <uni20/core/scalar_io.hpp>

#include <atomic>
#include <cstddef>
#include <exception>
#include <mutex>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace uni20::async
{

/// \brief Construction state reported for a debug DAG value node.
enum class NodeValueState
{
  Invalid,       ///< No storage or fallback value is available.
  Unconstructed, ///< Storage exists but does not currently contain a value.
  Constructed,   ///< A value is currently constructed.
  Expired,       ///< The storage has been released; only the node's identity remains.
};

/// \brief Returns a stable label for a construction state, suitable for Graphviz output.
/// \param state Construction state.
/// \return The label.
[[nodiscard]] constexpr std::string_view node_value_state_label(NodeValueState state) noexcept
{
  switch (state)
  {
    case NodeValueState::Invalid:
      return "invalid";
    case NodeValueState::Unconstructed:
      return "unconstructed";
    case NodeValueState::Constructed:
      return "constructed";
    case NodeValueState::Expired:
      return "expired";
  }
  return "invalid";
}

namespace detail
{

/// \brief Poison-pill declaration used to detect user ADL debug-value hooks.
void uni20_async_debug_value();

/// \brief Detects a one-line debug-value overload found by argument-dependent lookup.
/// \tparam T Value type to inspect.
template <typename T>
concept HasAdlAsyncDebugValue = requires(T const& value) { std::string{uni20_async_debug_value(value)}; };

/// \brief Format extents as a compact shape summary.
/// \tparam Extents Mdspan-style extents type.
/// \param extents Shape descriptor to format.
/// \return `shape=(...)` with one entry per axis.
template <typename Extents> std::string format_shape_debug_info(Extents const& extents)
{
  using extents_type = std::remove_cvref_t<Extents>;
  std::string out = "shape=(";
  for (std::size_t axis = 0; axis < static_cast<std::size_t>(extents_type::rank()); ++axis)
  {
    if (axis != 0) out += ", ";
    out += std::to_string(static_cast<unsigned long long>(extents.extent(axis)));
  }
  out += ")";
  return out;
}

/// \brief Detects objects that expose `mdspan().extents()`.
/// \tparam T Value type to inspect.
template <typename T>
concept HasMdspanExtents = requires(T const& value, std::size_t axis) {
  std::remove_cvref_t<decltype(value.mdspan().extents())>::rank();
  value.mdspan().extents().extent(axis);
};

/// \brief Detects objects that expose `extents()`.
/// \tparam T Value type to inspect.
template <typename T>
concept HasExtents = requires(T const& value, std::size_t axis) {
  std::remove_cvref_t<decltype(value.extents())>::rank();
  value.extents().extent(axis);
};

/// \brief Normalize debug value text to a single line for Graphviz node labels.
/// \param value Debug-value string to sanitize.
/// \return Sanitized debug-value string.
inline std::string sanitize_debug_value(std::string value)
{
  for (char& ch : value)
  {
    if (ch == '\n' || ch == '\r' || ch == '\t') ch = ' ';
  }
  return value;
}

/// \brief Format scalar values for async DAG labels.
/// \tparam T Scalar value type.
/// \param value Value to format.
/// \return One-line scalar representation, or `constructed` for non-scalars.
template <typename T> std::string scalar_debug_info(T const& value)
{
  using value_type = std::remove_cvref_t<T>;
  if constexpr (std::is_same_v<value_type, bool>)
  {
    return value ? "true" : "false";
  }
  else if constexpr (std::is_same_v<value_type, char>)
  {
    return std::string{"'"} + value + "'";
  }
  else if constexpr (std::is_integral_v<value_type>)
  {
    if constexpr (std::is_signed_v<value_type>)
      return std::to_string(static_cast<long long>(value));
    else
      return std::to_string(static_cast<unsigned long long>(value));
  }
  else if constexpr (uni20::RealOrComplex<value_type>)
  {
    return uni20::format_scalar(value);
  }
  else if constexpr (std::is_enum_v<value_type>)
  {
    using underlying_type = std::underlying_type_t<value_type>;
    return scalar_debug_info(static_cast<underlying_type>(value));
  }
  else
  {
    return {};
  }
}

/// \brief Default one-line debug value for values without a custom ADL hook.
/// \tparam T Value type to inspect.
/// \param value Value to summarize.
/// \return Scalar value, shape summary, or empty string when no useful value is available.
template <typename T> std::string default_debug_value(T const& value)
{
  if constexpr (HasMdspanExtents<T>)
  {
    auto span = value.mdspan();
    return format_shape_debug_info(span.extents());
  }
  else if constexpr (HasExtents<T>)
  {
    return format_shape_debug_info(value.extents());
  }
  else
  {
    return scalar_debug_info(value);
  }
}

/// \brief Resolve custom or default one-line debug value for a constructed value.
/// \tparam T Value type to inspect.
/// \param value Constructed value to summarize.
/// \return One-line debug value, or empty string when no useful value is available.
template <typename T> std::string debug_value_for_value(T const& value)
{
  if constexpr (HasAdlAsyncDebugValue<T>)
  {
    return sanitize_debug_value(std::string{uni20_async_debug_value(value)});
  }
  else
  {
    return sanitize_debug_value(default_debug_value(value));
  }
}

} // namespace detail

/// \brief Records node information for DAG/debugging visualization.
///
/// Each NodeInfo represents a single instance of an object participating
/// in the dependency DAG (e.g., an Async<T>). It is assigned a unique global
/// index at construction, and retains the identity of the async storage.
///
/// NodeInfo objects are always heap-allocated and intentionally leaked for
/// process-lifetime diagnostics. Each node is globally unique for the process lifetime.
/// The node does not keep its storage alive: diagnostic history may outlive the value,
/// and \ref observe reports such a value as expired.
class NodeInfo {
  public:
    class Observation;

    /// \brief Deleted default constructor. NodeInfo can only be constructed via \ref create.
    NodeInfo() = delete;

    /// \brief Returns the address of the shared storage control block.
    /// \details An identity only: the block may already have been released.
    /// \return Control-block address, or `nullptr` when the node was not storage-backed.
    [[nodiscard]] void const* storage_address() const { return storage_; }

    /// \brief Observe the node's value, keeping its storage alive while the observation exists.
    /// \details One observation covers the construction state, the value address and the value
    ///          text. For storage that is still owned, it holds a strong reference, taken through
    ///          the node's anchor so that it is never taken after the last owner has gone. If the
    ///          program drops its last owner meanwhile, the value is destroyed when the observation
    ///          ends, on the observing thread. Storage already released is reported as
    ///          `Expired`. A node created from a raw pointer has no lifetime guarantee, so its
    ///          observation carries only the pointer's identity and never formats the value.
    /// \return The observation.
    [[nodiscard]] Observation observe() const noexcept;

    /// \brief Returns a human-readable, demangled type name.
    ///
    /// The string is obtained from an interned map keyed by the mangled type name.
    /// The returned string_view is valid for the process lifetime.
    ///
    /// \note Thread-safe.
    [[nodiscard]] std::string_view type() const
    {
      std::lock_guard<std::mutex> lock(map_mutex_);
      auto [it, inserted] = map_.try_emplace(type_key_, uni20::demangle::demangle(type_key_));
      return std::string_view(it->second);
    }

    /// \brief Returns the unique global index of this node.
    ///
    /// \return Monotonically increasing value, unique across all NodeInfo instances for this process.
    [[nodiscard]] uint64_t global_index() const { return global_index_; }

    /// \brief Returns the stacktrace of where the node was created
#if 0 && UNI20_HAS_STACKTRACE
    std::stacktrace const& stack() const { return stack_; }
#endif

    /// \brief Factory method to create a NodeInfo for an object pointer.
    ///
    /// \tparam T  The type of the object.
    /// \param value  Pointer to the value for which a NodeInfo should be created.
    /// \return Pointer to the new NodeInfo (never deallocated).
    ///
    /// \post The returned NodeInfo outlives all references (process lifetime).
    // Keep the diagnostic-only allocation frame visible to sanitizer symbolizers.
    // GCC's libbacktrace can omit inlined frames even with full debug information.
    template <typename T> [[gnu::noinline]] static NodeInfo const* create(T const* value)
    {
      return new NodeInfo(value);
    }

    /// \brief Factory method to create a NodeInfo for shared async storage.
    ///
    /// \pre \p storage has not yet been shared with another thread, and has no node yet.
    /// \tparam T The async storage value type.
    /// \param storage Storage handle represented by this node.
    /// \return Pointer to the new NodeInfo (never deallocated).
    ///
    /// \post The returned NodeInfo outlives all references (process lifetime).
    template <typename T> [[gnu::noinline]] static NodeInfo const* create(shared_storage<T> const& storage)
    {
      return new NodeInfo(storage);
    }

  private:
    using Block = detail::shared_storage_control_block;
    using ReservedAddressObserver = void const* (*)(Block const*) noexcept;
    using ValueTextObserver = std::string (*)(Block const*);

    template <typename T> using TypedBlock = typename shared_storage<T>::control_block;

    // The address reserved for the value in the block. It is never dereferenced, so it is valid to
    // compute while another thread constructs or destroys the value in place.
    template <typename T> static void const* reserved_value_address(Block const* block) noexcept
    {
      return static_cast<TypedBlock<T> const*>(block)->storage;
    }

    template <typename T> static std::string text_for_storage_value(Block const* block)
    {
      auto* typed = const_cast<TypedBlock<T>*>(static_cast<TypedBlock<T> const*>(block));
      T const& value = *typed->ptr();
      return detail::debug_value_for_value(value);
    }

    template <typename T>
    NodeInfo(T const* value)
        : raw_value_address_(static_cast<void const*>(value)), type_key_(typeid(T).name()),
          global_index_(next_global_++)
#if 0 && UNI20_HAS_STACKTRACE
          ,
          stack_(std::stacktrace::current())
#endif
    {}

    template <typename T>
    explicit NodeInfo(shared_storage<T> const& storage)
        : storage_(storage.ctrl_), type_key_(typeid(T).name()),
          reserved_address_observer_(&NodeInfo::reserved_value_address<T>),
          value_text_observer_(&NodeInfo::text_for_storage_value<T>), global_index_(next_global_++)
#if 0 && UNI20_HAS_STACKTRACE
          ,
          stack_(std::stacktrace::current())
#endif
    {
#if UNI20_DEBUG_DAG
      if (storage_) storage_->attach_diagnostic_anchor(&anchor_);
#endif
    }

    // global data for the map of mangled name to demangled name (protected by map_mutex_), and the global index
    inline static std::unordered_map<const char*, std::string> map_;
    inline static std::mutex map_mutex_;
    inline static std::atomic<uint64_t> next_global_ = 0;

    // Per-instance data:
    Block* storage_{nullptr};                             ///< Storage control block; may have been released.
    void const* raw_value_address_{nullptr};              ///< Identity of the value of a raw-pointer node.
    char const* type_key_;                                ///< Mangled type name (from typeid).
    ReservedAddressObserver reserved_address_observer_{}; ///< Value address reserved in the block.
    ValueTextObserver value_text_observer_{};             ///< One-line text of a live value.
    uint64_t global_index_;                               ///< Globally unique node index.
    mutable detail::StorageObservationAnchor anchor_;     ///< Retired by the final release of storage_.
#if 0 && UNI20_HAS_STACKTRACE
    std::stacktrace stack_; ///< stacktrace of where the node was constructed
#endif
};

/// \brief One observation of a debug DAG node's value; see \ref NodeInfo::observe.
/// \details Holds a strong reference to the value's storage, if it was still owned, until the
///          observation is destroyed. The state and address are read once, when the observation
///          is made.
class NodeInfo::Observation {
  public:
    Observation(Observation&& other) noexcept
        : node_(other.node_), pinned_(std::exchange(other.pinned_, nullptr)), state_(other.state_),
          address_(other.address_)
    {}
    Observation(Observation const&) = delete;
    Observation& operator=(Observation const&) = delete;
    Observation& operator=(Observation&&) = delete;

    ~Observation()
    {
      if (pinned_) pinned_->release_ref();
    }

    /// \brief Construction state when the observation was made.
    [[nodiscard]] NodeValueState state() const noexcept { return state_; }

    /// \brief The value's address when the observation was made.
    /// \details For storage, the address reserved for a constructed value in the block; for a raw-pointer
    ///          node, the pointer it was created from. Only an identity: it is not dereferenced here.
    /// \return The address, or `nullptr` when no value was constructed or the storage had expired.
    [[nodiscard]] void const* address() const noexcept { return address_; }

    /// \brief Returns type-specific display text for the value.
    /// \details Calls the value's formatter, such as a custom `uni20_async_debug_value` hook. If
    ///          formatting throws, the text says so instead. Diagnostics are best effort: a dump
    ///          written by the diagnostics service or at a scheduler progress point must not end
    ///          the program, and the rest of the dump is still useful.
    /// \pre No writer modifies or destroys the value during the call.
    /// \return One-line text, such as a scalar value or `shape=(...)`. Empty when the value was not
    ///         constructed, its storage had expired, the node was created from a raw pointer, or
    ///         the type has no useful display.
    [[nodiscard]] std::string value_text() const
    {
      if (!pinned_ || state_ != NodeValueState::Constructed) return {};
      try
      {
        return node_->value_text_observer_(pinned_);
      }
      catch (std::exception const& e)
      {
        return detail::sanitize_debug_value(std::string("(debug value threw: ") + e.what() + ")");
      }
      catch (...)
      {
        return "(debug value threw)";
      }
    }

  private:
    friend class NodeInfo;

    explicit Observation(NodeInfo const* node) noexcept : node_(node) {}

    NodeInfo const* node_;
    Block* pinned_{nullptr};
    NodeValueState state_{NodeValueState::Invalid};
    void const* address_{nullptr};
};

inline NodeInfo::Observation NodeInfo::observe() const noexcept
{
  Observation observation(this);
  if (!storage_)
  {
    observation.state_ = raw_value_address_ ? NodeValueState::Constructed : NodeValueState::Invalid;
    observation.address_ = raw_value_address_;
    return observation;
  }
#if UNI20_DEBUG_DAG
  if (!detail::try_pin_observed_block(anchor_, storage_))
  {
    observation.state_ = NodeValueState::Expired;
    return observation;
  }
  observation.pinned_ = storage_;
  if (storage_->constructed())
  {
    observation.state_ = NodeValueState::Constructed;
    observation.address_ = reserved_address_observer_(storage_);
  }
  else
  {
    observation.state_ = NodeValueState::Unconstructed;
  }
#else
  // Without the DAG build, nothing retires an anchor, so the storage cannot be pinned safely.
  observation.state_ = NodeValueState::Invalid;
#endif
  return observation;
}

} // namespace uni20::async
