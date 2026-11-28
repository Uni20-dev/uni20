#pragma once

#include <uni20/random/identity.hpp>

namespace uni20::random
{
/// \brief Standard uniform random bit generator over consecutive words of one leaf.
/// \details Copying or transferring the stream preserves its next draw, including a partially consumed block.
class PhiloxStream {
  public:
    using result_type = std::uint32_t;

    UNI20_HOST_DEVICE constexpr explicit PhiloxStream(Identity leaf) : leaf_(leaf) {}
    UNI20_HOST_DEVICE static constexpr result_type min() { return 0; }
    UNI20_HOST_DEVICE static constexpr result_type max() { return UINT32_MAX; }

    UNI20_HOST_DEVICE constexpr result_type operator()()
    {
      if (lane_ == 0) block_ = value_block(leaf_, position_);
      auto word = block_.words[lane_++];
      if (lane_ == 4)
      {
        ++position_;
        lane_ = 0;
      }
      return word;
    }

    /// \brief Skip a count of 32-bit words in constant time. Skipping exactly to the end is valid.
    UNI20_HOST_DEVICE constexpr void discard(std::uint64_t words)
    {
      auto const lanes = lane_ + static_cast<unsigned>(words % 4);
      auto const advance = words / 4 + lanes / 4;
      auto const next = position_ + advance;
      auto const lane = lanes % 4;
      if (next > detail::value_domain || (next == detail::value_domain && lane != 0)) detail::address_out_of_range();
      if (lane != 0 && (advance != 0 || lane_ == 0)) block_ = value_block(leaf_, next);
      position_ = next;
      lane_ = lane;
    }

  private:
    Identity leaf_;
    std::uint64_t position_ = 0;
    Philox4x32Block block_{};
    unsigned lane_ = 0;
};
} // namespace uni20::random
