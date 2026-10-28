#pragma once
#include "../exact_constant.hpp"
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace uni20::detail
{
// Akiyama--Tanigawa recurrence in exact rationals. The internal row has
// B1=+1/2; next() exposes the conventional B1=-1/2 without changing that row.
class exact_bernoulli_sequence {
  public:
    explicit exact_bernoulli_sequence(std::size_t maximum_order)
    {
      if (maximum_order >= row_.max_size()) throw std::length_error("Bernoulli order exceeds storage capacity");
      row_.reserve(maximum_order + 1);
    }
    exact_constant next()
    {
      auto const m = row_.size();
      row_.push_back(exact_constant{1} / exact_constant{m + 1});
      for (std::size_t j = m; j > 0; --j)
        row_[j - 1] = exact_constant{j} * (row_[j - 1] - row_[j]);
      return m == 1 ? -row_[0] : row_[0];
    }

  private:
    std::vector<exact_constant> row_;
};

inline exact_constant exact_bernoulli(std::size_t n)
{
  exact_bernoulli_sequence sequence(n);
  exact_constant value;
  for (std::size_t i = 0; i <= n; ++i)
    value = sequence.next();
  return value;
}
} // namespace uni20::detail
