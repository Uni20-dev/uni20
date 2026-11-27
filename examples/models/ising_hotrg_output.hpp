/**
 * \file ising_hotrg_output.hpp
 * \brief Typed presentation of Ising HOTRG flow diagnostics.
 */
#pragma once

#include "ising_strip_reference.hpp"
#include <uni20/common/data_table_sinks.hpp>

#include <optional>

namespace uni20::examples::dense_hotrg
{
/// \brief Create a flow table without retaining a second copy of the numerical history.
template <Real R> auto make_flow_table(Precision precision)
{
  namespace p = presentation;
  // Above the fixture's precision, exact decimal reference values are displayed
  // with at most 100 significant digits, never extra binary-rounding digits.
  // The cut, the error against Onsager and the asymmetry are error measures, shown
  // to three significant digits; ln Z/N and the scaling dimensions keep the
  // working precision, which the columns declare so that a streaming terminal
  // lays them out for it from the first row.
  int const reference_digits = precision.bit_count() > 332 ? 100 : -1;
  return p::make_data_table(
      "HOTRG flow", {.retain = p::retention::none}, p::data_column<int>("log2_width").label("log2(width)"),
      p::data_column<index_type>("bond").label("Bond"), p::data_column<R>("discarded").label("Cut").scientific(2),
      p::data_column<R>("lnz_per_site").label("ln Z/N").working_precision(precision),
      p::data_column<R>("onsager_relative_error").label("vs Onsager").scientific(2),
      p::data_column<std::optional<R>>("x1").label("X_1").working_precision(precision),
      p::data_column<std::optional<R>>("exact_x1")
          .label("Exact X_1")
          .general(reference_digits)
          .working_precision(precision),
      p::data_column<std::optional<R>>("x2").label("X_2").working_precision(precision),
      p::data_column<std::optional<R>>("exact_x2")
          .label("Exact X_2")
          .general(reference_digits)
          .working_precision(precision),
      p::data_column<R>("asymmetry").label("Asymmetry").scientific(2));
}

/// \brief Read a displayed reference without suggesting precision beyond its decimal fixture.
template <Real R> R displayed_reference(std::string_view text, Precision precision)
{
#if UNI20_HAS_MPREAL
  if constexpr (std::same_as<R, mpreal>)
    if (precision.bit_count() > 332) return reference::value<R>(text, Precision::exact());
#endif
  return reference::value<R>(text, precision);
}
} // namespace uni20::examples::dense_hotrg
