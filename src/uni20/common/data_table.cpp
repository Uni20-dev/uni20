#include "data_table.hpp"

#include <cerrno>
#include <locale.h>
#include <system_error>

namespace uni20::presentation::data_table_detail
{
void validate_identifier(std::string_view identifier)
{
  auto start = [](char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; };
  if (identifier.empty() || !start(identifier.front()))
    throw std::invalid_argument("data table column identifier must start with an ASCII letter or underscore");
  for (char c : identifier)
    if (!start(c) && !(c >= '0' && c <= '9'))
      throw std::invalid_argument(
          "data table column identifier must contain only ASCII letters, digits or underscores");
}

void write_field(std::ostream& out, std::string_view field, char delimiter, bool quote_empty)
{
  bool const quote = (quote_empty && field.empty()) || field.find(delimiter) != std::string_view::npos ||
                     field.find_first_of("\"\r\n") != std::string_view::npos;
  if (quote) out.put('"');
  for (char c : field)
  {
    if (quote && c == '"') out.put('"');
    out.put(c);
  }
  if (quote) out.put('"');
}

void check_output(std::ostream& out)
{
  if (!out) throw std::ios_base::failure("data table output failed");
}

namespace
{
struct numeric_locale
{
    locale_t value = newlocale(LC_NUMERIC_MASK, "C", nullptr);
    numeric_locale()
    {
      if (!value) throw std::system_error(errno, std::generic_category(), "create data table numeric locale");
    }
    ~numeric_locale() { freelocale(value); }
};
} // namespace

classic_numeric_locale::classic_numeric_locale()
{
  static numeric_locale const locale;
  auto previous = uselocale(locale.value);
  if (!previous) throw std::system_error(errno, std::generic_category(), "select data table numeric locale");
  previous_ = previous;
}

classic_numeric_locale::~classic_numeric_locale() { uselocale(static_cast<locale_t>(previous_)); }
} // namespace uni20::presentation::data_table_detail
