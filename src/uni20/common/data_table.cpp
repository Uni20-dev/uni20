#include "data_table.hpp"

#include <cerrno>
#include <locale.h>
#include <system_error>

namespace uni20::presentation::data_table_detail
{
void validate_numeric_precision(real_format_notation notation, int digits)
{
  if (digits < -1) throw std::invalid_argument("data table precision must be -1 or nonnegative");
  if (notation == real_format_notation::general && digits == 0)
    throw std::invalid_argument("general display precision must be positive or -1");
}

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

namespace uni20::presentation::data_table_detail
{
void write_json_string(std::ostream& out, std::string_view value)
{
  // Reject invalid UTF-8, overlong sequences, surrogate code points and values beyond U+10FFFF.
  for (std::size_t i = 0; i < value.size();)
  {
    auto c = static_cast<unsigned char>(value[i++]);
    if (c < 0x80) continue;
    unsigned count = c >= 0xC2 && c <= 0xDF ? 1 : c >= 0xE0 && c <= 0xEF ? 2 : c >= 0xF0 && c <= 0xF4 ? 3 : 0;
    if (!count || count > value.size() - i) throw std::invalid_argument("invalid UTF-8 in data table JSON");
    unsigned code = c & ((1U << (6 - count)) - 1);
    for (unsigned j = 0; j < count; ++j)
    {
      auto next = static_cast<unsigned char>(value[i++]);
      if ((next & 0xC0) != 0x80) throw std::invalid_argument("invalid UTF-8 in data table JSON");
      code = (code << 6) | (next & 0x3F);
    }
    if ((count == 1 && code < 0x80) || (count == 2 && code < 0x800) || (count == 3 && code < 0x10000) ||
        (code >= 0xD800 && code <= 0xDFFF) || code > 0x10FFFF)
      throw std::invalid_argument("invalid UTF-8 in data table JSON");
  }
  out.put('"');
  constexpr char hex[] = "0123456789abcdef";
  for (unsigned char c : value)
  {
    if (c == '"' || c == '\\')
    {
      out.put('\\');
      out.put(static_cast<char>(c));
    }
    else if (c < 0x20)
    {
      out.write("\\u00", 4);
      out.put(hex[c >> 4]);
      out.put(hex[c & 15]);
    }
    else
      out.put(static_cast<char>(c));
  }
  out.put('"');
}
} // namespace uni20::presentation::data_table_detail
