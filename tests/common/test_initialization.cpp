#include <gtest/gtest.h>
#include <uni20/common/aligned_buffer.hpp>
#include <uni20/storage/host_storage.hpp>

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

namespace
{
using uni20::StorageInitialization;

TEST(Initialization, ZeroMeansNumericalZero)
{
  std::array<double, 4> real{1, 2, 3, 4};
  std::array<uni20::complex<double>, 2> complex{uni20::complex<double>{1, 2}, uni20::complex<double>{3, 4}};
  uni20::detail::initialize_host_elements(real.data(), real.size(), StorageInitialization::Zero);
  uni20::detail::initialize_host_elements(complex.data(), complex.size(), StorageInitialization::Zero);
  for (auto value : real)
    EXPECT_EQ(value, 0.0);
  for (auto value : complex)
    EXPECT_EQ(value, (uni20::complex<double>{0, 0}));
}

TEST(Initialization, UninitializedPreservesNontrivialObjects)
{
  std::array<std::string, 2> values{"first", "second"};
  uni20::detail::initialize_host_elements(values.data(), values.size(), StorageInitialization::Uninitialized);
  EXPECT_EQ(values[0], "first");
  EXPECT_EQ(values[1], "second");
}

TEST(Initialization, EmptyRangeAcceptsNull)
{
  uni20::detail::initialize_host_elements<double>(nullptr, 0, StorageInitialization::Zero);
  uni20::detail::initialize_host_elements<double>(nullptr, 0, StorageInitialization::Uninitialized);
}

TEST(Initialization, SupportsMoveOnlyAndNonassignableElements)
{
  std::array<std::unique_ptr<int>, 2> pointers{std::make_unique<int>(3), std::make_unique<int>(4)};
  uni20::detail::initialize_host_elements(pointers.data(), pointers.size(), StorageInitialization::Uninitialized);
  EXPECT_EQ(*pointers[0], 3);
  uni20::detail::initialize_host_elements(pointers.data(), pointers.size(), StorageInitialization::Zero);
  EXPECT_EQ(pointers[0], nullptr);
  EXPECT_EQ(pointers[1], nullptr);

  struct Nonassignable
  {
      Nonassignable() : value(17) {}
      ~Nonassignable() {}
      Nonassignable& operator=(Nonassignable const&) = delete;
      int const value;
  };
  std::array<Nonassignable, 2> values{};
  uni20::detail::initialize_host_elements(values.data(), values.size(), StorageInitialization::Uninitialized);
  uni20::detail::initialize_host_elements(values.data(), values.size(), StorageInitialization::Zero);
  EXPECT_EQ(values[0].value, 17);
}

TEST(Initialization, UninitializedSupportsTrivialElementsWithoutDefaultConstructor)
{
  struct Token
  {
      Token() = delete;
      explicit Token(int initial) : value(initial) {}
      int value;
  };
  static_assert(uni20::uninitialized_ok<Token>);
  static_assert(!uni20::detail::zero_initializable_v<Token>);
  auto raw = uni20::allocate_uninitialized_buffer<Token>(2);
  raw[0] = Token{3};
  raw[1] = Token{4};
  EXPECT_EQ(raw[0].value + raw[1].value, 7);
  uni20::HostBuffer<Token> buffer(2);
  buffer[0] = Token{5};
  buffer[1] = Token{6};
  buffer.resize(3);
  buffer[2] = Token{7};
  EXPECT_EQ(buffer[0].value + buffer[1].value + buffer[2].value, 18);
  EXPECT_THROW((uni20::HostBuffer<Token>(1, StorageInitialization::Zero)), std::invalid_argument);
}

TEST(Initialization, UnsupportedZeroRejectsNonassignableTrivialElements)
{
  struct ConstToken
  {
      ConstToken() = delete;
      explicit ConstToken(int initial) : value(initial) {}
      int const value;
  };
  static_assert(uni20::uninitialized_ok<ConstToken>);
  static_assert(!std::is_assignable_v<ConstToken&, ConstToken>);
  uni20::HostBuffer<ConstToken> buffer(1);
  std::construct_at(buffer.data(), 23);
  EXPECT_EQ(buffer[0].value, 23);
  EXPECT_THROW((uni20::HostBuffer<ConstToken>(1, StorageInitialization::Zero)), std::invalid_argument);
}

TEST(Initialization, DiagnosticFillPreservesSignalingBits)
{
#if UNI20_FILL_UNINITIALIZED_SNAN
  static_assert(uni20::detail::has_signaling_nan_v<double>);
  static_assert(uni20::detail::has_signaling_nan_v<uni20::complex<double>>);
  static_assert(!uni20::detail::has_signaling_nan_v<int>);
  auto real = uni20::allocate_uninitialized_buffer<double>(2);
  auto complex = uni20::allocate_uninitialized_buffer<uni20::complex<double>>(2);
  // Inspect the diagnostic representation deliberately, without floating-point arithmetic.
  uni20::detail::memory_diagnostics::mark_initialized(real.get(), 2 * sizeof(double));
  uni20::detail::memory_diagnostics::mark_initialized(complex.get(), 2 * sizeof(uni20::complex<double>));
  auto const expected = std::bit_cast<std::uint64_t>(uni20::numeric_limits<double>::signaling_NaN());
  for (int i = 0; i < 2; ++i)
    EXPECT_EQ(std::bit_cast<std::uint64_t>(real[i]), expected);
  std::array<std::uint64_t, 4> components{};
  static_assert(sizeof(components) == 2 * sizeof(uni20::complex<double>));
  std::memcpy(components.data(), complex.get(), sizeof(components));
  for (auto bits : components)
    EXPECT_EQ(bits, expected);
#else
  GTEST_SKIP() << "Signaling NaN filling is disabled for this configuration";
#endif
}

TEST(Initialization, OverwritingUninitializedStorageDefinesValues)
{
  auto values = uni20::allocate_uninitialized_buffer<double>(3);
  values[0] = 1;
  values[1] = 2;
  values[2] = 3;
  uni20::detail::memory_diagnostics::check_initialized(values.get(), 3 * sizeof(double));
  EXPECT_EQ(values[0] + values[1] + values[2], 6.0);
}

} // namespace
