#include "precision_registry.hpp"

#include <map>

namespace uni20::test
{
TEST(NumericalCoverage, RegisteredProbesMatchDeclaredMatrix)
{
  std::map<std::string, int> registered;
  auto const* unit = testing::UnitTest::GetInstance();
  for (int i = 0; i < unit->total_test_suite_count(); ++i)
  {
    auto const* suite = unit->GetTestSuite(i);
    std::string_view name = suite->name();
    if (!name.starts_with("NumericalScalar/") && !name.starts_with("NumericalLinalg/") &&
        !name.starts_with("NumericalKrylov/"))
      continue;
    for (int j = 0; j < suite->total_test_count(); ++j)
      ++registered[std::string(name) + "." + suite->GetTestInfo(j)->name()];
  }

  // Store the complete declared matrix as a TSV property in Google Test XML.
  // The reporter joins it to actual results; exclusions never become passes.
  std::string manifest;
  for_each_precision_case(PrecisionCases{}, [&]<class C>() {
    for (std::size_t i = 0; i < precision_probes.size(); ++i)
    {
      auto const& probe = precision_probes[i];
      auto const coverage = probe_coverage<C>(static_cast<PrecisionProbe>(i));
      auto const operation = std::string(probe.suite) + "." + probe.name;
      auto const name = std::string(probe.suite) + "/" + C::name() + "." + probe.name;
      EXPECT_EQ(registered[name], coverage.state == "ready" ? 1 : 0) << name << ": " << coverage.reason;
      registered.erase(name);
      manifest += operation + "\t" + C::name() + "\t" + probe.backend + "\t" + std::string(coverage.state) + "\t" +
                  std::string(coverage.reason) + "\n";
    }
  });
  EXPECT_TRUE(registered.empty()) << "numerical probes must declare their coverage in precision_registry.hpp";
  RecordProperty("precision_matrix_v1", manifest);
}
} // namespace uni20::test
