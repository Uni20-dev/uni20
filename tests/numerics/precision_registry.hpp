#pragma once

#include "precision_cases.hpp"

#include <array>
#include <string_view>

namespace uni20::test
{
enum class PrecisionProbe
{
  EpsilonAndRetainedIncrement,
  ReciprocalAccuracyImprovesWithPrecision,
  MathDispatchRetainsWorkingPrecision,
  ElementaryRetainsWorkingPrecision,
  ElementarySmallArguments,
  ElementaryAccuracyImprovesWithPrecision,
  UtilitiesRetainWorkingPrecision,
  FusedArithmeticRoundsOnce,
  UtilitiesBoundarySemantics,
  UtilityAccuracyImprovesWithPrecision,
  StandardSpecialAccuracy,
  LogGammaAccuracy,
  BesselAccuracy,
  MpfrSpecialIdentities,
  MpfrSpecialAccuracy,
  MpfrExtensionsStableArithmetic,
  CpuMatrixOneNormRetainsIncrement,
  CpuReductionsRetainIncrement,
  CpuGemmRetainsIncrement,
  ProviderGemmRetainsIncrement,
  CpuSolveResolvesSmallGap,
  ProviderSolveResolvesSmallGap,
  SolvePreservesExtendedExponentRange,
  SolveAccuracyImprovesWithPrecision,
  CpuLuResolvesSmallGap,
  ProviderLuResolvesSmallGap,
  CpuLuAccuracyImprovesWithPrecision,
  ProviderLuAccuracyImprovesWithPrecision,
  CpuLogDeterminantRetainsPrecision,
  ProviderLogDeterminantRetainsPrecision,
  CpuLogDeterminantPreservesExtendedExponentRange,
  ProviderLogDeterminantPreservesExtendedExponentRange,
  ProjectedTridiagonalResolvesGap,
  LanczosResolvesGapAndResidual,
  ArnoldiResolvesGap,
  ExponentialActionRetainsIncrement,
  count
};

struct ProbeDescription
{
    char const* suite;
    char const* name;
    char const* backend;
};

inline constexpr std::array precision_probes{
    ProbeDescription{"NumericalScalar", "EpsilonAndRetainedIncrement", "scalar"},
    ProbeDescription{"NumericalScalar", "ReciprocalAccuracyImprovesWithPrecision", "scalar"},
    ProbeDescription{"NumericalScalar", "MathDispatchRetainsWorkingPrecision", "scalar_math"},
    ProbeDescription{"NumericalScalar", "ElementaryRetainsWorkingPrecision", "scalar_math"},
    ProbeDescription{"NumericalScalar", "ElementarySmallArguments", "scalar_math"},
    ProbeDescription{"NumericalScalar", "ElementaryAccuracyImprovesWithPrecision", "scalar_math"},
    ProbeDescription{"NumericalScalar", "UtilitiesRetainWorkingPrecision", "scalar_math"},
    ProbeDescription{"NumericalScalar", "FusedArithmeticRoundsOnce", "scalar_math"},
    ProbeDescription{"NumericalScalar", "UtilitiesBoundarySemantics", "scalar_math"},
    ProbeDescription{"NumericalScalar", "UtilityAccuracyImprovesWithPrecision", "scalar_math"},
    ProbeDescription{"NumericalScalar", "StandardSpecialAccuracy", "scalar_math"},
    ProbeDescription{"NumericalScalar", "LogGammaAccuracy", "scalar_math"},
    ProbeDescription{"NumericalScalar", "BesselAccuracy", "scalar_math"},
    ProbeDescription{"NumericalScalar", "MpfrSpecialIdentities", "scalar_math"},
    ProbeDescription{"NumericalScalar", "MpfrSpecialAccuracy", "scalar_math"},
    ProbeDescription{"NumericalScalar", "MpfrExtensionsStableArithmetic", "scalar_math"},
    ProbeDescription{"NumericalLinalg", "CpuMatrixOneNormRetainsIncrement", "cpu_reference"},
    ProbeDescription{"NumericalLinalg", "CpuReductionsRetainIncrement", "cpu_reference"},
    ProbeDescription{"NumericalLinalg", "CpuGemmRetainsIncrement", "cpu_reference"},
    ProbeDescription{"NumericalLinalg", "ProviderGemmRetainsIncrement", "blas_or_mplapack"},
    ProbeDescription{"NumericalLinalg", "CpuSolveResolvesSmallGap", "cpu_reference"},
    ProbeDescription{"NumericalLinalg", "ProviderSolveResolvesSmallGap", "lapack_or_mplapack"},
    ProbeDescription{"NumericalLinalg", "SolvePreservesExtendedExponentRange", "cpu_or_mplapack_mpfr"},
    ProbeDescription{"NumericalLinalg", "SolveAccuracyImprovesWithPrecision", "cpu_or_mplapack_mpfr"},
    ProbeDescription{"NumericalLinalg", "CpuLuResolvesSmallGap", "cpu_reference"},
    ProbeDescription{"NumericalLinalg", "ProviderLuResolvesSmallGap", "lapack_or_mplapack"},
    ProbeDescription{"NumericalLinalg", "CpuLuAccuracyImprovesWithPrecision", "cpu_reference"},
    ProbeDescription{"NumericalLinalg", "ProviderLuAccuracyImprovesWithPrecision", "lapack_or_mplapack"},
    ProbeDescription{"NumericalLinalg", "CpuLogDeterminantRetainsPrecision", "cpu_reference"},
    ProbeDescription{"NumericalLinalg", "ProviderLogDeterminantRetainsPrecision", "lapack_or_mplapack"},
    ProbeDescription{"NumericalLinalg", "CpuLogDeterminantPreservesExtendedExponentRange", "cpu_reference"},
    ProbeDescription{"NumericalLinalg", "ProviderLogDeterminantPreservesExtendedExponentRange", "lapack_or_mplapack"},
    ProbeDescription{"NumericalKrylov", "ProjectedTridiagonalResolvesGap", "projected_lapack"},
    ProbeDescription{"NumericalKrylov", "LanczosResolvesGapAndResidual", "native_krylov_projected_lapack"},
    ProbeDescription{"NumericalKrylov", "ArnoldiResolvesGap", "native_krylov_projected_lapack"},
    ProbeDescription{"NumericalKrylov", "ExponentialActionRetainsIncrement", "native_krylov_projected_lapack"}};
static_assert(precision_probes.size() == static_cast<std::size_t>(PrecisionProbe::count));

struct ProbeCoverage
{
    std::string_view state = "ready";
    std::string_view reason = {};
};

// These are declared expectations, not queries of production kernel concepts.
// Losing a supported implementation must break its probe, not remove the test.
template <class C> constexpr ProbeCoverage probe_coverage(PrecisionProbe probe)
{
  if constexpr (!C::available)
    return {"unavailable", "scalar dependency or native format not configured"};
  else
  {
    using enum PrecisionProbe;
    bool const cpu_lu = probe == CpuLuResolvesSmallGap || probe == CpuLuAccuracyImprovesWithPrecision ||
                        probe == CpuLogDeterminantRetainsPrecision ||
                        probe == CpuLogDeterminantPreservesExtendedExponentRange;
    [[maybe_unused]] bool const provider =
        probe == ProviderGemmRetainsIncrement || probe == ProviderSolveResolvesSmallGap ||
        probe == ProviderLuResolvesSmallGap || probe == ProviderLuAccuracyImprovesWithPrecision ||
        probe == ProviderLogDeterminantRetainsPrecision ||
        probe == ProviderLogDeterminantPreservesExtendedExponentRange;
    if constexpr (C::runtime)
      if (cpu_lu) return {"unsupported", "CPU LU declines runtime-precision scalars"};
#if !UNI20_ENABLE_MPLAPACK_BINARY80
    if constexpr (C::binary80_provider)
      if (provider) return {"unavailable", "configure UNI20_ENABLE_MPLAPACK_BINARY80"};
#endif
#if UNI20_ENABLE_MPLAPACK_BINARY80 && !UNI20_HAS_MPLAPACK_BINARY80_COMPLEX_LU
    if constexpr (C::binary80_provider && C::is_complex)
      if (provider && probe != ProviderGemmRetainsIncrement)
        return {"unsupported", "provider complex division is unsafe with distinct _Float64x"};
#endif
    if constexpr (C::is_complex)
      if (probe == ElementaryRetainsWorkingPrecision || probe == ElementarySmallArguments ||
          probe == ElementaryAccuracyImprovesWithPrecision || probe == UtilitiesRetainWorkingPrecision ||
          probe == FusedArithmeticRoundsOnce || probe == UtilitiesBoundarySemantics ||
          probe == UtilityAccuracyImprovesWithPrecision)
        return {"not_applicable", "elementary and utility expansion covers real scalars"};
    if (probe == LogGammaAccuracy || probe == BesselAccuracy)
    {
      if constexpr (C::is_complex) return {"not_applicable", "native special functions cover real scalars"};
      if constexpr (!C::runtime)
      {
        using R = typename C::real_type;
        bool gamma = false, bessel = false;
        if constexpr (std::same_as<R, float>)
        {
          gamma = UNI20_HAS_LGAMMA_R_FLOAT;
          bessel = UNI20_HAS_BESSEL_FLOAT;
        }
        else if constexpr (std::same_as<R, double>)
        {
          gamma = UNI20_HAS_LGAMMA_R_DOUBLE;
          bessel = UNI20_HAS_BESSEL_DOUBLE;
        }
        else if constexpr (std::same_as<R, long double>)
        {
          gamma = UNI20_HAS_LGAMMA_R_LONG_DOUBLE;
          bessel = UNI20_HAS_BESSEL_LONG_DOUBLE;
        }
#if UNI20_HAS_FLOAT128
        else if constexpr (std::same_as<R, float128>)
        {
          gamma = UNI20_HAS_LGAMMA_R_FLOAT128;
          bessel = UNI20_HAS_BESSEL_FLOAT128;
        }
#endif
        if (!(probe == LogGammaAccuracy ? gamma : bessel))
          return {"unavailable", "native special-function provider not detected for this scalar"};
      }
    }
    if (probe == StandardSpecialAccuracy || probe == MpfrSpecialIdentities ||
        probe == MpfrSpecialAccuracy || probe == MpfrExtensionsStableArithmetic)
    {
      if constexpr (C::is_complex) return {"not_applicable", "special-function expansion covers real scalars"};
      if constexpr (!C::runtime)
        if (probe != StandardSpecialAccuracy)
          return {"unsupported", "native implementation deferred; no MPFR adapter for native types"};
    }
    switch (probe)
    {
      case StandardSpecialAccuracy:
      case LogGammaAccuracy:
      case BesselAccuracy:
      case UtilityAccuracyImprovesWithPrecision:
      case ElementaryAccuracyImprovesWithPrecision:
      case ReciprocalAccuracyImprovesWithPrecision:
      case SolveAccuracyImprovesWithPrecision:
      case CpuLuAccuracyImprovesWithPrecision:
      case ProviderLuAccuracyImprovesWithPrecision:
#if !UNI20_ENABLE_MPFR
        return {"unavailable", "MPFR required for independent 512-bit error measurement"};
#endif
        break;
      case CpuGemmRetainsIncrement:
        if constexpr (C::runtime) return {"unsupported", "CPU GEMM handles exact mode only for runtime scalars"};
        break;
      case CpuSolveResolvesSmallGap:
        if constexpr (C::runtime) return {"unsupported", "CPU solve declines runtime-precision scalars"};
        break;
      case SolvePreservesExtendedExponentRange:
      case CpuLogDeterminantPreservesExtendedExponentRange:
      case ProviderLogDeterminantPreservesExtendedExponentRange:
        if constexpr (!C::runtime)
          if constexpr (numeric_limits<typename C::real_type>::max_exponent <= 1024)
            return {"not_applicable", "this probe requires a wider exponent range than double"};
        break;
      case ProjectedTridiagonalResolvesGap:
        if constexpr (C::is_complex) return {"not_applicable", "Hermitian tridiagonal projection is real"};
        [[fallthrough]];
      case LanczosResolvesGapAndResidual:
      case ArnoldiResolvesGap:
      case ExponentialActionRetainsIncrement:
        if constexpr (!C::projected_lapack)
          return {"unsupported",
                  "projected LAPACK wrappers and dependent Krylov solvers are not wired for this scalar"};
        break;
      default:
        break;
    }
#if !UNI20_ENABLE_MPLAPACK_MPFR
    if constexpr (C::runtime)
      if (provider || probe == SolvePreservesExtendedExponentRange || probe == SolveAccuracyImprovesWithPrecision)
        return {"unavailable", "configure UNI20_ENABLE_MPLAPACK_MPFR"};
#endif
    return {};
  }
}

template <class... Cases, class Action> void for_each_precision_case(testing::Types<Cases...>, Action action)
{
  (action.template operator()<Cases>(), ...);
}

template <PrecisionProbe Probe, template <class> class Test> bool register_precision_probe(char const* file, int line)
{
  for_each_precision_case(PrecisionCases{}, [&]<class C>() {
    if constexpr (probe_coverage<C>(Probe).state == "ready")
    {
      constexpr auto description = precision_probes[static_cast<std::size_t>(Probe)];
      auto suite = std::string(description.suite) + "/" + C::name();
      // All probes in a scalar's suite share this fixture type. Google Test
      // owns the returned instance, just as it does for TEST/TYPED_TEST.
      testing::RegisterTest(suite.c_str(), description.name, nullptr, nullptr, file, line,
                            []() -> PrecisionTest<C>* { return new Test<C>; });
    }
  });
  return true;
}

// TYPED_TEST applies one type list to every operation in a suite. RegisterTest
// lets us retain the suite names while selecting applicable cases per operation.
#define UNI20_PRECISION_TEST(SUITE, NAME)                                                                              \
  static_assert(std::string_view(precision_probes[static_cast<std::size_t>(PrecisionProbe::NAME)].suite) == #SUITE &&  \
                std::string_view(precision_probes[static_cast<std::size_t>(PrecisionProbe::NAME)].name) == #NAME);     \
  template <class TypeParam> class SUITE##_##NAME : public PrecisionTest<TypeParam> {                                  \
      void TestBody() override;                                                                                        \
  };                                                                                                                   \
  [[maybe_unused]] static bool const SUITE##_##NAME##_registered =                                                     \
      register_precision_probe<PrecisionProbe::NAME, SUITE##_##NAME>(__FILE__, __LINE__);                              \
  template <class TypeParam> void SUITE##_##NAME<TypeParam>::TestBody()
} // namespace uni20::test
