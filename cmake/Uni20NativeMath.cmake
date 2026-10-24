# Probe each scalar separately: a double overload must not supply extended
# precision by conversion, and log-Gamma must not use global signgam state.
include(CheckCXXSourceCompiles)

function(uni20_check_native_special_math suffix type lgamma jn yn)
  block(SCOPE_FOR VARIABLES)
    set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
    check_cxx_source_compiles("
      #include <cmath>
      #include <type_traits>
      int main() {
        using R = ${type};
        static_assert(std::is_same_v<decltype(${lgamma}(R(1), nullptr)), R>);
        R (*volatile function)(R, int*) = &${lgamma};
        int sign;
        return function(R(1), &sign) != R(0);
      }" UNI20_HAS_LGAMMA_R_${suffix})
    check_cxx_source_compiles("
      #include <cmath>
      #include <type_traits>
      int main() {
        using R = ${type};
        static_assert(std::is_same_v<decltype(${jn}(0, R(1))), R>);
        static_assert(std::is_same_v<decltype(${yn}(0, R(1))), R>);
        R (*volatile first)(int, R) = &${jn};
        R (*volatile second)(int, R) = &${yn};
        return first(0, R(1)) == second(0, R(1));
      }" UNI20_HAS_BESSEL_${suffix})
  endblock()
endfunction()

uni20_check_native_special_math(FLOAT "float" lgammaf_r jnf ynf)
uni20_check_native_special_math(DOUBLE "double" lgamma_r jn yn)
uni20_check_native_special_math(LONG_DOUBLE "long double" lgammal_r jnl ynl)
if(UNI20_HAS_FLOAT128)
  uni20_check_native_special_math(FLOAT128 "_Float128" lgammaf128_r jnf128 ynf128)
else()
  set(UNI20_HAS_LGAMMA_R_FLOAT128 OFF)
  set(UNI20_HAS_BESSEL_FLOAT128 OFF)
endif()

# C23 pi-scaled functions are not implied by a C++ feature macro or a compiler
# builtin. Require exact signatures and linked symbols for each scalar family.
function(uni20_check_native_pi_math suffix type function_suffix)
  block(SCOPE_FOR VARIABLES)
    set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
    foreach(family IN ITEMS TRIG INVERSE)
      if(family STREQUAL "TRIG")
        set(functions sinpi cospi tanpi)
      else()
        set(functions asinpi acospi atanpi)
      endif()
      set(probe "#include <cmath>\n#include <type_traits>\nint main() { using R = ${type};\n")
      foreach(function IN LISTS functions)
        string(APPEND probe "
          static_assert(std::is_same_v<decltype(::${function}${function_suffix}(R(0))), R>);
          R (*volatile ${function}_ptr)(R) = &::${function}${function_suffix};
          (void)${function}_ptr(R(0));\n")
      endforeach()
      if(family STREQUAL "INVERSE")
        string(APPEND probe "
          static_assert(std::is_same_v<decltype(::atan2pi${function_suffix}(R(0), R(1))), R>);
          R (*volatile atan2pi_ptr)(R, R) = &::atan2pi${function_suffix};
          (void)atan2pi_ptr(R(0), R(1));\n")
      endif()
      string(APPEND probe "return 0; }")
      check_cxx_source_compiles("${probe}" UNI20_HAS_PI_${family}_${suffix})
    endforeach()
  endblock()
endfunction()

uni20_check_native_pi_math(FLOAT "float" "f")
uni20_check_native_pi_math(DOUBLE "double" "")
uni20_check_native_pi_math(LONG_DOUBLE "long double" "l")
if(UNI20_HAS_FLOAT128)
  uni20_check_native_pi_math(FLOAT128 "_Float128" "f128")
else()
  set(UNI20_HAS_PI_TRIG_FLOAT128 OFF)
  set(UNI20_HAS_PI_INVERSE_FLOAT128 OFF)
endif()

# Unary extensions may be introduced independently by the platform libc.
function(uni20_check_native_exponential_math suffix type function_suffix)
  block(SCOPE_FOR VARIABLES)
    set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
    foreach(function IN ITEMS exp10 exp2m1 exp10m1 log2p1 log10p1)
      string(TOUPPER "${function}" upper)
      check_cxx_source_compiles("
        #include <cmath>
        #include <type_traits>
        int main() {
          using R = ${type};
          static_assert(std::is_same_v<decltype(::${function}${function_suffix}(R(0))), R>);
          R (*volatile function)(R) = &::${function}${function_suffix};
          return function(R(0)) != R(0);
        }" UNI20_HAS_${upper}_${suffix})
    endforeach()
  endblock()
endfunction()

uni20_check_native_exponential_math(FLOAT "float" "f")
uni20_check_native_exponential_math(DOUBLE "double" "")
uni20_check_native_exponential_math(LONG_DOUBLE "long double" "l")
if(UNI20_HAS_FLOAT128)
  uni20_check_native_exponential_math(FLOAT128 "_Float128" "f128")
else()
  foreach(function IN ITEMS EXP10 EXP2M1 EXP10M1 LOG2P1 LOG10P1)
    set(UNI20_HAS_${function}_FLOAT128 OFF)
  endforeach()
endif()
