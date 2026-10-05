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
