include(CheckSourceCompiles)
include(CMakePushCheckState)

function(uni20_select_extents)
  cmake_push_check_state(RESET)
  # This probe only needs to compile. In CUDA builds use the same language mode
  # and relaxed constexpr setting as Uni20's device targets.
  set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
  check_source_compiles(CXX [=[
    #include <mdspan>
    using Shape = std::extents<int, 2, std::dynamic_extent>;
    static_assert(Shape{3}.extent(1) == 3);
    static_assert(std::dextents<int, 0>::rank() == 0);
  ]=] UNI20_CXX_HAS_STD_EXTENTS)

  set(use_std_extents ${UNI20_CXX_HAS_STD_EXTENTS})
  if(use_std_extents AND UNI20_ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_REQUIRED_FLAGS "--expt-relaxed-constexpr")
    check_source_compiles(CUDA [=[
      #include <mdspan>
      using Shape = std::extents<int, 2, std::dynamic_extent>;
      __global__ void read_extent(Shape extents, int* output)
      {
        *output = extents.extent(1);
      }
    ]=] UNI20_CUDA_HAS_STD_EXTENTS)
    set(use_std_extents ${UNI20_CUDA_HAS_STD_EXTENTS})
  endif()
  cmake_pop_check_state()

  if(use_std_extents)
    message(STATUS "Uni20 extents: standard library")
  else()
    message(STATUS "Uni20 extents: built-in fallback")
  endif()
  set(UNI20_USE_STD_EXTENTS ${use_std_extents} PARENT_SCOPE)
endfunction()
