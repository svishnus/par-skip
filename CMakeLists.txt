# mskip: parallel metric skip lists on ParlayLib (header-only).
#
# As a dependency:
#   find_package(mskip CONFIG REQUIRED)      # after `cmake --install`
#   target_link_libraries(app PRIVATE mskip::mskip)
# or
#   add_subdirectory(path/to/par-skip)       # or FetchContent; needs the submodule
#   target_link_libraries(app PRIVATE mskip::mskip)
# ParlayLib is taken from an installed `Parlay` package when one is found,
# otherwise from the external/parlaylib submodule.
cmake_minimum_required(VERSION 3.14)
project(mskip VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set(MSKIP_IS_TOP_LEVEL ON)
else()
  set(MSKIP_IS_TOP_LEVEL OFF)
endif()
option(MSKIP_BUILD_TESTS "Build the tests" ${MSKIP_IS_TOP_LEVEL})
option(MSKIP_BUILD_BENCH "Build the benchmarks" ${MSKIP_IS_TOP_LEVEL})
option(MSKIP_BUILD_EXAMPLES "Build the examples" ${MSKIP_IS_TOP_LEVEL})

# --- ParlayLib -------------------------------------------------------------
if(NOT TARGET Parlay::parlay)
  find_package(Parlay CONFIG QUIET)
endif()
if(NOT TARGET Parlay::parlay)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/parlaylib/CMakeLists.txt")
    add_subdirectory(external/parlaylib)  # its install rules put ParlayLib into the same prefix
    # treat ParlayLib's headers as system headers, as an installed package would be
    get_target_property(_parlay_inc parlay INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(parlay PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_parlay_inc}")
    add_library(Parlay::parlay ALIAS parlay)
    set(MSKIP_PARLAY_FROM_SUBMODULE ON)
  else()
    message(FATAL_ERROR "ParlayLib not found: install it (find_package(Parlay)) or run `git submodule update --init`")
  endif()
endif()

# --- the library -----------------------------------------------------------
add_library(mskip INTERFACE)
add_library(mskip::mskip ALIAS mskip)
target_include_directories(mskip INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(mskip INTERFACE cxx_std_17)
target_link_libraries(mskip INTERFACE Parlay::parlay)

# --- tests, benchmarks, examples -------------------------------------------
if(MSKIP_BUILD_TESTS OR MSKIP_BUILD_BENCH OR MSKIP_BUILD_EXAMPLES)
  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  endif()
  add_library(mskip_dev INTERFACE)
  target_link_libraries(mskip_dev INTERFACE mskip)
  target_compile_options(mskip_dev INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>)
  option(MSKIP_NATIVE "Compile tests and benchmarks with -march=native" ON)
  if(MSKIP_NATIVE)
    target_compile_options(mskip_dev INTERFACE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-march=native>)
  endif()
  option(MSKIP_SEQUENTIAL "Define PARLAY_SEQUENTIAL (single-threaded)" OFF)
  if(MSKIP_SEQUENTIAL)
    target_compile_definitions(mskip_dev INTERFACE PARLAY_SEQUENTIAL)
  endif()
  find_package(Threads REQUIRED)
  target_link_libraries(mskip_dev INTERFACE Threads::Threads)
endif()

if(MSKIP_BUILD_TESTS)
  enable_testing()
  file(GLOB MSKIP_TESTS CONFIGURE_DEPENDS tests/*.cpp)
  foreach(src ${MSKIP_TESTS})
    get_filename_component(name ${src} NAME_WE)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE mskip_dev)
    target_include_directories(${name} PRIVATE tests)
    add_test(NAME ${name} COMMAND ${name})
  endforeach()
endif()

if(MSKIP_BUILD_BENCH)
  file(GLOB MSKIP_BENCHES CONFIGURE_DEPENDS bench/*.cpp)
  foreach(src ${MSKIP_BENCHES})
    get_filename_component(name ${src} NAME_WE)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE mskip_dev)
    target_include_directories(${name} PRIVATE bench)
  endforeach()
endif()

if(MSKIP_BUILD_EXAMPLES)
  file(GLOB MSKIP_EXAMPLES CONFIGURE_DEPENDS examples/*.cpp)
  foreach(src ${MSKIP_EXAMPLES})
    get_filename_component(name ${src} NAME_WE)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE mskip_dev)
  endforeach()
endif()

# --- install / export --------------------------------------------------------
install(TARGETS mskip EXPORT mskipTargets)
install(DIRECTORY include/mskip DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT mskipTargets NAMESPACE mskip:: DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/mskip/cmake)
configure_package_config_file(cmake/mskipConfig.cmake.in "${PROJECT_BINARY_DIR}/mskipConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/mskip/cmake)
write_basic_package_version_file("${PROJECT_BINARY_DIR}/mskipConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES "${PROJECT_BINARY_DIR}/mskipConfig.cmake" "${PROJECT_BINARY_DIR}/mskipConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/mskip/cmake)
