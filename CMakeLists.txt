cmake_minimum_required(VERSION 3.15...4.0)

project(
    PastelMath 
    VERSION 0.1.0
    DESCRIPTION "Educational header-only math library for graphics"
    LANGUAGES CXX
)

add_library(PastelMath INTERFACE)
add_library(PastelMath::PastelMath ALIAS PastelMath)

target_include_directories(
    PastelMath INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(PastelMath INTERFACE cxx_std_20)

option(PASTELMATH_BUILD_TESTS "Build tests" ON)
option(PASTELMATH_BUILD_EXAMPLES "Build examples" ON)

if(PASTELMATH_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

if(PASTELMATH_BUILD_EXAMPLES AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/examples)
    add_subdirectory(examples)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(
    DIRECTORY include/PastelMath
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    TARGETS PastelMath
    EXPORT PastelMathTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    EXPORT PastelMathTargets
    FILE PastelMathTargets.cmake
    NAMESPACE PastelMath::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PastelMath
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/PastelMathConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/PastelMathConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PastelMath
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/PastelMathConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
    ${CMAKE_CURRENT_BINARY_DIR}/PastelMathConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/PastelMathConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PastelMath
)