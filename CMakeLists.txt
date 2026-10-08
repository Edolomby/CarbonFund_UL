cmake_minimum_required(VERSION 3.22)
if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()
project(CarbonFund_UL VERSION 0.1.0 LANGUAGES CXX)

# ==============================================================================
# 1. COMPILER CONFIGURATION
# ==============================================================================

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CARBONFUND_ENABLE_OPENMP "Enable OpenMP support" ON)

# ==============================================================================
# 2. DEPENDENCIES
# ==============================================================================

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

# Prefer a local Boost bundle unless a root was explicitly supplied.
# An external directory can be supplied with -DBOOST_ROOT=/path/to/boost.
if(NOT DEFINED BOOST_ROOT
   AND NOT DEFINED Boost_ROOT
   AND NOT DEFINED ENV{BOOST_ROOT}
   AND NOT DEFINED ENV{Boost_ROOT}
   AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/boost_1_90_0")
    set(BOOST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/boost_1_90_0")
endif()

find_package(Boost 1.88.0 REQUIRED)

if(CARBONFUND_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

# ==============================================================================
# 3. LIBRARY TARGET 
# ==============================================================================

# Header-only library
add_library(carbonfund INTERFACE)
add_library(CarbonFund::carbonfund ALIAS carbonfund)

target_compile_features(carbonfund INTERFACE cxx_std_20)

target_include_directories(carbonfund INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(carbonfund INTERFACE
    Eigen3::Eigen
    Boost::headers
)

if(CARBONFUND_ENABLE_OPENMP)
    target_link_libraries(carbonfund INTERFACE
        OpenMP::OpenMP_CXX
    )
endif()


#DEPENDENCY COMPILATION CHECK
add_executable(check_dependencies examples/check_dependencies.cpp)

target_link_libraries(check_dependencies PRIVATE
    CarbonFund::carbonfund
)


# ==============================================================================
# 4. Automated tests
# ==============================================================================

include(CTest)

if(BUILD_TESTING)
    add_executable(test_carbon_models tests/test_carbon_models.cpp)
    target_link_libraries(test_carbon_models PRIVATE CarbonFund::carbonfund)
    add_test(NAME carbon_models COMMAND test_carbon_models)
    
    add_executable(test_simulation_rng tests/test_simulation_rng.cpp)
    target_link_libraries(test_simulation_rng PRIVATE CarbonFund::carbonfund)
    add_test(NAME simulation_rng COMMAND test_simulation_rng)
    
    add_executable(test_carbon_allocation tests/test_carbon_allocation.cpp)
    target_link_libraries(test_carbon_allocation PRIVATE CarbonFund::carbonfund)
    add_test(NAME carbon_allocation COMMAND test_carbon_allocation)
    
    add_executable(test_gompertz_makeham tests/test_gompertz_makeham.cpp)
    target_link_libraries(test_gompertz_makeham PRIVATE CarbonFund::carbonfund)
    add_test(NAME gompertz_makeham COMMAND test_gompertz_makeham)
    
    add_executable(test_cohorts tests/test_cohorts.cpp)
    target_link_libraries(test_cohorts PRIVATE CarbonFund::carbonfund)
    add_test(NAME cohorts COMMAND test_cohorts)
    
    add_executable(test_petiei_mortality_weights tests/test_petiei_mortality_weights.cpp)
    target_link_libraries(test_petiei_mortality_weights PRIVATE carbonfund)
    add_test(NAME petiei_mortality_weights COMMAND test_petiei_mortality_weights)
endif()


# ==============================================================================
# 5. STATUS REPORT
# ==============================================================================

message(STATUS "---------------------------------------------------")
message(STATUS "CarbonFund_UL Configured")
message(STATUS "Boost version: ${Boost_VERSION}")
message(STATUS "Boost include directories: ${Boost_INCLUDE_DIRS}")
message(STATUS "OpenMP enabled: ${CARBONFUND_ENABLE_OPENMP}")
message(STATUS "---------------------------------------------------")
