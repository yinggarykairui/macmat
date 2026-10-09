cmake_minimum_required(VERSION 3.20)
project(macmat LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Release: -O3 -DNDEBUG with debug symbols retained for Instruments.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -g")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

# MACMAT_SANITIZE = none | address | thread   (address also enables undefined)
set(MACMAT_SANITIZE "none" CACHE STRING "Sanitizer: none, address (ASan+UBSan) or thread (TSan)")
# MACMAT_VECTORIZE_REPORT = ON prints Clang's loop-vectorize remarks for the kernels.
option(MACMAT_VECTORIZE_REPORT "Emit -Rpass=loop-vectorize diagnostics" OFF)

add_library(macmat STATIC src/matrix.cpp src/multiply.cpp src/kernels.cpp src/accelerate.cpp src/micro.cpp)
target_include_directories(macmat PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(macmat PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(macmat PUBLIC "-framework Accelerate")
find_package(Threads REQUIRED)
target_link_libraries(macmat PUBLIC Threads::Threads)

if(MACMAT_VECTORIZE_REPORT)
  set_source_files_properties(src/kernels.cpp PROPERTIES COMPILE_OPTIONS
    "-Rpass=loop-vectorize;-Rpass-missed=loop-vectorize;-Rpass-analysis=loop-vectorize")
endif()

if(MACMAT_SANITIZE STREQUAL "address")
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
elseif(MACMAT_SANITIZE STREQUAL "thread")
  add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
  add_link_options(-fsanitize=thread)
endif()

# Record the exact flags in the benchmark binary so CSV rows are self-describing.
string(TOUPPER "${CMAKE_BUILD_TYPE}" _bt)
set(_flags "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_bt}}")
if(NOT MACMAT_SANITIZE STREQUAL "none")
  string(APPEND _flags " -fsanitize=${MACMAT_SANITIZE}")
endif()
string(STRIP "${_flags}" _flags)

add_executable(macmat_cli benchmarks/main.cpp benchmarks/bench.cpp)
set_target_properties(macmat_cli PROPERTIES OUTPUT_NAME macmat)
target_link_libraries(macmat_cli PRIVATE macmat)
target_compile_definitions(macmat_cli PRIVATE MACMAT_CXX_FLAGS="${_flags}")

enable_testing()
add_executable(macmat_tests tests/test_correctness.cpp)
target_link_libraries(macmat_tests PRIVATE macmat)
add_test(NAME correctness COMMAND macmat_tests)
