cmake_minimum_required(VERSION 3.15)

get_filename_component(PRJ_NAME ${CMAKE_SOURCE_DIR} NAME)

project(${PRJ_NAME}
    VERSION 0.1.0
    DESCRIPTION "A lightweight finite volume method CFD solver"
    LANGUAGES CXX
    HOMEPAGE_URL "https://github.com/yxcGi/CFD_FVM_Solver"
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g -O0 -Wall -Wextra -pedantic)
else()
    add_compile_options(-O3 -DNDEBUG)
endif()

# 禁止编译器把 a*b+c 融合为 FMA：保证不同进程数、不同编译选项下的浮点结果一致
add_compile_options(-ffp-contract=off)

# ---------------- MPI ----------------
# 默认启用 MPI；找不到 MPI 或 -DCFD_USE_MPI=OFF 时编译为单进程版本（与原串行程序一致）
option(CFD_USE_MPI "Build the MPI parallel solver" ON)
if(CFD_USE_MPI)
    find_package(MPI COMPONENTS CXX)
    if(MPI_CXX_FOUND)
        message(STATUS "MPI found: ${MPI_CXX_COMPILER} (parallel build)")
    else()
        message(WARNING "MPI not found, building the single-process version")
        set(CFD_USE_MPI OFF)
    endif()
endif()
find_package(Threads REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

add_subdirectory(src)
add_subdirectory(example)
