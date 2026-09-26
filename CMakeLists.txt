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
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-g> $<$<COMPILE_LANGUAGE:CXX>:-O0>
                        $<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                        $<$<COMPILE_LANGUAGE:CXX>:-pedantic>)
else()
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-O3> -DNDEBUG)
endif()

# 禁止编译器把 a*b+c 融合为 FMA：保证不同进程数、不同编译选项下的浮点结果一致
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)

# ---------------- CUDA (GPU) ----------------
# 测试用：没有 GPU 时用主机代码模拟 GPU 后端（与 CFD_USE_CUDA 互斥，仅用于验证 GPU 路径）
option(CFD_GPU_EMULATE "Emulate the GPU backend on the host (testing only)" OFF)
if(CFD_GPU_EMULATE)
    set(CFD_USE_CUDA OFF)
endif()
# 默认尝试启用 CUDA；找不到 CUDA 编译器或 -DCFD_USE_CUDA=OFF 时只编译 CPU 版本。
# 启用后若运行时没有 GPU（或设置 CFD_USE_GPU=0），程序自动使用 CPU，结果不变。
option(CFD_USE_CUDA "Build the CUDA GPU backend" ON)
if(CFD_USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            # 常见架构：V100(70) T4/RTX20(75) A100(80) RTX30(86) RTX40(89) H100(90) RTX50(120)
            # 可用 -DCMAKE_CUDA_ARCHITECTURES=86 之类只编译自己显卡的架构
            # CUDA 13 起不再支持 sm_70（Volta），CUDA 12.8 起支持 sm_120
            execute_process(COMMAND ${CMAKE_CUDA_COMPILER} --version
                            OUTPUT_VARIABLE CFD_NVCC_VERSION_OUTPUT ERROR_QUIET)
            string(REGEX MATCH "release ([0-9]+)\\.([0-9]+)" _ "${CFD_NVCC_VERSION_OUTPUT}")
            set(CFD_NVCC_VERSION "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")
            if(CFD_NVCC_VERSION VERSION_GREATER_EQUAL 13.0)
                set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90 120)
            elseif(CFD_NVCC_VERSION VERSION_GREATER_EQUAL 12.8)
                set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90 120)
            else()
                set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
            endif()
        endif()
        enable_language(CUDA)
        message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER} (GPU build, arch ${CMAKE_CUDA_ARCHITECTURES})")
    else()
        message(WARNING "CUDA compiler not found, building the CPU-only version")
        set(CFD_USE_CUDA OFF)
    endif()
endif()

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
