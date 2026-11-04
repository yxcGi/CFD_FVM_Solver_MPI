file(GLOB_RECURSE CORE_SOURCES CONFIGURE_DEPENDS
    Geometry/*.cpp
    Math/*.cpp
    Parallel/*.cpp
)

# GPU 后端：启用 CUDA 时编译 Gpu.cu；CFD_GPU_EMULATE=ON 时编译主机模拟实现（仅测试用）；
# 否则编译占位实现（始终使用 CPU）
if(CFD_USE_CUDA)
    list(APPEND CORE_SOURCES Gpu/Gpu.cu)
elseif(CFD_GPU_EMULATE)
    list(APPEND CORE_SOURCES Gpu/GpuEmulate.cpp)
else()
    list(APPEND CORE_SOURCES Gpu/GpuStub.cpp)
endif()

add_library(core SHARED
    ${CORE_SOURCES}
)

target_include_directories(core
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/Geometry
    ${CMAKE_CURRENT_SOURCE_DIR}/Math
    ${CMAKE_CURRENT_SOURCE_DIR}/Parallel
    ${CMAKE_CURRENT_SOURCE_DIR}/Gpu
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/FieldOperators
    ${CMAKE_CURRENT_SOURCE_DIR}/FVM
    ${CMAKE_CURRENT_SOURCE_DIR}/thread_pool
    ${CMAKE_CURRENT_SOURCE_DIR}/algorithm
    ${CMAKE_CURRENT_SOURCE_DIR}/algorithm/SIMPLE
)

target_compile_features(core
    PUBLIC
    cxx_std_20
)

target_link_libraries(core PUBLIC Threads::Threads)

if(CFD_USE_MPI)
    target_link_libraries(core PUBLIC MPI::MPI_CXX)
    target_compile_definitions(core PUBLIC CFD_USE_MPI)
endif()

if(CFD_USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(core PUBLIC CUDA::cudart)
    target_compile_definitions(core PUBLIC CFD_USE_CUDA)
    set_target_properties(core PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    # -fmad=false：GPU 上同样禁止 FMA 融合，保证与 CPU 结果逐位一致；
    # 不使用 --use_fast_math（保持 IEEE 精确的除法与开方）
    target_compile_options(core PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-fmad=false>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
endif()
