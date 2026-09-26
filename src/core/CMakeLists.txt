file(GLOB_RECURSE CORE_SOURCES CONFIGURE_DEPENDS
    Geometry/*.cpp
    Math/*.cpp
    Parallel/*.cpp
)

add_library(core SHARED
    ${CORE_SOURCES}
)

target_include_directories(core
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/Geometry
    ${CMAKE_CURRENT_SOURCE_DIR}/Math
    ${CMAKE_CURRENT_SOURCE_DIR}/Parallel
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
