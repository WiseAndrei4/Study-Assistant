cmake_minimum_required(VERSION 3.18)
project(hnsw_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(hnsw_core NB_STATIC cpp/bindings.cpp)
target_include_directories(hnsw_core PRIVATE cpp)

if(MSVC)
    target_compile_options(hnsw_core PRIVATE /O2 /arch:AVX2)
else()
    target_compile_options(hnsw_core PRIVATE -O3 -march=native)
endif()

nanobind_add_stub(
    hnsw_core_stub
    MODULE hnsw_core
    OUTPUT hnsw_core.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:hnsw_core>
    DEPENDS hnsw_core
)

install(TARGETS hnsw_core LIBRARY DESTINATION .)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/hnsw_core.pyi" DESTINATION .)