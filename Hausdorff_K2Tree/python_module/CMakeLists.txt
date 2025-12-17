cmake_minimum_required(VERSION 3.12)
project(hausdorff_k2tree_python)

# Configurar C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Encontrar pybind11
find_package(pybind11 REQUIRED HINTS ${PYTHON_EXECUTABLE} ${PYTHON_INCLUDE_DIR})

# Incluir directorios del proyecto original
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../libcds/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../Snapshot
    ${CMAKE_CURRENT_SOURCE_DIR}/../Util
    ${CMAKE_CURRENT_SOURCE_DIR}/../IO
    ${CMAKE_CURRENT_SOURCE_DIR}/../Logs
)

# Agregar libcds
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../libcds libcds_build)

# Recopilar archivos fuente del proyecto original
file(GLOB_RECURSE SNAPSHOT_SOURCES "../Snapshot/*.cpp")
file(GLOB_RECURSE UTIL_SOURCES "../Util/*.cpp")
file(GLOB_RECURSE IO_SOURCES "../IO/*.cpp")
file(GLOB_RECURSE LOGS_SOURCES "../Logs/*.cpp")

# Archivos específicos
set(PROJECT_SOURCES
    ${SNAPSHOT_SOURCES}
    ${UTIL_SOURCES}
    ${IO_SOURCES}
    ${LOGS_SOURCES}
    ../ryu-kamata.cpp
    ../utils.cpp
)

# Filtrar main.cpp si existe
list(FILTER PROJECT_SOURCES EXCLUDE REGEX ".*main\\.cpp$")

# Crear módulo pybind11
pybind11_add_module(hausdorff_k2tree_core src/python_bindings.cpp ${PROJECT_SOURCES})

# Vincular con libcds
target_link_libraries(hausdorff_k2tree_core PRIVATE cds)

# Propiedades del compilador para el módulo
target_compile_definitions(hausdorff_k2tree_core PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

# Configuraciones específicas para Windows
if(WIN32)
    target_compile_definitions(hausdorff_k2tree_core PRIVATE _WIN32_WINNT=0x0601)
endif()

# Configuraciones de optimización
target_compile_options(hausdorff_k2tree_core PRIVATE 
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Debug>:-g>
)