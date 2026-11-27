# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli

cmake_minimum_required(VERSION 3.16)

project(libwex VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(WEX_BUILD_SHARED "Build libwex as a shared library" ON)
option(WEX_BUILD_CUDA   "Build optional libwex_cuda CUDA backend" OFF)
option(WEX_BUILD_OPENMP "Enable OpenMP parallel CPU decode" OFF)
option(WEX_BUILD_TESTS  "Build libwex C/IO unit tests" ON)

include(GNUInstallDirs)

if(WIN32)
    set(WEX_IO_SRC src/wex_io_win.c)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(WEX_IO_SRC src/wex_io_linux.c)
else()
    set(WEX_IO_SRC src/wex_io_posix.c)
endif()

set(WEX_CORE_SRCS
    src/wex_store.c
    src/wex_cache.c
    src/wex_verify.c
    ${WEX_IO_SRC}
)

if(WEX_BUILD_SHARED)
    add_library(wex SHARED ${WEX_CORE_SRCS})
    target_compile_definitions(wex PRIVATE LIBWEX_BUILDING_DLL)
    if(WIN32)
        set_target_properties(wex PROPERTIES PREFIX "lib")
    endif()
else()
    add_library(wex STATIC ${WEX_CORE_SRCS})
endif()

set_target_properties(wex PROPERTIES
    OUTPUT_NAME wex
    C_VISIBILITY_PRESET default
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(wex
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../c
)

if(MSVC)
    target_compile_options(wex PRIVATE /W3)
    target_compile_definitions(wex PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(wex PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-unused-function)
endif()

if(NOT WIN32)
    find_package(Threads REQUIRED)
    target_link_libraries(wex PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
endif()

if(WEX_BUILD_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(wex PRIVATE OpenMP::OpenMP_C)
endif()

set(WEX_INSTALL_TARGETS wex)

if(WEX_BUILD_CUDA)
    enable_language(CUDA)
    add_library(wex_cuda SHARED src/wex_dev_cuda.cu)
    set_target_properties(wex_cuda PROPERTIES
        OUTPUT_NAME wex_cuda
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )
    if(WIN32)
        set_target_properties(wex_cuda PROPERTIES PREFIX "lib")
    endif()
    target_include_directories(wex_cuda PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../c
    )
    list(APPEND WEX_INSTALL_TARGETS wex_cuda)
endif()

if(WEX_BUILD_TESTS)
    enable_testing()

    add_executable(test_libwex_io
        tests/test_libwex_io.c
        ${WEX_IO_SRC}
        src/wex_verify.c
    )
    target_include_directories(test_libwex_io PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../c
    )
    if(MSVC)
        target_compile_options(test_libwex_io PRIVATE /W3)
        target_compile_definitions(test_libwex_io PRIVATE _CRT_SECURE_NO_WARNINGS)
    else()
        target_compile_options(test_libwex_io PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-unused-function)
    endif()
    if(NOT WIN32)
        target_link_libraries(test_libwex_io PRIVATE Threads::Threads)
    endif()
    add_test(NAME test_libwex_io COMMAND test_libwex_io)

    add_executable(test_libwex_c
        tests/test_libwex_c.c
        ${WEX_CORE_SRCS}
    )
    target_include_directories(test_libwex_c PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../c
    )
    if(MSVC)
        target_compile_options(test_libwex_c PRIVATE /W3)
        target_compile_definitions(test_libwex_c PRIVATE _CRT_SECURE_NO_WARNINGS)
    else()
        target_compile_options(test_libwex_c PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-unused-function)
    endif()
    if(NOT WIN32)
        target_link_libraries(test_libwex_c PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
    endif()
    if(WEX_BUILD_OPENMP)
        target_link_libraries(test_libwex_c PRIVATE OpenMP::OpenMP_C)
    endif()

    if(WEX_BUILD_SHARED)
        add_test(
            NAME test_libwex_c
            COMMAND test_libwex_c --lib $<TARGET_FILE:wex>
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
    else()
        add_test(
            NAME test_libwex_c
            COMMAND test_libwex_c
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
    endif()
endif()

install(TARGETS ${WEX_INSTALL_TARGETS}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(FILES
    include/wex_store.h
    include/wex_loader.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
