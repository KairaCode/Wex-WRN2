# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli

cmake_minimum_required(VERSION 3.16)
project(ggml_wex VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(GGML_INCLUDE_DIR "" CACHE PATH "Optional path to llama.cpp / ggml include directory to enable GGML_WEX_WITH_GGML")
option(GGML_WEX_BUILD_TESTS "Build standalone test_ggml_wex executable" ON)

add_library(ggml-wex SHARED
    ggml_wex.c
    ggml_wex.h
)

set_target_properties(ggml-wex PROPERTIES
    OUTPUT_NAME "ggml-wex"
    C_VISIBILITY_PRESET hidden
)

target_compile_definitions(ggml-wex PRIVATE GGML_WEX_BUILDING_DLL)
target_include_directories(ggml-wex
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)

if(GGML_INCLUDE_DIR)
    target_compile_definitions(ggml-wex PUBLIC GGML_WEX_WITH_GGML)
    target_include_directories(ggml-wex PUBLIC "${GGML_INCLUDE_DIR}")
endif()

if(NOT WIN32)
    target_link_libraries(ggml-wex PRIVATE ${CMAKE_DL_LIBS})
endif()

if(MSVC)
    target_compile_options(ggml-wex PRIVATE /W4)
else()
    target_compile_options(ggml-wex PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

if(GGML_WEX_BUILD_TESTS)
    add_executable(test_ggml_wex test_ggml_wex.c)
    target_include_directories(test_ggml_wex PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../c"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../wnr4"
    )
    if(NOT WIN32)
        target_link_libraries(test_ggml_wex PRIVATE ${CMAKE_DL_LIBS} m)
    endif()
    if(MSVC)
        target_compile_definitions(test_ggml_wex PRIVATE _CRT_SECURE_NO_WARNINGS)
        target_compile_options(test_ggml_wex PRIVATE /W4)
    else()
        target_compile_options(test_ggml_wex PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-unused-function)
    endif()
endif()
