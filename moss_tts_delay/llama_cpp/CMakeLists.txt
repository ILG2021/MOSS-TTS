cmake_minimum_required(VERSION 3.21)
project(moss_tts_backbone_bridge C)

set(LLAMA_CPP_DIR "" CACHE PATH "Path to a built llama.cpp source tree")
if(NOT LLAMA_CPP_DIR)
    message(FATAL_ERROR "Set LLAMA_CPP_DIR to the llama.cpp source directory")
endif()

set(_llama_lib_hints
    "${LLAMA_CPP_DIR}/build/bin/Release"
    "${LLAMA_CPP_DIR}/build/bin"
    "${LLAMA_CPP_DIR}/build/src/Release"
    "${LLAMA_CPP_DIR}/build/src"
    "${LLAMA_CPP_DIR}/build/Release"
    "${LLAMA_CPP_DIR}/build")

find_library(LLAMA_LIBRARY NAMES llama HINTS ${_llama_lib_hints} NO_DEFAULT_PATH)
if(NOT LLAMA_LIBRARY)
    message(FATAL_ERROR
        "Could not find the llama library. Build llama.cpp with BUILD_SHARED_LIBS=ON first. "
        "Searched: ${_llama_lib_hints}")
endif()

add_library(backbone_bridge SHARED backbone_bridge.c)
target_include_directories(backbone_bridge PRIVATE
    "${LLAMA_CPP_DIR}/include"
    "${LLAMA_CPP_DIR}/ggml/include")
target_link_libraries(backbone_bridge PRIVATE "${LLAMA_LIBRARY}")

set_target_properties(backbone_bridge PROPERTIES
    PREFIX ""
    OUTPUT_NAME "backbone_bridge")

if(NOT WIN32)
    set_target_properties(backbone_bridge PROPERTIES
        BUILD_RPATH "${LLAMA_CPP_DIR}/build/bin")
endif()
