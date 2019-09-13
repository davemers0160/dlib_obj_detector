# ----------------------------------------------------------------------------
#  Root CMake file for ML Project
# ----------------------------------------------------------------------------
#
# Example CMake build line for windows & VS2015
# windows w/cuda: cmake -G "Visual Studio 14 2015 Win64" -DUSE_AVX_INSTRUCTIONS=ON ..
# linux w/o cuda: cmake -DUSE_AVX_INSTRUCTIONS=ON -DDLIB_USE_CUDA=OFF ..
#
#
# To compile:
# windows: cmake --build . --config Release
# linux: cmake --build . --config Release -- -j10
#


cmake_minimum_required(VERSION 2.8.12)

PROJECT(YJ_Project)

set(PLATFORM "$ENV{PLATFORM}" CACHE INTERNAL "Copied from environment variable")
message("Platform: " ${PLATFORM})

if(WIN32)
    if(${PLATFORM} STREQUAL "BEAST")
        include(D:/dlib-19.17/dlib/cmake)
        set(COMMON_ROOT "D:/Common")
    else()
        include(D:/dlib-19.17/dlib/cmake)
        set(COMMON_ROOT "D:/Common")
    endif()    
else()
    if(${PLATFORM} STREQUAL "HPC")
        include(~/dlib-19.17/dlib/cmake)
        set(COMMON_ROOT "~/Common")
    else()
        include(~/dlib-19.17/dlib/cmake)
        set(COMMON_ROOT "~/Common")
    endif()
endif()


set(exe_name "yj_ex")


# grab all of the *.cpp files in the src directory
file(GLOB SOURCES "src/*.cpp" )	

message(" ")
message("--------------------------------------------------------")
message("Source Files: ")
foreach(S ${SOURCES})
    message("  " ${S})
endforeach()
message("--------------------------------------------------------")
message(" ")

set(project_includes "include" 
	"${COMMON_ROOT}/include"
	)

message("--------------------------------------------------------")
message("Project Includes: ")
foreach(S ${project_includes})
    message("  " ${S})
endforeach()
message("--------------------------------------------------------")
message(" ")

include_directories(${project_includes})
add_executable(${exe_name} ${SOURCES})
target_link_libraries(${exe_name} dlib::dlib)

if(WIN32)
    message("MSVC Version: " ${MSVC_VERSION})
    # uncomment for MSVC and cmake greater than 3.6.0
    if(MSVC AND (${CMAKE_VERSION} VERSION_GREATER "3.6.0"))
        target_compile_options(${exe_name} PRIVATE "/MP")
        set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${exe_name})
    endif()
endif()

message("Build Complete!")
message(" ")
