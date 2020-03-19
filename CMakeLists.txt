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

PROJECT(Obj_Det_Project)

set(PLATFORM "$ENV{PLATFORM}" CACHE INTERNAL "Copied from environment variable")
message("Platform: " ${PLATFORM})

if(WIN32)
    if(${PLATFORM} STREQUAL "Laptop")
        set(COMMON_ROOT "D:/Common")
        set(DLIB_CONTRIB "D:/Projects/dlib-contrib")
        include(D:/dlib-19.17/dlib/cmake)
    elseif(${PLATFORM} STREQUAL "LaptopN")
        set(COMMON_ROOT "C:/Projects/Common")
        set(DLIB_CONTRIB "C:/Projects/dlib-contrib")
        include(C:/dlib-19.17/dlib/cmake)    
    elseif(${PLATFORM} STREQUAL "BEAST")
        set(COMMON_ROOT "D:/Common")
        set(DLIB_CONTRIB "D:/Projects/dlib-contrib")
        include(D:/dlib-19.17/dlib/cmake)
    else()
        set(COMMON_ROOT "D:/Common")
        set(DLIB_CONTRIB "D:/Projects/dlib-contrib")
        include(D:/dlib-19.17/dlib/cmake)
    endif()    
else()
    if(${PLATFORM} STREQUAL "HPC")
        set(COMMON_ROOT "~/Common")
        set(DLIB_CONTRIB "~/Projects/dlib-contrib")
        include(~/dlib-19.19/dlib/cmake)
    else()
        set(COMMON_ROOT "~/Projects/Common")
        set(DLIB_CONTRIB "~/Projects/dlib-contrib")
        include(~/dlib-19.19/dlib/cmake)
    endif()
endif()


set(exe_name "obj_det")


# grab all of the *.cpp files in the src directory
file(GLOB SOURCES "src/*.cpp" 
    "${COMMON_ROOT}/src/*.cpp"
    "${DLIB_CONTRIB}/src/*.cpp"
    )
    
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
    "${DLIB_CONTRIB}/include" 
    )

# find OpenCV
if(${PLATFORM} STREQUAL "HPC")
    message("HPC detected.  OpenCV will not be used.")
else()
    find_package(OpenCV REQUIRED)
    if (OpenCV_FOUND)
        set(project_includes ${project_includes} ${OpenCV_INCLUDE_DIRS})
    else()
        message("OpenCV not found, so we won't build the project.")
    endif()
endif()

message("--------------------------------------------------------")
message("Project Includes: ")
foreach(S ${project_includes})
    message("  " ${S})
endforeach()
message("--------------------------------------------------------")
message(" ")

include_directories(${project_includes})
add_executable(${exe_name} ${SOURCES})
target_link_libraries(${exe_name} dlib::dlib ${OpenCV_LIBS})

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
