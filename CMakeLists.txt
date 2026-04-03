cmake_minimum_required(VERSION 3.16)
project(PowerGridRL CXX)
set(CMAKE_CXX_STANDARD 17)

# ── Raylib ────────────────────────────────────────────────────────────────────
find_package(raylib QUIET)
if (NOT raylib_FOUND)
    include(FetchContent)
    set(FETCHCONTENT_QUIET OFF)
    FetchContent_Declare(raylib
        GIT_REPOSITORY https://github.com/raysan5/raylib.git
        GIT_TAG 5.0)
    FetchContent_MakeAvailable(raylib)
endif()

# ── ZeroMQ + cppzmq ───────────────────────────────────────────────────────────
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)
find_path(CPPZMQ_INCLUDE zmq.hpp)   # MSYS2: /mingw64/include  |  Linux: /usr/include

# ── nlohmann/json ─────────────────────────────────────────────────────────────
find_package(nlohmann_json QUIET)
if (NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3)
    FetchContent_MakeAvailable(json)
endif()

# ── Executable ────────────────────────────────────────────────────────────────
# Source file is simulator.cpp (same directory as CMakeLists.txt)
add_executable(powergrid_rl simulator.cpp)

target_include_directories(powergrid_rl PRIVATE
    ${CPPZMQ_INCLUDE}
    ${ZMQ_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}   # finds rl_bridge.h in the same folder
)

target_link_libraries(powergrid_rl PRIVATE
    raylib
    ${ZMQ_LIBRARIES}
    nlohmann_json::nlohmann_json
)

# ── Platform-specific ─────────────────────────────────────────────────────────
if (WIN32)
    target_link_libraries(powergrid_rl PRIVATE winmm gdi32 opengl32)
elseif (APPLE)
    target_link_libraries(powergrid_rl PRIVATE
        "-framework OpenGL" "-framework Cocoa" "-framework IOKit")
elseif (UNIX)
    target_link_libraries(powergrid_rl PRIVATE GL pthread dl m)
endif()
