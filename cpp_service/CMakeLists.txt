cmake_minimum_required(VERSION 3.16)
project(myhome_verify_server LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)
find_package(CURL REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    /opt/homebrew/include
)

link_directories(/opt/homebrew/lib)

add_executable(myhome_verify_server
    src/main.cpp
    src/face_engine.cpp
    src/gesture_engine.cpp
)

target_link_libraries(myhome_verify_server PRIVATE
    ${OpenCV_LIBS}
    CURL::libcurl
    yaml-cpp
    nlohmann_json::nlohmann_json
)
