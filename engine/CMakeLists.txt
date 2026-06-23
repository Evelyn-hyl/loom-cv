cmake_minimum_required(VERSION 3.15)
project(PerspectiveSensei VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(ONNXRUNTIME_ROOT "C:/onnxruntime/onnxruntime-win-x64-1.27.0")

find_package(OpenCV REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(httplib CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${ONNXRUNTIME_ROOT}/include
)

add_executable(engine 
    src/main.cpp
    src/edge_detection.cpp
    src/vanishing_point.cpp
    src/pose_estimator.cpp
)

target_link_libraries(engine PRIVATE 
    ${OpenCV_LIBS}
    Eigen3::Eigen
    httplib::httplib
    nlohmann_json::nlohmann_json
    ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib
)