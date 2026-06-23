#include "../include/pose_estimator.hpp"
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <array>
#include <filesystem>
#include <onnxruntime_cxx_api.h>

PoseEstimate detectSinglePose(const cv::Mat& image, const std::string& modelPath) {
    const int IMAGE_SIZE = 256;
    cv::Mat resized_image, rgb;
    PoseEstimate pose_estimate;

    cv::resize(image, resized_image, cv::Size(IMAGE_SIZE, IMAGE_SIZE));

    cv::cvtColor(resized_image, rgb, cv::COLOR_BGR2RGB);

    rgb.convertTo(rgb, CV_32S);

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::array<int64_t, 4> input_shape = {1, IMAGE_SIZE, IMAGE_SIZE, 3};

    Ort::Value input_tensor = Ort::Value::CreateTensor<int32_t>(
        mem_info,
        (int32_t*)rgb.data,
        IMAGE_SIZE * IMAGE_SIZE * 3,
        input_shape.data(),
        input_shape.size()
    );

    std::wstring wpath(modelPath.begin(), modelPath.end());

    // TODO: Persist env and session
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "pose");
    Ort::Session session(env, wpath.c_str(), Ort::SessionOptions{});

    const char* input_names[] = {"input"};
    const char* output_names[] = {"output_0"};

    auto output_tensors = session.Run(
        Ort::RunOptions{},
        input_names, &input_tensor, 1,
        output_names, 1
    );

    float* output_data = output_tensors[0].GetTensorMutableData<float>();

    for (int k = 0;k < 17; k++) {
        pose_estimate.keypoints[k].x = output_data[k * 3 + 1] * image.cols;
        pose_estimate.keypoints[k].y = output_data[k * 3] * image.rows;
        pose_estimate.keypoints[k].confidence = output_data[k * 3 + 2];
    }

    return pose_estimate;
}