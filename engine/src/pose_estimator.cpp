#include "../include/pose_estimator.hpp"
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <array>
#include <filesystem>
#include <onnxruntime_cxx_api.h>
#include <opencv2/imgcodecs.hpp>

PoseEstimate detectSinglePose(const cv::Mat& image, const std::string& modelPath) {
    const int IMAGE_SIZE = 640;
    const double SCALE = std::min((double)IMAGE_SIZE / image.cols, (double)IMAGE_SIZE / image.rows);

    cv::Mat resized_image, rgb;
    std::vector<cv::Mat> channels;
    cv::Mat letterboxed(IMAGE_SIZE, IMAGE_SIZE, CV_8UC3, cv::Scalar(0, 0, 0));
    PoseEstimate pose_estimate;

    cv::resize(image, resized_image, cv::Size(image.cols * SCALE, image.rows * SCALE));

    int offset_x = (IMAGE_SIZE - resized_image.cols) / 2;
    int offset_y = (IMAGE_SIZE - resized_image.rows) / 2;

    cv::Rect roi(offset_x, offset_y, resized_image.cols, resized_image.rows);

    resized_image.copyTo(letterboxed(roi));

    cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);

    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    cv::split(rgb, channels);

    std::vector<float> nchw_buffer;
    nchw_buffer.reserve(3 * rgb.rows * rgb.cols);

    for (const auto& channel : channels) {
        const float* ptr = (const float*)channel.data;
        nchw_buffer.insert(nchw_buffer.end(), ptr, ptr + channel.total());
    }


    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::array<int64_t, 4> input_shape = {1, 3, IMAGE_SIZE, IMAGE_SIZE};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        nchw_buffer.data(),
        nchw_buffer.size(),
        input_shape.data(),
        input_shape.size()
    );

    std::wstring wpath(modelPath.begin(), modelPath.end());

    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "pose");
    static Ort::Session session(env, wpath.c_str(), Ort::SessionOptions{});

    const char* input_names[] = {"images"};
    const char* output_names[] = {"output0"};

    auto output_tensors = session.Run(
        Ort::RunOptions{},
        input_names, &input_tensor, 1,
        output_names, 1
    );

    float* output_data = output_tensors[0].GetTensorMutableData<float>();

    int best = 0;

    for (int j = 0; j < 8400; j++) {
        if(output_data[4 * 8400 + j] > output_data[4 * 8400 + best]) {
            best = j;
        }
    }

    for (int k = 0;k < 17; k++) {
        int x_index = (k * 3 + 5) * 8400 + best, 
            y_index = (k * 3 + 6) * 8400 + best;
        double keypoint_x = output_data[x_index], keypoint_y = output_data[y_index];

        pose_estimate.keypoints[k].x = (keypoint_x - offset_x) / SCALE;
        pose_estimate.keypoints[k].y = (keypoint_y - offset_y) / SCALE;
        pose_estimate.keypoints[k].confidence = output_data[(k * 3 + 7) * 8400 + best];
    }

    return pose_estimate;
}