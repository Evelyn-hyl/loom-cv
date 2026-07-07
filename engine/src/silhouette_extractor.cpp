#include "../include/silhouette_extractor.hpp"
#include <string>
#include <cstdint>
#include <array>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <ranges>
#include <iterator>
#include <span>

cv::Mat extractSingleSilhouette(const cv::Mat& croppedImage, Ort::Session& session) {
    const int IMAGE_SIZE = 320;

    cv::Mat resized_image;
    cv::resize(croppedImage, resized_image, cv::Size(IMAGE_SIZE, IMAGE_SIZE));

    cv::Mat rgb;
    cv::cvtColor(resized_image, rgb, cv::COLOR_BGR2RGB);
    
    rgb.convertTo(rgb, CV_32F);

    std::vector<cv::Mat> channels;
    cv::split(rgb, channels);

    std::vector<float> nchw_buffer;
    nchw_buffer.reserve(3 * rgb.rows * rgb.cols);

    // Values in R, G, B order
    const std::vector<float> MEAN = {0.485, 0.456, 0.406};
    const std::vector<float> STD = {0.229, 0.224, 0.225};

    for (size_t i = 0; i < channels.size(); ++i) {
        const cv::Mat& current_mat = channels[i];

        if (current_mat.isContinuous()) {
            std::span<const float> float_span(current_mat.ptr<float>(), current_mat.total());

            auto normalized = float_span | std::views::transform([&](float pix){
                return (pix / 255.0 - MEAN[i]) / STD[i];
            });

            std::ranges::copy(normalized, std::back_inserter(nchw_buffer));
        } else {
            // Fallback row-by-row normalization loop
            for (int r = 0; r < current_mat.rows; ++r) {
                std::span<const float> row_span(current_mat.ptr<float>(r), current_mat.cols);
                
                auto processed_row = row_span | std::views::transform([&](float pix) {
                    return (pix / 255.0 - MEAN[i]) / STD[i];
                });
                
                std::ranges::copy(processed_row, std::back_inserter(nchw_buffer));
            }
        }
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

    const char* input_names[] = {"input.1"};
    const char* output_names[] = {"1959"};
    
    auto output_tensors = session.Run(
        Ort::RunOptions{},
        input_names, &input_tensor, 1,
        output_names, 1
    );

    float* output_data = output_tensors[0].GetTensorMutableData<float>();

    cv::Mat raw_mat(IMAGE_SIZE, IMAGE_SIZE, CV_32FC1, output_data);

    cv::Mat debug_alpha, debug_alpha_resized;
    raw_mat.convertTo(debug_alpha, CV_8UC1, 255.0);
    cv::resize(debug_alpha, debug_alpha_resized, cv::Size(croppedImage.cols, croppedImage.rows));
    
    std::vector<cv::Mat> bgra_channels;
    cv::split(croppedImage, bgra_channels);

    bgra_channels.push_back(debug_alpha_resized);

    // [ Debug ]
    cv::Mat debug_rgba;
    cv::merge(bgra_channels, debug_rgba);

    cv::imwrite("debug_silhouette_rgba.png", debug_rgba);
    // [ Debug ]

    cv::Mat mat_8u;
    raw_mat.convertTo(mat_8u, CV_8UC1, 255.0);

    cv::Mat binary_mask;
    cv::threshold(mat_8u, binary_mask, 3, 255, cv::THRESH_BINARY);

    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(binary_mask, labels, stats, centroids);

    int largest_label = 1, largest_area = 0;
    for (int i = 1; i < num_labels; i++) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > largest_area) {
            largest_area = area;
            largest_label = i;
        }
    }

    cv::Mat filtered_mask = (labels == largest_label);

    cv::Mat resized_binary_mask;
    cv::resize(filtered_mask, resized_binary_mask, cv::Size(croppedImage.cols, croppedImage.rows));

    return resized_binary_mask;
}