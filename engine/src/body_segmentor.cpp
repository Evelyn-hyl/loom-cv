#include "../include/body_segmentor.hpp"
#include <iostream>
#include <string>
#include <array>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <set>
#include <cmath>
#include <algorithm>
#include <ranges>
#include <iterator>
#include <span>

BodySegmentation detectBodySegments(const cv::Mat& image, Ort::Session& session, cv::Mat silhouette) {
    const int INPUT_HEIGHT = 1024, INPUT_WIDTH = 768;
    const int OUTPUT_HEIGHT = 512, OUTPUT_WIDTH = 384;
    const double SCALE = std::min((double)INPUT_WIDTH / image.cols, (double)INPUT_HEIGHT / image.rows);

    BodySegmentation body_segment;

    std::vector<cv::Mat> rgb_channels(3);
    cv::Mat letterbox(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));

    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(image.cols * SCALE, image.rows * SCALE));

    int offset_x = (INPUT_WIDTH - resized_image.cols) / 2;
    int offset_y = (INPUT_HEIGHT - resized_image.rows) / 2;

    cv::Rect roi(offset_x, offset_y, resized_image.cols, resized_image.rows);

    resized_image.copyTo(letterbox(roi));

    cv::Mat rgb;
    cv::cvtColor(letterbox, rgb, cv::COLOR_BGR2RGB);

    rgb.convertTo(rgb, CV_32F);

    cv::split(rgb, rgb_channels);
    
    std::vector<float> nchw_buffer;
    nchw_buffer.reserve(3 * rgb.rows * rgb.cols);

    const std::vector<float> MEAN = {123.5, 116.5, 103.5};
    const std::vector<float> STD = {58.5,  57.0,  57.5};

    for (size_t i = 0; i < rgb_channels.size(); ++i) {
        const cv::Mat& current_mat = rgb_channels[i];
        
        if (current_mat.isContinuous()) {    
            std::span<const float> float_span(current_mat.ptr<float>(), current_mat.total());

            auto normalized = float_span | std::views::transform([&](float pix){
                return (pix - MEAN[i]) / STD[i];
            });

            std::ranges::copy(normalized, std::back_inserter(nchw_buffer));
        } else {
            // Fallback row-by-row normalization loop
            for (int r = 0; r < current_mat.rows; ++r) {
                std::span<const float> row_span(current_mat.ptr<float>(r), current_mat.cols);
                
                auto processed_row = row_span | std::views::transform([&](float pix) {
                    return (pix - MEAN[i]) / STD[i];
                });
                
                std::ranges::copy(processed_row, std::back_inserter(nchw_buffer));
            }
        }
    }

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::array<int64_t, 4> input_shape = {1, 3, INPUT_HEIGHT, INPUT_WIDTH};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        nchw_buffer.data(),
        nchw_buffer.size(),
        input_shape.data(),
        input_shape.size()
    );

    const char* input_names[] = {"pixel_values"};
    const char* output_names[] = {"logits"};

    auto output_tensors = session.Run(
        Ort::RunOptions{},
        input_names, &input_tensor, 1,
        output_names, 1
    );

    // Output is a 3D block [28, 512, 384]
    float* output_data = output_tensors[0].GetTensorMutableData<float>();

    cv::Mat mask(OUTPUT_HEIGHT, OUTPUT_WIDTH, CV_8UC1);

    for (int y = 0; y < OUTPUT_HEIGHT; y++) {
        for (int x = 0; x < OUTPUT_WIDTH; x++) {
            int best_class = 0;
            float best_score = output_data[y * OUTPUT_WIDTH + x];

            for (int c = 1; c < 28; c++) {
                float score = output_data[c * OUTPUT_HEIGHT * OUTPUT_WIDTH + y * OUTPUT_WIDTH + x];
                
                if (score > best_score) {
                    best_class = c;
                    best_score = score;
                };
            }

            mask.at<uchar>(y, x) = best_class;
        }
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

    cv::Mat resized_silhouette;
    cv::resize(silhouette, resized_silhouette, cv::Size(resized_image.cols / 2, resized_image.rows / 2));

    cv::Mat silhouette_letterbox(OUTPUT_HEIGHT, OUTPUT_WIDTH, CV_8UC1, cv::Scalar(0, 0, 0));
    cv::Rect silhouette_roi(offset_x / 2, offset_y / 2, resized_silhouette.cols, resized_silhouette.rows);

    resized_silhouette.copyTo(silhouette_letterbox(silhouette_roi));

    mask.setTo(0, silhouette_letterbox == 0);

    // [ Debug ]
    cv::Mat color_mask(mask.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    static const cv::Vec3b COLORS[28] = {
        {0,0,0},       // background
        {128,0,0},     // apparel
        {255,178,102}, // face_neck
        {102,0,204},   // hair
        {0,204,102},   // left_foot
        {255,51,51},   // left_hand
        {255,153,51},  // left_lower_arm
        {255,255,51},  // left_lower_leg
        {51,255,51},   // left_shoe
        {51,255,255},  // left_sock
        {51,153,255},  // left_upper_arm
        {153,51,255},  // left_upper_leg
        {255,51,255},  // lower_clothing
        {0,102,204},   // right_foot
        {204,0,0},     // right_hand
        {204,102,0},   // right_lower_arm
        {204,204,0},   // right_lower_leg
        {0,204,0},     // right_shoe
        {0,204,204},   // right_sock
        {0,0,204},     // right_upper_arm
        {102,0,102},   // right_upper_leg
        {204,102,204}, // torso
        {102,204,204}, // upper_clothing
        {255,102,178}, // lower_lip
        {255,51,153},  // upper_lip
        {204,255,153}, // lower_teeth
        {153,255,204}, // upper_teeth
        {255,204,153}, // tongue
    };
    for (int y = 0; y < mask.rows; y++)
        for (int x = 0; x < mask.cols; x++)
            color_mask.at<cv::Vec3b>(y, x) = COLORS[mask.at<uchar>(y, x)];
    cv::imwrite("debug_segmentation.jpg", color_mask);
    // [ Debug ]

    return body_segment;
}