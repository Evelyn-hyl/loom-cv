#include "../include/edge_detection.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>

std::vector<cv::Vec4i> detectLines(const cv::Mat& image) {
    if (image.empty()) {
        std::cerr << "Could not open or find the image!" << "\n";
        return {};
    }

    cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_STD, 0.8, 0.6, 2.0, 30.0, 0.0, 0.5);
    cv::Mat gray;
    std::vector<cv::Vec4f> lines_f;
    std::vector<cv::Vec4i> lines_i;

    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    lsd->detect(gray, lines_f);
    
    for (auto line : lines_f) {
        lines_i.push_back(cv::Vec4i(
            (int)line[0], (int)line[1], (int)line[2], (int)line[3]
        ));
    }
    return lines_i;
}

std::vector<cv::Vec4i> filterDiagonalLines(const std::vector<cv::Vec4i>& lines, double angleThresholdDeg) {
    std::vector<cv::Vec4i> diagonal_lines;

    for (const auto& line : lines) {
        double angle_deg = std::abs(
            std::atan2((line[3] - line[1]), (line[2] - line[0])) * 180 / CV_PI
        );

        if (angle_deg > angleThresholdDeg && angle_deg < 90 - angleThresholdDeg || 
            angle_deg > 90 + angleThresholdDeg && angle_deg < 180 - angleThresholdDeg
        ) {
            diagonal_lines.push_back(line);
        }
    }

    return diagonal_lines;
}