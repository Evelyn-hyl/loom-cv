// edge_detection.hpp
#pragma once
#include <opencv2/core.hpp>
#include <vector>

std::vector<cv::Vec4i> detectLines(
    const cv::Mat& image, 
    double lowThreshold = 40, 
    double highThreshold = 100,
    int isALineThreshold = 150,
    double minLineLength = 150,
    double maxLineGap = 10
);