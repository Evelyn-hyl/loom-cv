#include "../include/edge_detection.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>

/**
 * @brief [brief description]
 * 
 * [description]
 * 
 * @param image
 * @param lowThreshold
 * @param highThreshold
 * @param isALineThreshold
 * @param minLineLength
 * @param maxLineGap
 * 
 * 
 * @return an array of line segments
 */
std::vector<cv::Vec4i> detectLines(
    const cv::Mat& image, 
    double lowThreshold, 
    double highThreshold, 
    int isALineThreshold, 
    double minLineLength, 
    double maxLineGap) {
    if (image.empty()) {
        std::cout << "Could not open or find the image!" << "\n";
        return {};
    }

    const double RHO = 1.0;
    const double THETA = CV_PI/180;

    cv::Mat gray, blurred, edges;
    std::vector<cv::Vec4i> lines;

    // Stage #1: Canny — Make edges findable
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    cv::GaussianBlur(gray, blurred, cv::Size(5,5), 0);

    cv::Canny(blurred, edges, lowThreshold, highThreshold);

    // Stage #2: Hough — Extract array of line segments
    cv::HoughLinesP(edges, lines, RHO, THETA, isALineThreshold, minLineLength, maxLineGap);

    return lines;
}