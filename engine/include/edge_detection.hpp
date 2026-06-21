// edge_detection.hpp
#pragma once
#include <opencv2/core.hpp>
#include <vector>

/**
 * Detects all line segments in the image
 * 
 * @param image A BGR color image
 * 
 * @return An array of line segments
 */
std::vector<cv::Vec4i> detectLines(const cv::Mat& image);

/**
 * Filters out vertical/horizontal line segments
 * @note May be removed with the introduction of 3-point perspective analysis
 * 
 * @param lines An array of line segments
 * @param angleThresholdDeg An angle measured relative to 0°, 90°, and 180°; 
 * when angle is exactly at the threshold, it is considered a vertical/horizontal line and filtered out
 * 
 * @return An array of non-vertical/horizontal line segments
 */
std::vector<cv::Vec4i> filterDiagonalLines(const std::vector<cv::Vec4i>& lines, double angleThresholdDeg = 5.0);