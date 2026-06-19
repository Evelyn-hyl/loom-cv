// vanishing_point.hpp
#pragma once
#include <opencv2/core.hpp>
#include <vector>

/**
 * @brief RANSAC algorithm identifies valid perspective lines to detect vanishing point(s)
 * 
 * The RANSAC algorithm repeats the following steps:
 * 1. Finds two random lines and calculate their cross product
 * 2. Convert the cross product to a point (aka vanishing point)
 * 3. Count inlier lines that point to the vanishing point
 * 4. Get the best vanishing point, which has the most inlier lines
 * 5. Repeat for a second vanishing point
 * 
 * @param lines
 * 
 * @return An array of vanishing points
 */
std::vector<cv::Point2d> detectVanishingPoints(const std::vector<cv::Vec4i>& lines);