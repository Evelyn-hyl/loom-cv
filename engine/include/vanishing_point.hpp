// vanishing_point.hpp
#pragma once
#include <opencv2/core.hpp>
#include <vector>

/**
 * Detects up to two vanishing points from a set of line segments using RANSAC.
 * 
 * @param lines An array of line segments
 * 
 * @return Vanishing points sorted by inlier count
 */
std::vector<cv::Point2d> detectVanishingPoints(const std::vector<cv::Vec4i>& lines);

/**
 * Retrieves all lines that align with the given vanishing point
 * 
 * @param lines An array of line segments
 * @param vp A vanishing point
 * @param angleThresholdDeg The maximum angular deviation (in degrees) 
 * between a segment's direction and the direction toward the vanishing point
 * 
 * @return An array of inlier line segments
 */
std::vector<cv::Vec4i> getInlierLines(const std::vector<cv::Vec4i>& lines, cv::Point2d vp, double angleThresholdDeg = 10.0);