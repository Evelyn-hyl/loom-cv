#include "../include/vanishing_point.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>

/**
 * Converts line segment to homogeneous Eigen line vector
 * 
 * @param line A line segment
 * 
 * @return A homogeneous Eigen line vector of the segment
 */
Eigen::Vector3d toHomogeneous(const cv::Vec4i& line) {

    double x1 = line[0], y1 = line[1];
    double x2 = line[2], y2 = line[3];
    Eigen::Vector3d hom(y1 - y2, -(x1 - x2), x1 * y2 - x2 * y1);
    
    return hom;
}

/**
 * Determine if the line segment is an inlier of the given vanishing point 
 * 
 * @param line A line segment
 * @param vp A vanishing point
 * @param angleThresholdDeg The maximum angular deviation (in degrees) 
 * between a line segment's direction and the direction toward the vanishing point
 * 
 * @return true if the segment is within angleThresholdDeg of the vanishing point, false otherwise 
 */
bool isInlier(const cv::Vec4i& line, cv::Point2d vp, double angleThresholdDeg = 5.0) {
    Eigen::Vector2d line_direction(line[2] - line[0], line[3] - line[1]);
    line_direction.normalize();

    cv::Point2d line_midpoint((line[0] + line[2]) / 2.0, (line[1] + line[3]) / 2.0);
    
    Eigen::Vector2d mid_to_vp_direction(vp.x - line_midpoint.x, vp.y - line_midpoint.y);
    mid_to_vp_direction.normalize();

    double angle_deg = std::acos(std::abs(line_direction.dot(mid_to_vp_direction))) * (180.0 / CV_PI);
    
    return angle_deg <= angleThresholdDeg;
}

/**
 * Counts the number of inlier line segments a vanishing point has
 * 
 * @param lines An array of line segments
 * @param vp A vanishing point
 * @param angleThresholdDeg The maximum angular deviation (in degrees) 
 * between a line segment's direction and the direction toward the vanishing point
 * 
 * @return number of inlier line segments of a vanishing point
 */
int countInliers(const std::vector<cv::Vec4i>& lines, cv::Point2d vp, double angleThresholdDeg = 5.0) {
    int count = 0;

    for (const auto& line : lines) {
        if (isInlier(line, vp, angleThresholdDeg)) {
            count++;
        }
    }

    return count;
}

std::vector<cv::Point2d> detectVanishingPoints(const std::vector<cv::Vec4i>& lines) {
    std::mt19937 rng(42);

    std::vector<cv::Point2d> vanishingPoints;
    std::vector<cv::Vec4i> remaining(lines);

    // TODO: Update to add a third vp to detect 3-point perspective compositions
    for (int pass = 0; pass < 2; pass++) {
        
        if (remaining.size() < 2) {
            break;
        }

        const int ITERATIONS = 500, MIN_INLIERS = 10;

        std::uniform_int_distribution<int> distrib(0, remaining.size() - 1);

        cv::Point2d best_vp(0.0, 0.0);
        int best_vp_inlier_count = 0;

        for (int iter = 0; iter < ITERATIONS; iter++) {
            int index1 = distrib(rng), index2 = distrib(rng);

            Eigen::Vector3d line1 = toHomogeneous(remaining[index1]);
            Eigen::Vector3d line2 = toHomogeneous(remaining[index2]);

            Eigen::Vector3d cross_product = line1.cross(line2);

            // Rule out parallel lines; we need intersecting vectors to find a vanishing point 
            if (std::abs(cross_product[2]) < 1e-6) {
                continue;
            }

            // Convert vp homogeneous point (x, y, w) to a 2D point (x/w, y/w)
            cv::Point2d vp(cross_product[0] / cross_product[2], cross_product[1] / cross_product[2]);

            int inliers = countInliers(remaining, vp);
            
            if (inliers > best_vp_inlier_count) {
                best_vp = vp;
                best_vp_inlier_count = inliers;
            }
        }

        // Note: MIN_INLIERS and divided by 20 are arbitrary thresholds
        int min_inlier = std::max(MIN_INLIERS, (int)(remaining.size() / 20));
        
        if (best_vp_inlier_count > min_inlier) {
            vanishingPoints.push_back(best_vp);
        }

        // Cleanup
        remaining.erase(
            std::remove_if(remaining.begin(), remaining.end(), [&](const cv::Vec4i& line) {
                return isInlier(line, best_vp);
            }), remaining.end()
        );
    }

    return vanishingPoints;
}

std::vector<cv::Vec4i> getInlierLines(const std::vector<cv::Vec4i>& lines, cv::Point2d vp, double angleThresholdDeg) {
    std::vector<cv::Vec4i> vpInliers;

    std::copy_if(lines.begin(), lines.end(), std::back_inserter(vpInliers), [&](const cv::Vec4i& line) {
            return isInlier(line, vp);
    });

    return vpInliers;
}