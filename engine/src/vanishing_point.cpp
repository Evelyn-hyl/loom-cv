#include "../include/vanishing_point.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <algorithm>

/**
 * Converts Vec4i segment to homogeneous Eigen line vector
 * 
 * @param seg
 * 
 * @return an Eigen line vector of the segment
 */
Eigen::Vector3d toLine(const cv::Vec4i& seg) {

    double x1 = seg[0], y1 = seg[1];
    double x2 = seg[2], y2 = seg[3];
    Eigen::Vector3d line(y1 - y2, -(x1 - x2), x1 * y2 - x2 * y1);
    
    return line;
}

/**
 * Determine if the segment is an inlier of the given vanishing point 
 * 
 * @param seg
 * @param vp
 * @param angleThresholdDeg
 * 
 * @return bool
 */
bool isInlier(const cv::Vec4i& seg, cv::Point2d vp, double angleThresholdDeg) {
    Eigen::Vector2d seg_vector(seg[2] - seg[0], seg[3] - seg[1]);
    seg_vector.normalize();

    cv::Point2d seg_midpoint((seg[0] + seg[2]) / 2.0, (seg[1] + seg[3]) / 2.0);
    
    Eigen::Vector2d mid_to_vp_vector(vp.x - seg_midpoint.x, vp.y - seg_midpoint.y);
    mid_to_vp_vector.normalize();

    double angle_deg = std::acos(std::abs(seg_vector.dot(mid_to_vp_vector))) * (180.0 / CV_PI);
    
    return angle_deg <= angleThresholdDeg;
}

/**
 * Counts the number of inlier segments a vanishing point has
 * 
 * @param segs
 * @param vp
 * @param angleThresholdDeg
 * 
 * @return number of inlier segments of a vanishing point
 */
int countInliers(const std::vector<cv::Vec4i>& segs, cv::Point2d vp, double angleThresholdDeg) {
    int count = 0;

    for (auto seg : segs) {
        if (isInlier(seg, vp, angleThresholdDeg)) {
            count++;
        }
    }

    return count;
}

std::vector<cv::Point2d> detectVanishingPoints(const std::vector<cv::Vec4i>& lines) {
    std::mt19937 rng(42);

    const double ANGLE_THRESHOLD_DEG = 5.0;

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

            Eigen::Vector3d line1 = toLine(remaining[index1]);
            Eigen::Vector3d line2 = toLine(remaining[index2]);

            Eigen::Vector3d cross_product = line1.cross(line2);

            // Rule out parallel lines; we need intersecting vectors to find a vanishing point 
            if (std::abs(cross_product[2]) < 1e-6) {
                continue;
            }

            // Convert vp homogeneous point (x, y, w) to a 2D point (x/w, y/w)
            cv::Point2d vp(cross_product[0] / cross_product[2], cross_product[1] / cross_product[2]);

            int inliers = countInliers(remaining, vp, ANGLE_THRESHOLD_DEG);
            
            if (inliers > best_vp_inlier_count) {
                best_vp = vp;
                best_vp_inlier_count = inliers;
            }
        }

        // Note: MIN_INLIERS and divided by 4 are arbitrary thresholds
        int min_inlier = std::max(MIN_INLIERS, (int)(remaining.size() / 4));
        
        if (best_vp_inlier_count > min_inlier) {
            vanishingPoints.push_back(best_vp);
        }

        // Cleanup
        remaining.erase(
            std::remove_if(remaining.begin(), remaining.end(), [&](const cv::Vec4i& seg) {
                return isInlier(seg, best_vp, ANGLE_THRESHOLD_DEG);
            }), remaining.end()
        );
    }

    return vanishingPoints;
}