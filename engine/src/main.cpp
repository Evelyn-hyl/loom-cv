#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Core>
#include <nlohmann/json.hpp>
#include "../include/edge_detection.hpp"
#include "../include/vanishing_point.hpp"

int main(int argc, char* argv[]) {

    if (argc < 2) {
        return -1;
    }

    cv::Mat image;
    cv::Mat result;
    std::vector<cv::Vec4i> lines;
    std::vector<cv::Point2d> vanishing_points;
    std::vector<cv::Vec4i> inlier_lines;

    nlohmann::json perspective_analysis;
    perspective_analysis["vanishingPoints"] = nlohmann::json::array();
    
    image = cv::imread(argv[1]);

    if (image.empty()) {
        std::cerr << "Could not open or find the image!" << "\n";
        return -1;
    }

    result = image.clone();

    lines = detectLines(image);

    lines = filterDiagonalLines(lines);

    vanishing_points = detectVanishingPoints(lines);

    if (vanishing_points.empty()) {
        std::cerr << "No vanishing points detected in the image.";
        return -1;
    }

    for (auto vp : vanishing_points) {
        nlohmann::json vanishing_point;
        vanishing_point["x"] = vp.x;
        vanishing_point["y"] = vp.y;
        vanishing_point["inliers"] = nlohmann::json::array();

        cv::Scalar color(0, 255, 0);

        cv::circle(result, cv::Point(vp), 20, color, -1);

        inlier_lines = getInlierLines(lines, vp);

        for (auto line : inlier_lines) {
            cv::line(result, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 0, 255));

            nlohmann::json segment;
            segment["x1"] = line[0];
            segment["y1"] = line[1];
            segment["x2"] = line[2];
            segment["y2"] = line[3];
            segment["length"] = std::sqrt(
                (line[2] - line[0]) * (line[2] - line[0]) + (line[3] - line[1]) * (line[3] - line[1])
            );
           
            vanishing_point["inliers"].push_back(segment);
        }

        perspective_analysis["vanishingPoints"].push_back(vanishing_point);
    }

    std::cout << perspective_analysis.dump() << "\n";

    if (!cv::imwrite("output.jpg", result)) {
        std::cerr << "An error occurred.";
        return -1;
    }

    return 0;
}
