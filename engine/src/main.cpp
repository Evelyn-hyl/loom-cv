#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Core>
#include <nlohmann/json.hpp>
#include "../include/edge_detection.hpp"
#include "../include/vanishing_point.hpp"

int main(int argc, char* argv[]) {
    cv::Mat image;
    cv::Mat result;
    std::vector<cv::Vec4i> lines;
    std::vector<cv::Point2d> vanishing_points;
    
    image = cv::imread(argv[1]);

    if (image.empty()) {
        std::cout << "Could not open or find the image!" << "\n";
        return -1;
    }

    result = image.clone();
    lines = detectLines(image);
    
    for (auto line : lines) {
        cv::line(result, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 0, 255));
    }

    vanishing_points = detectVanishingPoints(lines);

    std::cout << "Vanishing points: " << vanishing_points << "\n";

    if (!cv::imwrite("output.jpg", result)) {
        std::cout << "An error occurred.";
        return -1;
    }

    return 0;
}
