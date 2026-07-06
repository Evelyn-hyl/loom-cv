#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Core>
#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>
#include "../include/edge_detection.hpp"
#include "../include/vanishing_point.hpp"
#include "../include/pose_estimator.hpp"
#include "../include/body_segmentor.hpp"
#include "../include/silhouette_extractor.hpp"

int main(int argc, char* argv[]) {

    if (argc < 5) {
        return -1;
    }

    cv::Mat image = cv::imread(argv[1]);
    std::string pose_model_path = argv[2];
    std::string segment_model_path = argv[3];
    std::string silhouette_model_path = argv[4];

    if (image.empty()) {
        std::cerr << "Could not open or find the image!" << "\n";
        return -1;
    }

    if (pose_model_path.empty()) {
        std::cerr << "Could not find path to Yolov8n model!" << "\n";
        return -1;
    }

    if (segment_model_path.empty()) {
        std::cerr << "Could not find path to Sapiens Seg model!" << "\n";
        return -1;
    }

    if (silhouette_model_path.empty()) {
        std::cerr << "Could not find path to U2-Net model!" << "\n";
        return -1;
    }

    std::wstring pose_wpath(pose_model_path.begin(), pose_model_path.end());
    std::wstring segment_wpath(segment_model_path.begin(), segment_model_path.end());
    std::wstring silhouette_wpath(silhouette_model_path.begin(), silhouette_model_path.end());

    cv::Mat result = image.clone();

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "loomcv");

    Ort::Session pose_session(env, pose_wpath.c_str(), Ort::SessionOptions{});

    Ort::SessionOptions seg_options;
    seg_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    Ort::Session seg_session(env, segment_wpath.c_str(), seg_options);

    Ort::Session sil_session(env, silhouette_wpath.c_str(), Ort::SessionOptions{});

    // --- Perspective Analysis ---
    std::vector<cv::Vec4i> lines = detectLines(image);

    lines = filterDiagonalLines(lines);

    std::vector<cv::Point2d> vanishing_points = detectVanishingPoints(lines);

    if (vanishing_points.empty()) {
        std::cerr << "No vanishing points detected in the image.";
        return -1;
    }

    std::vector<cv::Vec4i> inlier_lines;
    nlohmann::json perspective_analysis;
    perspective_analysis["vanishingPoints"] = nlohmann::json::array();

    for (auto vp : vanishing_points) {
        nlohmann::json vanishing_point;
        vanishing_point["x"] = vp.x;
        vanishing_point["y"] = vp.y;
        vanishing_point["inliers"] = nlohmann::json::array();

        cv::Scalar color(0, 255, 0);

        cv::circle(result, cv::Point(vp), 10, color, -1);

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

    // --- Pose Analysis ---
    PoseEstimate pose;

    try {
        pose = detectSinglePose(image, pose_session);
    } catch (const std::exception& e) {
        std::cerr << "Pose error: " << e.what() << "\n";
        return -1;
    }

    nlohmann::json pose_estimate;

    for (int i = 0; i < 17; i++) {
        pose_estimate[keypoint_keys[i]] = {
            {"x", pose.keypoints[i].x},
            {"y", pose.keypoints[i].y},
            {"confidence", pose.keypoints[i].confidence}
        };
    }

    for (const auto& [a, b] : skeleton) {
        cv::line(result,
            cv::Point(pose.keypoints[a].x, pose.keypoints[a].y),
            cv::Point(pose.keypoints[b].x, pose.keypoints[b].y),
            cv::Scalar(0, 255, 255), 2);
    }

    nlohmann::json output;
    output["perspectiveAnalysis"] = perspective_analysis;
    output["poseEstimate"] = pose_estimate;

    std::cout << output.dump() << "\n";

    int left = std::max(0, (int)(pose.figure_bounding_box.x - pose.figure_bounding_box.width / 2));
    int top = std::max(0, (int)(pose.figure_bounding_box.y - pose.figure_bounding_box.height / 2));
    int width = std::min((int)pose.figure_bounding_box.width, image.cols - left);
    int height = std::min((int)pose.figure_bounding_box.height, image.rows - top);
    cv::Rect crop_rect(left, top, width, height);
    cv::Mat person_crop = image(crop_rect).clone();

    cv::imwrite("debug_crop.jpg", person_crop);

    cv::Mat silhouette_mask;

    try {
        silhouette_mask = extractSingleSilhouette(person_crop, sil_session);
    } catch (const std::exception& e) {
        std::cerr << "Silhouette extraction error: " << e.what() << "\n";
        return -1;
    }

    cv::imwrite("debug_silhouette.jpg", silhouette_mask);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(20, 20));
    cv::morphologyEx(silhouette_mask, silhouette_mask, cv::MORPH_DILATE, kernel);

    cv::Mat debug_dilated;
    person_crop.copyTo(debug_dilated, silhouette_mask);

    cv::imwrite("debug_silhouette_dilated.jpg", debug_dilated);

    try {
        detectBodySegments(person_crop, seg_session, silhouette_mask);
    } catch (const std::exception& e) {
        std::cerr << "Segment error: " << e.what() << "\n";
        return -1;
    }

    if (!cv::imwrite("output.jpg", result)) {
        std::cerr << "An error occurred.";
        return -1;
    }

    return 0;
}
