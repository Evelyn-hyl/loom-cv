// pose_estimator.hpp
#pragma once
#include <array>
#include <opencv2/core.hpp>

struct Keypoint {
    double x;
    double y;
    double confidence;
};

enum PoseEstimateIndex {
    nose,
    left_eye,
    right_eye,
    left_ear,
    right_ear,
    left_shoulder,
    right_shoulder,
    left_elbow,
    right_elbow,
    left_wrist,
    right_wrist,
    left_hip,
    right_hip, 
    left_knee,
    right_knee,
    left_ankle,
    right_ankle,
};

inline constexpr std::array<const char*, 17> keypoint_keys = {
    "nose",
    "leftEye",
    "rightEye",
    "leftEar",
    "rightEar",
    "leftShoulder",
    "rightShoulder",
    "leftElbow",
    "rightElbow",
    "leftWrist",
    "rightWrist",
    "leftHip",
    "rightHip", 
    "leftKnee",
    "rightKnee",
    "leftAnkle",
    "rightAnkle",
};

inline const std::vector<std::pair<int, int>> skeleton = {
    {nose, left_eye}, {nose, right_eye},
    {left_eye, left_ear}, {right_eye, right_ear},
    {left_shoulder, right_shoulder},
    {left_shoulder, left_elbow}, {left_elbow, left_wrist},
    {right_shoulder, right_elbow}, {right_elbow, right_wrist},
    {left_shoulder, left_hip}, {right_shoulder, right_hip},
    {left_hip, right_hip},
    {left_hip, left_knee}, {left_knee, left_ankle},
    {right_hip, right_knee}, {right_knee, right_ankle},
};

struct PoseEstimate {
    std::array<Keypoint, 17> keypoints;
};

/**
 * Detects the 17-joint skeleton of one human figure in an image.
 * Uses YOLOv8s-pose; returns the highest-confidence detection.
 * 
 * @param image A BGR color image
 * @param modelPath File path to the YOLOv8s-pose ONNX model
 * 
 * @return A PoseEstimate object with 17 keypoints in original image coordinates
 */
PoseEstimate detectSinglePose(const cv::Mat& image, const std::string& modelPath);