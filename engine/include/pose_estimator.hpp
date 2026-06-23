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

struct PoseEstimate {
    std::array<Keypoint, 17> keypoints;
};

PoseEstimate detectSinglePose(const cv::Mat& image, const std::string& modelPath);