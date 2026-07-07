// body_segmentor.hpp
#pragma once
#include <array>
#include <opencv2/core.hpp>
#include <onnxruntime_cxx_api.h>

enum BodySegmentIndex {
    background,
    apparel,
    face_neck, 
    hair,
    left_foot,
    left_hand,
    left_lower_arm,
    left_lower_leg,
    left_shoe,
    left_sock,
    left_upper_arm,
    left_upper_leg,
    lower_clothing,
    right_foot,
    right_hand,
    right_lower_arm,
    right_lower_leg,
    right_shoe,
    right_sock,
    right_upper_arm,
    right_upper_leg,
    torso,
    upper_clothing,
    lower_lip,
    upper_lip,
    lower_teeth,
    upper_teeth,
    tongue,
};

struct BodyPartContour {
    std::vector<cv::Point2d> points;
};
struct BodySegmentation {
    std::array<BodyPartContour, 28> parts;
};

/**
 * Detects body part segments of one human figure in the image.
 * Uses Sapiens body part segmentation model; classifies each pixel as one of 28 body part categories.
 * 
 * @param image An image containing a human figure
 * @param session The inference session
 * @param silhouette A binary mask of the human figure's silhouette
 * 
 * @return The data of each detected body part segment
 */
BodySegmentation detectBodySegments(const cv::Mat& image, Ort::Session& session, cv::Mat silhouette);