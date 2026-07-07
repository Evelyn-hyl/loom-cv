// silhouette_extractor.hpp
#pragma once
#include <opencv2/core.hpp>
#include <onnxruntime_cxx_api.h>

/**
 * Extracts the silhouette of one human figure in an image.
 * Uses U2-Net; returns the largest connected region / blob.
 * 
 * @param croppedImage An image cropped around the human
 * @param session The inference session
 * 
 * @return A binary mask of the human figure's silhouette
 */
cv::Mat extractSingleSilhouette(const cv::Mat& croppedImage, Ort::Session& session);