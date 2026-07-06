// silhouette_extractor.hpp
#pragma once
#include <opencv2/core.hpp>
#include <onnxruntime_cxx_api.h>

cv::Mat extractSingleSilhouette(const cv::Mat& croppedImage, Ort::Session& session);