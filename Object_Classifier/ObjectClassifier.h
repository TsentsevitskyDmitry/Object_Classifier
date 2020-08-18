#pragma once
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include "Object.h"

#define CLASSIFIER_DEBUG

class ObjectClassifier
{
private:
	const double CANNY_TRESHOLD = 200;
	const double ADAPTIVE_TRESHOLD_C = 32;
	const double ADAPTIVE_TRESHOLD_MAX_VALUE = 255;
	const int	 ADAPTIVE_TRESHOLD_BLOCK_SIZE = 1025;
	const int	 REDUCE_NOISE_MIN_LEN = 100;
	const int	 BLUR_KERNEL_SIZE = 3;
	const int	 OBJECT_ID_EMPTY = -1;
	const double TM_MATCH_TRESHOLD = 0.4;

	void reduceNoise(std::vector<std::vector<cv::Point>>& contours, const int minLen = 100);
	void sortContours(std::vector<std::vector<cv::Point>>& contours);
	void match(const std::vector<cv::Mat>& normalized_objects, std::vector<Object>& objects);
	void hconcatMatrix(const std::vector<cv::Mat>& src, const std::vector<int>& indexes, cv::Mat& dst);
	int  hfindMatrixIndexByPosition(const std::vector<cv::Mat>& src, const std::vector<int>& indexes, const cv::Point& position);

public:
	std::vector<Object> process(const cv::Mat& image);
};

