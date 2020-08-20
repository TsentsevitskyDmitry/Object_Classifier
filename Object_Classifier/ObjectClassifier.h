#pragma once
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include "opencv2/imgproc.hpp"
#include "Object.h"

#define CLASSIFIER_DEBUG

class ObjectClassifier
{
private:
	const int	 BLUR_KERNEL_SIZE = 3;
	const int	 IN_RANGE_LOWB = 100;
	const int	 IN_RANGE_UPPB = 255;
	const int	 MORPH_SIZE = 11;
	const int	 MORPH_SIZE_MAX = 3;
	const int	 MORPH_TYPE = cv::MORPH_ELLIPSE;
	const double CANNY_TRESHOLD = 200;
	const int	 REDUCE_NOISE_MIN_LEN = 200;
	const int	 OBJECT_ID_EMPTY = -1;
	const double TM_MATCH_TRESHOLD = 0.85;

	void reduceNoise(std::vector<std::vector<cv::Point>>& contours, const int minLen = 100);
	void sortContours(std::vector<std::vector<cv::Point>>& contours);
	void match(const std::vector<cv::Mat>& normalized_objects, std::vector<Object>& objects);
	void hconcatMatrix(const std::vector<cv::Mat>& src, const std::vector<int>& indexes, cv::Mat& dst);
	int  hfindMatrixIndexByPosition(const std::vector<cv::Mat>& src, const std::vector<int>& indexes, const cv::Point& position);

public:
	std::vector<Object> process(const cv::Mat& image);
};

