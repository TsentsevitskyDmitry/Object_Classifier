#pragma once
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include "Object.h"

#define CLASSIFIER_DEBUG

class ObjectClassifier
{
private:
	const double CANNY_TRESHOLD = 200;
	void reduceNoise(std::vector<std::vector<cv::Point>>& contours);

public:
	std::vector<Object> process(cv::Mat image);
};

