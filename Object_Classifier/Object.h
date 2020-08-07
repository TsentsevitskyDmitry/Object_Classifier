#pragma once
#include <opencv2/core/types.hpp>
#include <vector>

struct Object
{
	int id;
	cv::Rect roi;
	std::vector<cv::Point> contour;
};

