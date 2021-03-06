#include "ObjectClassifier.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <algorithm>
#include <opencv2/photo.hpp>

using namespace cv;
using namespace std;

extern int all, bll;

std::vector<Object> ObjectClassifier::process(const cv::Mat& image)
{
    std::vector<Object> objects;
    std::vector<Mat> normalized_objects;
    Mat src_bin, canny_output,
        drawing = Mat::zeros(image.size(), CV_8UC3);

    inRange(image, Scalar::all(IN_RANGE_LOWB), Scalar::all(IN_RANGE_UPPB), src_bin);

    bitwise_not(src_bin, src_bin);

    Mat kernel = getStructuringElement(MORPH_TYPE,
        Size(2 * MORPH_SIZE + 1, 2 * MORPH_SIZE + 1));

    dilate(src_bin, src_bin, kernel);
    erode(src_bin, src_bin, kernel);

    for (int r = 1; r <= MORPH_SIZE_MAX; r++)
    {
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(2 * r + 1, 2 * r + 1));
        morphologyEx(src_bin, src_bin, MORPH_CLOSE, kernel);
        morphologyEx(src_bin, src_bin, MORPH_OPEN, kernel);
    }

    Canny(src_bin, canny_output, CANNY_TRESHOLD, CANNY_TRESHOLD * 2);

    vector<Vec4i> hierarchy;
    vector<vector<Point> > contours;
    findContours(canny_output, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    reduceNoise(contours, REDUCE_NOISE_MIN_LEN);
    sortContours(contours);

    for (size_t i = 0; i < contours.size(); ++i)
    {
        Object object;

        Vec4f vertical_line;
        fitLine(contours[i], vertical_line, DIST_L2, 0, 0.01, 0.01);
        float vx = vertical_line[0];
        float vy = vertical_line[1];
        object.id = OBJECT_ID_EMPTY;
        object.contour = contours[i];
        object.roi = boundingRect(object.contour);

        Mat image_mask = Mat::zeros(image.size(), CV_8UC1);
        Mat image_contour_mask = Mat::zeros(image.size(), CV_8UC1);

        drawContours(image_mask, contours, i, Scalar::all(255), FILLED);
        //drawContours(image_contour_mask, contours, i, Scalar::all(255), 1);
        subtract(image_mask, image_contour_mask, image_mask);
        Mat object_image, object_background_mask;
        image(object.roi).copyTo(object_image, image_mask(object.roi));
        image_mask(object.roi).copyTo(object_background_mask);

        // angle between vertical and fitLine result
        double angle = -atanf(vx / vy) / CV_PI * 180.0;
        Point2f center(object_image.cols / 2.0, object_image.rows / 2.0);
        Mat rot = getRotationMatrix2D(center, angle, 1.0);
        Rect bbox = RotatedRect(center, object_image.size(), angle).boundingRect();
        rot.at<double>(0, 2) += bbox.width / 2.0 - center.x;
        rot.at<double>(1, 2) += bbox.height / 2.0 - center.y;

        Mat object_rotated, object_background_mask_rotated;
        warpAffine(object_image, object_rotated, rot, bbox.size());
        warpAffine(object_background_mask, object_background_mask_rotated, rot, bbox.size());
        bitwise_not(object_background_mask_rotated, object_background_mask_rotated);

        Mat bounding_object;
        cvtColor(object_rotated, bounding_object, COLOR_BGR2GRAY);
        Rect crop_roi = boundingRect(bounding_object);

        Mat object_background_color_mask, object_background_color;
        cvtColor(object_background_mask_rotated, object_background_color_mask, COLOR_GRAY2BGR);
        object_background_color.create(object_background_color_mask.size(), CV_8UC3);
        object_background_color = Scalar::all(BACKGROUND_COLOR);
        bitwise_and(object_background_color, object_background_color_mask, object_background_color);

        Mat normalized_object, normalized_background;
        if (crop_roi.area()) {
            object_rotated(crop_roi).copyTo(normalized_object);
            object_background_color(crop_roi).copyTo(normalized_background);
        }
        else {
            object_rotated.copyTo(normalized_object);
            object_background_color.copyTo(normalized_background);
        }

        // pencil nose detect
        Mat half_object;
        Rect half_roi = Rect(0, 0, normalized_object.cols, normalized_object.rows / 2);
        normalized_object(half_roi).copyTo(half_object);
        cvtColor(half_object, half_object, COLOR_BGR2GRAY);
        int upside = countNonZero(half_object);

        half_roi.y = normalized_object.rows / 2;
        normalized_object(half_roi).copyTo(half_object);
        cvtColor(half_object, half_object, COLOR_BGR2GRAY);
        int downside = countNonZero(half_object);

        if (downside < upside) {
            rotate(normalized_object, normalized_object, ROTATE_180);
            rotate(normalized_background, normalized_background, ROTATE_180);
        }

        bitwise_or(normalized_object, normalized_background, normalized_object);
        normalized_objects.push_back(normalized_object);
        objects.push_back(object);
    }

    if (objects.size() > 0)
        match(normalized_objects, objects);

    return objects;
}

void ObjectClassifier::reduceNoise(vector<vector<Point>>& contours, const int minLen)
{
    auto condition = [minLen](vector<Point>& v) {
        return arcLength(v, true) < minLen ? true : false;
    };
    contours.erase(std::remove_if(contours.begin(), contours.end(), condition), contours.end());
}

void ObjectClassifier::sortContours(std::vector<std::vector<cv::Point>>& contours)
{
    auto condition = [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
        return arcLength(a, true) > arcLength(b, true) ? true : false;
    };
    std::sort(begin(contours), end(contours), condition);
}

void ObjectClassifier::match(const std::vector<Mat>& templates, std::vector<Object>& objects)
{
    int id_counter = 0;
    for (size_t i = 0; i < objects.size(); ++i)    
    {
        if (objects[i].id != OBJECT_ID_EMPTY){
            continue;
        }

        vector<int> indexes;
        for (int j = 0; j < objects.size(); ++j) 
        {
            if (i == j || objects[j].id != OBJECT_ID_EMPTY) {
                continue;
            }
            indexes.push_back(j);
        }

        if (indexes.size() > 0) 
        {
            Mat tm_image;
            hconcatMatrix(templates, indexes, tm_image);

            Mat tm_result;
            const Mat& templ = templates[i];
            int result_cols = tm_image.cols - templ.cols + 1;
            int result_rows = tm_image.rows - templ.rows + 1;
            tm_result.create(result_rows, result_cols, CV_32FC1);

            matchTemplate(tm_image, templ, tm_result, TM_CCOEFF_NORMED);

            while (1) 
            {
                double minVal; double maxVal; Point minLoc; Point maxLoc;
                Point matchLoc;
                minMaxLoc(tm_result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
                matchLoc = maxLoc;

                if (maxVal < TM_MATCH_TRESHOLD)
                    break;

                int index = hfindMatrixIndexByPosition(templates, indexes, matchLoc + Point(templ.cols / 2, 0));
                objects[index].id = id_counter;
                rectangle(tm_result, Point(matchLoc.x - templ.cols / 2, 0), Point(matchLoc.x + templ.cols / 2, tm_result.size().height), Scalar(0, 0, 0), cv::FILLED);
            }
        }

        objects[i].id = id_counter;
        ++id_counter;
    }
}

void ObjectClassifier::hconcatMatrix(const std::vector<cv::Mat>& src, const std::vector<int>& indexes, cv::Mat& dst)
{
    if (!src.size()) return;

    vector<int> heights;
    int height, max_width = 0, width = 0;

    for (int index : indexes) {
        width += src[index].size().width;
    }
    for (const auto& object : src) {
        heights.push_back(object.size().height);
        max_width = object.size().width > max_width ? object.size().width : max_width;
    }
    height = *std::max_element(begin(heights), end(heights));
    width = width > max_width ? width : max_width;

    dst = Mat::zeros(Size(width, height), src[0].type());
    dst = Scalar::all(BACKGROUND_COLOR);
    Point2i y_offset;
    for (int index : indexes) {
        const Mat& object = src[index];
        Rect roi(Rect(y_offset, y_offset + Point2i(object.size().width, object.size().height)));
        object.copyTo(dst(roi));
        y_offset += Point2i(object.size().width, 0);
    }
}

int ObjectClassifier::hfindMatrixIndexByPosition(const std::vector<cv::Mat>& src, const std::vector<int>& indexes, const cv::Point& position)
{
    if (!src.size()) return -1;

    int width = 0;
    int result_index = -1;

    for (int index : indexes) {
        width += src[index].size().width;
        if (width > position.x) {
            result_index = index;
            break;
        }
    }

    return result_index;
}
