#include "ObjectClassifier.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <algorithm>

using namespace cv;
using namespace std;

std::vector<Object> ObjectClassifier::process(cv::Mat image)
{
    std::vector<Object> objects;
    std::vector<Mat> normalized_objects;
    Mat src_gray, src_bin, canny_output;

	cvtColor(image, src_gray, COLOR_BGR2GRAY);
	blur(src_gray, src_gray, Size(3, 3));
    //inRange(src_gray, cv::Scalar(a), cv::Scalar(b), bin);
    //threshold(src_gray, bin, cv::Scalar(a), cv::Scalar(b), THRESH_BINARY);
    adaptiveThreshold(src_gray, src_bin, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 1025, 32); // MAGIC!
    Canny(src_bin, canny_output, CANNY_TRESHOLD, CANNY_TRESHOLD * 2);

#ifdef CLASSIFIER_DEBUG
    imshow("src_gray", src_gray);                           // Debug Only
    imshow("bin", src_bin);                                 // Debug Only
    imshow("canny_output", canny_output);                   // Debug Only  
    Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3); // Debug Only  
#endif

    vector<Vec4i> hierarchy;
    vector<vector<Point> > contours;
    findContours(canny_output, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    reduceNoise(contours);

    for (size_t i = 0; i < contours.size(); i++)                                      
    {
        Object object;

        Vec4f vertical_line;
        fitLine(contours[i], vertical_line, DIST_L2, 0, 0.01, 0.01);
        float vx = vertical_line[0];
        float vy = vertical_line[1];
        object.id = -1;
        object.contour = contours[i];
        object.roi = boundingRect(object.contour);

        Mat object_mask = Mat::zeros(drawing.size(), CV_8UC1);
        drawContours(object_mask, contours, i, Scalar::all(255), -1); // -1 == filled
        Mat masked_image;
        bitwise_and(src_gray, object_mask, masked_image);
        Mat object_image;
        masked_image(object.roi).copyTo(object_image);

        // angle between vertical and fitLine result
        double angle = -atanf(vx / vy) / CV_PI * 180.0;
        Point2f center(object_image.cols / 2.0, object_image.rows / 2.0);
        Mat rot = getRotationMatrix2D(center, angle, 1.0);
        // stackoverflow here
        Rect bbox = RotatedRect(center, object_image.size(), angle).boundingRect();
        rot.at<double>(0, 2) += bbox.width / 2.0 - center.x;
        rot.at<double>(1, 2) += bbox.height / 2.0 - center.y;

        Mat object_rotated;
        warpAffine(object_image, object_rotated, rot, bbox.size());

        Rect crop_roi = boundingRect(object_rotated);
        Mat normalized_object;
        object_rotated(crop_roi).copyTo(normalized_object);

        // pencil nose detect
        Mat half_object;
        Rect half_roi = Rect(0, 0, normalized_object.cols, normalized_object.rows / 2);
        normalized_object(half_roi).copyTo(half_object);
        int upside = countNonZero(half_object);

        half_roi.y = normalized_object.rows / 2;
        normalized_object(half_roi).copyTo(half_object);
        int downside = countNonZero(half_object);

        if (downside < upside)
            rotate(normalized_object, normalized_object, ROTATE_180);

        normalized_objects.push_back(normalized_object);
        objects.push_back(object);

#ifdef CLASSIFIER_DEBUG
        Scalar contours_color = Scalar(255, 255, 255);                                          // Debug Only
        drawContours(drawing, contours, (int)i, contours_color, FILLED, LINE_8, hierarchy, 0);  // Debug Only  
        float x = vertical_line[2];                                                             // Debug Only  
        float y = vertical_line[3];                                                             // Debug Only  
        Point point1(x - vx * 200, y - vy * 200);                                               // Debug Only  
        Point point2(x + vx * 200, y + vy * 200);                                               // Debug Only  
        Scalar line_color = Scalar(0, 0, 255);                                                  // Debug Only  
        line(drawing, point1, point2, line_color);                                              // Debug Only  
        Scalar rect_color = Scalar(0, 255, 0);                                                  // Debug Only  
        rectangle(drawing, object.roi, rect_color);
        imshow(to_string(i), normalized_object);
#endif
    }

    imshow("Res", drawing);

    return objects;
}

void ObjectClassifier::reduceNoise(vector<vector<Point>>& contours)
{
    static int minLen = 100;

    auto condition = [](vector<Point>& v) {
        return arcLength(v, true) < minLen ? true : false;
    };

    contours.erase(std::remove_if(contours.begin(), contours.end(), condition), contours.end());
}
