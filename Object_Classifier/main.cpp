#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <algorithm>

#include "ObjectClassifier.h"


using namespace cv;
using namespace std;


int main(int argc, char** argv)
{
    Mat src = imread(samples::findFile("../pics/img3.jpg"));

    ObjectClassifier classifier;
    vector<Object> objects = classifier.process(src);

    Scalar rect_color = Scalar(0, 255, 0);
    for(auto object : objects)
        rectangle(src, object.roi, rect_color);
    imshow("main", src);

    waitKey();
    return 0;
}
