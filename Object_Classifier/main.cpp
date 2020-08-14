#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <algorithm>

#include "ObjectClassifier.h"

using namespace cv;
using namespace std;

Scalar get_color_by_index(int index)
{
    RNG rng(index);
    Scalar res((int)rng, (int)rng, (int)rng);
    return res;
}

int main(int argc, char** argv)
{
    Mat src = imread(samples::findFile("../pics/img3.jpg"));

    ObjectClassifier classifier;
    vector<Object> objects = classifier.process(src);

    for (auto object : objects) {
        cout << object.id << endl;
        rectangle(src, object.roi, get_color_by_index(object.id), 2);
    }

    imshow("main", src);
    waitKey();
    return 0;
}
