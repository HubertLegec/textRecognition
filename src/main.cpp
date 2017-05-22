#include <string>
#include "opencv2/text.hpp"
#include "opencv2/highgui.hpp"
#include "ImageLoader.h"
#include "TextDetector.h"
#include "TextRecognizer.h"

using namespace std;
using namespace cv;


int main(int argc, char *argv[]) {
    string imagePath(argv[1]);
    string classifierNM1Path = "trained_classifierNM1.xml";
    string classifierNM2Path = "trained_classifierNM2.xml";

    ImageLoader loader(imagePath);
    loader.loadImage();

    TextDetector detector(classifierNM1Path, classifierNM2Path, loader.getImage(), loader.getChannels());
    detector.detect();
    Mat decomposition = detector.getImageDecomposition();

    TextRecognizer recognizer(loader.getImage(), loader.getChannels());
    recognizer.recognize(detector.getRegions(), detector.getNmBoxes(), detector.getNmRegionGroups());
    Mat outImg = recognizer.getOutImage();

    namedWindow("decomposition", WINDOW_NORMAL);
    imshow("decomposition", decomposition);
    namedWindow("recognition", WINDOW_NORMAL);
    imshow("recognition", outImg);
    namedWindow("detection", WINDOW_GUI_NORMAL);
    imshow("detection", detector.getImageDetection());

    waitKey(0);
    return 0;
}