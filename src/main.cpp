#include <string>
#include "opencv2/text.hpp"
#include "opencv2/highgui.hpp"
#include "ImageLoader.h"
#include "TextDetector.h"
#include "TextRecognizer.h"


int main(int argc, char *argv[]) {
    std::string imagePath(argv[1]);
    std::string classifierNM1Path = "trained_classifierNM1.xml";
    std::string classifierNM2Path = "trained_classifierNM2.xml";

    ImageLoader loader(imagePath);
    loader.loadImage();
    vector<Mat> channels = loader.getColorChannels();

    TextDetector detector(classifierNM1Path, classifierNM2Path, loader.getImage(), channels);
    detector.detect();
    Mat decomposition = detector.getImageDecomposition();

    TextRecognizer recognizer(loader.getImage(), channels);
    recognizer.recognize(detector.getRegions(), detector.getNmBoxes(), detector.getNmRegionGroups());
    Mat outImg = recognizer.getOutImage();

    namedWindow("decomposition", WINDOW_NORMAL);
    imshow("decomposition", decomposition);
    namedWindow("recognition", WINDOW_NORMAL);
    imshow("recognition", outImg);

    waitKey(0);
    return 0;
}