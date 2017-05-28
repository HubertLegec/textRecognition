#include <string>
#include <iostream>
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
    auto decompositions = detector.getImageDecompositions();

    TextRecognizer recognizer(loader.getImage(), channels);
    recognizer.recognize(detector.getRegions(), detector.getNmBoxes(), detector.getNmRegionGroups());
    Mat outImg = recognizer.getOutImage();

    std::cout << "--- words ---" << std::endl;
    for (auto word : recognizer.getWordsDetection()) {
        std::cout << word.toString() << endl;
    }
    std::cout << "-------------" << std::endl;

    for(int i = 0; i < decompositions.size(); i++) {
        namedWindow("decomposition" + to_string(i), WINDOW_NORMAL);
        imshow("decomposition" + to_string(i), decompositions[i]);
    }
    namedWindow("recognition", WINDOW_NORMAL);
    imshow("recognition", outImg);

    waitKey(0);
    return 0;
}