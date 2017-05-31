#include <string>
#include <iostream>
#include "opencv2/text.hpp"
#include "opencv2/highgui.hpp"
#include "RecognitionModule.h"


int main(int argc, char *argv[]) {
    std::string imagePath(argv[1]);
    std::string classifierNM1Path = "trained_classifierNM1.xml";
    std::string classifierNM2Path = "trained_classifierNM2.xml";

    RecognitionModule recognitionModule(imagePath);
    Mat outImg = recognitionModule.process(classifierNM1Path, classifierNM2Path);
    auto decompositions = recognitionModule.getDecompositions();
    auto words = recognitionModule.getWords();

    std::cout << "--- words ---" << std::endl;
    for (auto word : words) {
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