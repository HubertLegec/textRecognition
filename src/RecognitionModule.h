#ifndef TEXTRECOGNITION_RECOGNITIONMODULE_H
#define TEXTRECOGNITION_RECOGNITIONMODULE_H

#include <string>
#include <vector>
#include "ImageLoader.h"
#include "TextDetector.h"
#include "TextRecognizer.h"

enum ColorMode {
    GRAYSCALE,
    RGBL,
    IHSG
};

class RecognitionModule {
private:
    ImageLoader loader;
    cv::Mat outImg;
    std::vector<cv::Mat> decompositions;
    ColorMode mode;
    std::vector<DetectedWord> words;
    Mat process(std::string classifierNM1Path, std::string classifierNM2Path);
public:
    /*RecognitionModule(std::string imagePath, ColorMode mode = RGBL);*/
    RecognitionModule(ColorMode mode = RGBL);

    cv::Mat process(std::string classifierNM1Path, std::string classifierNM2Path, cv::Mat image);

    /*const vector<Mat> &getDecompositions() const;

    const vector<DetectedWord> &getWords() const;*/
};


#endif //TEXTRECOGNITION_RECOGNITIONMODULE_H
