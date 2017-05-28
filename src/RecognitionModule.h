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
    Mat outImg;
    std::vector<Mat> decompositions;
    ColorMode mode;
    std::vector<DetectedWord> words;
public:
    RecognitionModule(std::string imagePath, ColorMode mode = RGBL);
    void process(std::string classifierNM1Path, std::string classifierNM2Path);

    const Mat &getOutImg() const;

    const vector<Mat, allocator<Mat>> &getDecompositions() const;

    const vector<DetectedWord, allocator<DetectedWord>> &getWords() const;
};


#endif //TEXTRECOGNITION_RECOGNITIONMODULE_H
