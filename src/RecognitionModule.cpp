#include "RecognitionModule.h"

using namespace std;

RecognitionModule::RecognitionModule(std::string imagePath, ColorMode mode) : loader(imagePath) {
    this->mode = mode;
    loader.loadImage();
}

void RecognitionModule::process(std::string classifierNM1Path, std::string classifierNM2Path) {
    vector<Mat> channels;
    if (mode == GRAYSCALE) {
        channels = loader.getChannels();
    } else if (mode == RGBL) {
        channels = loader.getColorChannels();
    } else {
        channels = loader.getColorChannels(ERFILTER_NM_IHSGrad);
    }
    TextDetector detector(classifierNM1Path, classifierNM2Path, loader.getImage(), channels);
    detector.detect();
    decompositions = detector.getImageDecompositions();
    TextRecognizer recognizer(loader.getImage(), channels);
    recognizer.recognize(detector.getRegions(), detector.getNmBoxes(), detector.getNmRegionGroups());
    outImg = recognizer.getOutImage();
    words = recognizer.getWordsDetection();
}

const Mat &RecognitionModule::getOutImg() const {
    return outImg;
}

const vector<Mat, allocator<Mat>> &RecognitionModule::getDecompositions() const {
    return decompositions;
}

const vector<DetectedWord, allocator<DetectedWord>> &RecognitionModule::getWords() const {
    return words;
}


