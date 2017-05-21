#ifndef TEXTRECOGNITION_TEXTRECOGNIZER_H
#define TEXTRECOGNITION_TEXTRECOGNIZER_H

#include <vector>
#include <string>
#include "opencv2/text.hpp"

using namespace std;
using namespace cv;
using namespace cv::text;

class TextRecognizer {
private:
    Mat image;
    vector<Mat> channels;
    Mat outImage;
    Mat outImageDetection;
    Mat outImageSegmentation;
    Ptr<OCRTesseract> ocr;
    string output;
    float scaleImage;
    float scaleFont;
    vector<string> wordsDetection;

    static bool isRepetitive(const string &s);

    void drawRectOnOutput(const Rect &rect);

    void erDraw(vector<vector<ERStat>> regions, vector<Vec2i> group, Mat &segmentation) const;

public:
    TextRecognizer(const Mat &image, const vector<Mat> channels);

    void recognize(vector<vector<ERStat>> regions, vector<Rect> nmBoxes, vector<vector<Vec2i>> nmRegionGroups);

    Mat getOutImage() const;
};

#endif TEXTRECOGNITION_TEXTRECOGNIZER_H
