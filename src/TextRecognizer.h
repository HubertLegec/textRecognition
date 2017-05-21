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
    Ptr<OCRTesseract> ocr;
    float scaleFont;
    vector<string> wordsDetection;

    void erDraw(vector<vector<ERStat>> regions, vector<Vec2i> group, Mat &segmentation) const;

    static bool isWordToOmit(string word, float confidence);

    void drawTextBox(const Rect &rect, const string &text);

public:
    TextRecognizer(const Mat &image, const vector<Mat> channels);

    void recognize(vector<vector<ERStat>> regions, vector<Rect> nmBoxes, vector<vector<Vec2i>> nmRegionGroups);

    Mat getOutImage() const;

    vector<string> getWordsDetection() const;
};

#endif TEXTRECOGNITION_TEXTRECOGNIZER_H
