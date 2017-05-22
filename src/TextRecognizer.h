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
    static const int TEXT_IMAGE_BORDER;
    Mat image;
    vector<Mat> channels;
    Mat outImage;
    Ptr<OCRTesseract> ocr;
    float scaleFont;
    vector<string> wordsDetection;

    Mat getTextGroupImage(Rect nmBox, vector<Vec2i> nmRegionGroup, vector<vector<ERStat>> regions) const;

    void erDraw(vector<vector<ERStat>> regions, vector<Vec2i> group, Mat &segmentation) const;

    static bool isWordToOmit(string text, float confidence);

    void drawTextBox(const Rect &rect, const string &text);

public:
    TextRecognizer(const Mat &image, const vector<Mat> channels);

    /**
     * This method must be called before other in this class
     * @param regions Vector of ER's retrieved from the ERFilter algorithm from each channel
     * @param nmBoxes List of boxes enclosing a group of characters
     * @param nmRegionGroups Set of lists of indexes to provided regions
     * */
    void recognize(vector<vector<ERStat>> regions, vector<Rect> nmBoxes, vector<vector<Vec2i>> nmRegionGroups);

    /** Get image with recognized words printed on it */
    Mat getOutImage() const;

    /** Get list of detected words */
    vector<string> getWordsDetection() const;
};

#endif TEXTRECOGNITION_TEXTRECOGNIZER_H
