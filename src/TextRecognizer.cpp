#include "TextRecognizer.h"
#include "opencv2/imgproc.hpp"

const int TextRecognizer::TEXT_IMAGE_BORDER = 15;

TextRecognizer::TextRecognizer(const Mat &image, const vector<Mat> channels) {
    this->image = image;
    this->channels = channels;
    this->ocr = OCRTesseract::create();
    float scaleImage = 600.0f / image.rows;
    scaleFont = (2 - scaleImage) / 1.4f;
    image.copyTo(outImage);
}

void TextRecognizer::recognize(vector<vector<ERStat>> regions, vector<Rect> nmBoxes, vector<vector<Vec2i>> nmRegionGroups) {
    for (int i = 0; i < nmBoxes.size(); i++) {
        Mat groupImg = getTextGroupImage(nmBoxes[i], nmRegionGroups[i], regions);

        vector<Rect> boxes;
        vector<string> words;
        vector<float> confidences;
        string output;
        ocr->run(groupImg, output, &boxes, &words, &confidences, OCR_LEVEL_WORD);
        output.erase(remove(output.begin(), output.end(), '\n'), output.end());

        if (output.size() < 3) {
            continue;
        }

        for (int j = 0; j < boxes.size(); j++) {
            if (isWordToOmit(words[j], confidences[j])) {
                continue;
            }
            boxes[j].x += nmBoxes[i].x - TEXT_IMAGE_BORDER;
            boxes[j].y += nmBoxes[i].y - TEXT_IMAGE_BORDER;
            wordsDetection.push_back(words[j]);
            drawTextBox(boxes[j], words[j]);
        }
    }
}

void TextRecognizer::erDraw(vector<vector<ERStat>> regions, vector<Vec2i> group, Mat &segmentation) const {
    for (Vec2i gr : group) {
        ERStat er = regions[gr[0]][gr[1]];
        // deprecate the root region
        if (er.parent == NULL) {
            continue;
        }
        Mat channel = channels[gr[0]];
        int newMaskVal = 255;
        int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
        Point seedPoint(er.pixel % channel.cols, er.pixel / channel.cols);
        floodFill(channel, segmentation, seedPoint, Scalar(255), 0, Scalar(er.level), Scalar(0), flags);
    }
}

bool TextRecognizer::isWordToOmit(string text, float confidence) {
    return text.size() < 2 || confidence < 51 ||
           (text.size() == 2 && text[0] == text[1]) ||
           (text.size() < 4 && confidence < 60);
}

void TextRecognizer::drawTextBox(const Rect &rect, const string &text) {
    Scalar textColor(255, 255, 255);
    Scalar fillColor(55, 55, 255);
    Scalar borderColor(55, 55, 255);
    rectangle(outImage, rect.tl(), rect.br(), borderColor, 2);
    Size wordSize = getTextSize(text, FONT_HERSHEY_SIMPLEX, scaleFont, (int) (3 * scaleFont), NULL);
    rectangle(outImage, rect.tl() - Point(3, wordSize.height + 3), rect.tl() + Point(wordSize.width, 0), fillColor, -1);
    putText(outImage, text, rect.tl() - Point(1, 1), FONT_HERSHEY_SIMPLEX, scaleFont, textColor, (int) (3 * scaleFont));
}

Mat TextRecognizer::getTextGroupImage(Rect nmBox, vector<Vec2i> nmRegionGroup, vector<vector<ERStat>> regions) const {
    Mat groupImg = Mat::zeros(image.rows + 2, image.cols + 2, CV_8UC1);
    erDraw(regions, nmRegionGroup, groupImg);
    groupImg(nmBox).copyTo(groupImg);
    copyMakeBorder(groupImg, groupImg, TEXT_IMAGE_BORDER, TEXT_IMAGE_BORDER, TEXT_IMAGE_BORDER, TEXT_IMAGE_BORDER, BORDER_CONSTANT);
    return groupImg;
}

Mat TextRecognizer::getOutImage() const {
    return outImage;
}

vector<string> TextRecognizer::getWordsDetection() const {
    return wordsDetection;
}
