#include "TextRecognizer.h"
#include "opencv2/imgproc.hpp"

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
        Mat groupImg = Mat::zeros(image.rows + 2, image.cols + 2, CV_8UC1);
        erDraw(regions, nmRegionGroups[i], groupImg);
        groupImg(nmBoxes[i]).copyTo(groupImg);
        copyMakeBorder(groupImg, groupImg, 15, 15, 15, 15, BORDER_CONSTANT, Scalar(0));

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
            boxes[j].x += nmBoxes[i].x - 15;
            boxes[j].y += nmBoxes[i].y - 15;

            if (isWordToOmit(words[j], confidences[j])) {
                continue;
            }

            wordsDetection.push_back(words[j]);
            drawTextBox(boxes[j], words[j]);
        }
    }
}

Mat TextRecognizer::getOutImage() const {
    return outImage;
}

void TextRecognizer::erDraw(vector<vector<ERStat>> regions, vector<Vec2i> group, Mat &segmentation) const {
    for (Vec2i gr : group) {
        ERStat er = regions[gr[0]][gr[1]];
        // deprecate the root region
        if (er.parent == NULL) {
            continue;
        }
        int newMaskVal = 255;
        int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
        Point seedPoint(er.pixel % channels[gr[0]].cols, er.pixel / channels[gr[0]].cols);
        floodFill(channels[gr[0]], segmentation, seedPoint, Scalar(255), 0, Scalar(er.level), Scalar(0), flags);
    }
}

bool TextRecognizer::isWordToOmit(string word, float confidence) {
    return word.size() < 2 || confidence < 51 ||
           (word.size() == 2 && word[0] == word[1]) ||
           (word.size() < 4 && confidence < 60);
}

void TextRecognizer::drawTextBox(const Rect &rect, const string &text) {
    rectangle(outImage, rect.tl(), rect.br(), Scalar(215, 0, 255), 2);
    Size word_size = getTextSize(text, FONT_HERSHEY_SIMPLEX, scaleFont, (int) (3 * scaleFont), NULL);
    rectangle(outImage, rect.tl() - Point(3, word_size.height + 3),
              rect.tl() + Point(word_size.width, 0), Scalar(255, 0, 255), -1);
    putText(outImage, text, rect.tl() - Point(1, 1), FONT_HERSHEY_SIMPLEX, scaleFont,
            Scalar(255, 255, 255), (int) (3 * scaleFont));
}

vector<string> TextRecognizer::getWordsDetection() const {
    return wordsDetection;
}
