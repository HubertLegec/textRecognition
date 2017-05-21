#include "TextRecognizer.h"
#include "opencv2/imgproc.hpp"

TextRecognizer::TextRecognizer(const Mat &image, const vector<Mat> channels) {
    this->image = image;
    this->channels = channels;
    this->ocr = OCRTesseract::create();
    scaleImage = 600.0f / image.rows;
    scaleFont = (2 - scaleImage) / 1.4f;
    image.copyTo(outImage);
    image.copyTo(outImageDetection);
    outImageSegmentation = Mat::zeros(image.rows + 2, image.cols + 2, CV_8UC1);
}

bool TextRecognizer::isRepetitive(const string &s) {
    int count = 0;
    for (int i = 0; i < s.size(); i++) {
        if ((s[i] == 'i') ||
            (s[i] == 'l') ||
            (s[i] == 'I'))
            count++;
    }
    return count > ((s.size() + 1) / 2);
}

void TextRecognizer::recognize(vector<vector<ERStat>> regions, vector<Rect> nmBoxes, vector<vector<Vec2i>> nmRegionGroups) {
    for (int i = 0; i < nmBoxes.size(); i++) {
        drawRectOnOutput(nmBoxes[i]);
        Mat groupImg = Mat::zeros(image.rows + 2, image.cols + 2, CV_8UC1);
        erDraw(regions, nmRegionGroups[i], groupImg);
        Mat groupSegmentation;
        groupImg.copyTo(groupSegmentation);
        groupImg(nmBoxes[i]).copyTo(groupImg);
        copyMakeBorder(groupImg, groupImg, 15, 15, 15, 15, BORDER_CONSTANT, Scalar(0));

        vector<Rect> boxes;
        vector<string> words;
        vector<float> confidences;
        ocr->run(groupImg, output, &boxes, &words, &confidences, OCR_LEVEL_WORD);

        output.erase(remove(output.begin(), output.end(), '\n'), output.end());

        if (output.size() < 3) {
            continue;
        }

        for (int j = 0; j < boxes.size(); j++) {
            boxes[j].x += nmBoxes[i].x - 15;
            boxes[j].y += nmBoxes[i].y - 15;

            if ((words[j].size() < 2) || (confidences[j] < 51) ||
                ((words[j].size() == 2) && (words[j][0] == words[j][1])) ||
                ((words[j].size() < 4) && (confidences[j] < 60)) ||
                isRepetitive(words[j])) {
                continue;
            }

            wordsDetection.push_back(words[j]);
            rectangle(outImage, boxes[j].tl(), boxes[j].br(), Scalar(255, 0, 255), 3);
            Size word_size = getTextSize(words[j], FONT_HERSHEY_SIMPLEX, (double) scaleFont, (int) (3 * scaleFont),
                                         NULL);
            rectangle(outImage, boxes[j].tl() - Point(3, word_size.height + 3),
                      boxes[j].tl() + Point(word_size.width, 0), Scalar(255, 0, 255), -1);
            putText(outImage, words[j], boxes[j].tl() - Point(1, 1), FONT_HERSHEY_SIMPLEX, scaleFont,
                    Scalar(255, 255, 255), (int) (3 * scaleFont));
            outImageSegmentation = outImageSegmentation | groupSegmentation;
        }
    }
}

Mat TextRecognizer::getOutImage() const {
    return outImage;
}

void TextRecognizer::drawRectOnOutput(const Rect &rect) {
    rectangle(outImageDetection, rect.tl(), rect.br(), Scalar(0, 255, 255), 3);
}

void TextRecognizer::erDraw(vector<vector<ERStat>> regions, vector<Vec2i> group, Mat &segmentation) const {
    for (int r = 0; r < group.size(); r++) {
        ERStat er = regions[group[r][0]][group[r][1]];
        // deprecate the root region
        if (er.parent == NULL) {
            continue;
        }
        int newMaskVal = 255;
        int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
        Point seedPoint(er.pixel % channels[group[r][0]].cols, er.pixel / channels[group[r][0]].cols);
        floodFill(channels[group[r][0]], segmentation, seedPoint, Scalar(255), 0, Scalar(er.level), Scalar(0), flags);
    }
}
