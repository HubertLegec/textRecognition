#include "TextDetector.h"
#include "opencv2/imgproc.hpp"
#include <iostream>

const int TextDetector::THRESHOLD_DELTA = 8;
const float TextDetector::MIN_AREA = 0.0001;
const float TextDetector::MAX_AREA = 0.5;
const float TextDetector::MIN_PROBABILITY = 0.8;
const bool TextDetector::NON_MAX_SUPPRESSION = true;
const float TextDetector::MIN_PROBABILITY_DIFF = 0.2;
const float TextDetector::MIN_PROBABILITY_NM2 = 0.7;

TextDetector::TextDetector(const string classifierNM1Path, const string classifierNM2Path, const Mat &image,
                           const vector<Mat> &channels) {
    this->image = image;
    this->channels = channels;
    regions = vector<vector<ERStat>>(channels.size());
    // Create ERFilter objects with the 1st and 2nd stage default classifiers
    erFilter1 = createERFilterNM1(
            loadClassifierNM1(classifierNM1Path),
            THRESHOLD_DELTA,
            MIN_AREA,
            MAX_AREA,
            MIN_PROBABILITY,
            NON_MAX_SUPPRESSION,
            MIN_PROBABILITY_DIFF
    );
    erFilter2 = createERFilterNM2(
            loadClassifierNM2(classifierNM2Path),
            MIN_PROBABILITY_NM2
    );
}

void TextDetector::detect() {
    cout << "Detection...\n";
    // Apply the default cascade classifier to each independent channel
    for (int i = 0; i < channels.size(); i++) {
        erFilter1->run(channels[i], regions[i]);
        erFilter2->run(channels[i], regions[i]);
    }
    // Detect character groups
    erGrouping(image, channels, regions, nmRegionGroups, nmBoxes, ERGROUPING_ORIENTATION_HORIZ);
    cout << "Number of detected text regions: " << nmBoxes.size() << endl;
}

vector<vector<ERStat>> TextDetector::getRegions() const {
    return regions;
}

vector<Mat> TextDetector::getImageDecompositions() const {
    vector<Mat> outImgDecompositions;
    vector<Vec2i> tmpGroup;
    for (int i = 0; i < regions.size(); i++) {
        for (int j = 0; j < regions[i].size(); j++) {
            tmpGroup.push_back(Vec2i(i, j));
        }
        Mat tmp = Mat::zeros(image.rows + 2, image.cols + 2, CV_8UC1);
        erDraw(tmpGroup, tmp);
        if (i > 0) {
            tmp = tmp / 2;
        }
        outImgDecompositions.push_back(tmp);
        tmpGroup.clear();
    }
    return outImgDecompositions;
}

void TextDetector::erDraw(vector<Vec2i> group, Mat &segmentation) const {
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

vector<Rect> TextDetector::getNmBoxes() const {
    return nmBoxes;
}

vector<vector<Vec2i>> TextDetector::getNmRegionGroups() const {
    return nmRegionGroups;
}
