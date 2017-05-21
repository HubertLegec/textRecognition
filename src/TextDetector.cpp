#include "TextDetector.h"
#include "opencv2/imgproc.hpp"

const int TextDetector::thresholdDelta = 8;
const float TextDetector::minArea = 0.00025;
const float TextDetector::maxArea = 0.13;
const float TextDetector::minProbability = 0.4;
const bool TextDetector::nonMaxSuppression = true;
const float TextDetector::minProbabilityDiff = 0.1;
const float TextDetector::minProbabilityNM2 = 0.7;

TextDetector::TextDetector(const string classifierNM1Path, const string classifierNM2Path, const Mat &image,
                           const vector<Mat> &channels) {
    this->erFilter1 = createERFilterNM1(
            loadClassifierNM1(classifierNM1Path),
            thresholdDelta,
            minArea,
            maxArea,
            minProbability,
            nonMaxSuppression,
            minProbabilityDiff
    );
    this->erFilter2 = createERFilterNM2(
            loadClassifierNM2(classifierNM2Path),
            minProbabilityNM2
    );
    this->image = image;
    this->channels = channels;
    regions = vector<vector<ERStat>>(channels.size());
}

void TextDetector::detect() {
    for (int i = 0; i < channels.size(); i++) {
        erFilter1->run(channels[i], regions[i]);
        erFilter2->run(channels[i], regions[i]);
    }
    erGrouping(image, channels, regions, nmRegionGroups, nmBoxes, ERGROUPING_ORIENTATION_HORIZ);
}

vector<vector<ERStat>> TextDetector::getRegions() const {
    return regions;
}

Mat TextDetector::getImageDecomposition() const {
    Mat outImgDecomposition = Mat::zeros(image.rows + 2, image.cols + 2, CV_8UC1);
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
        outImgDecomposition = outImgDecomposition | tmp;
        tmpGroup.clear();
    }
    return outImgDecomposition;
}

void TextDetector::erDraw(vector<Vec2i> group, Mat &segmentation) const {
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

vector<Rect> TextDetector::getNmBoxes() const {
    return nmBoxes;
}

vector<vector<Vec2i>> TextDetector::getNmRegionGroups() const {
    return nmRegionGroups;
}
