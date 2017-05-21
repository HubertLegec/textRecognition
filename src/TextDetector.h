#ifndef TEXTRECOGNITION_TEXTDETECTOR_H
#define TEXTRECOGNITION_TEXTDETECTOR_H

#include <vector>
#include <string>
#include "opencv2/text.hpp"

using namespace std;
using namespace cv;
using namespace cv::text;

class TextDetector {
private:
    static const int thresholdDelta;
    static const float minArea;
    static const float maxArea;
    static const float minProbability;
    static const bool nonMaxSuppression;
    static const float minProbabilityDiff;
    static const float minProbabilityNM2;
    Mat image;
    vector<Mat> channels;
    Ptr<ERFilter> erFilter1;
    Ptr<ERFilter> erFilter2;
    vector<vector<ERStat>> regions;
    vector<Rect> nmBoxes;
    vector<vector<Vec2i>> nmRegionGroups;

    //Draw ER's in an image via floodFill
    void erDraw(vector<Vec2i> group, Mat &segmentation) const;

    static void drawRectOnImage(const Rect &rect, Mat &image);

public:
    TextDetector(const string classifierNM1Path, const string classifierNM2Path, const Mat &image, const vector<Mat> &channels);

    void detect();

    vector<Rect> getNmBoxes() const;

    vector<vector<Vec2i>> getNmRegionGroups() const;

    vector<vector<ERStat>> getRegions() const;

    Mat getImageDecomposition() const;

    Mat getImageDetection() const;
};

#endif TEXTRECOGNITION_TEXTDETECTOR_H
