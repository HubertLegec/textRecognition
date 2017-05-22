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
    /* Threshold step in subsequent thresholds when extracting the component tree */
    static const int THRESHOLD_DELTA;
    /* The minimum area (% of image size) allowed for retrieved ER’s */
    static const float MIN_AREA;
    /* The maximum area (% of image size) allowed for retrieved ER’s */
    static const float MAX_AREA;
    /* The minimum probability P(er|character) allowed for retrieved ER’s */
    static const float MIN_PROBABILITY;
    /* Whenever non-maximum suppression is done over the branch probabilities */
    static const bool NON_MAX_SUPPRESSION;
    /* The minimum probability difference between local maxima and local minima ERs */
    static const float MIN_PROBABILITY_DIFF;
    /* The minimum probability P(er|character) allowed for retreived ER's */
    static const float MIN_PROBABILITY_NM2;
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
    TextDetector(const string classifierNM1Path, const string classifierNM2Path, const Mat &image,
                 const vector<Mat> &channels);

    /** Must be called before other methods in this class */
    void detect();

    /** Get list of boxes enclosing a group of characters */
    vector<Rect> getNmBoxes() const;

    /** Get set of lists of indexes to provided regions */
    vector<vector<Vec2i>> getNmRegionGroups() const;

    /** Get vector of ER's retrieved from the ERFilter algorithm from each channel */
    vector<vector<ERStat>> getRegions() const;

    /** Get image with extracted Extremal Regions */
    Mat getImageDecomposition() const;

    /** Get image with marked regions which contains text */
    Mat getImageDetection() const;
};

#endif TEXTRECOGNITION_TEXTDETECTOR_H
