//
// Created by Hubert Legęć on 21.05.2017.
//

#ifndef TEXTRECOGNITION_IMAGELOADER_H
#define TEXTRECOGNITION_IMAGELOADER_H

#include <string>
#include <vector>
#include <opencv2/text.hpp>


using namespace std;
using namespace cv;
using namespace cv::text;

class ImageLoader {
private:
    string imagePath;
    Mat image;
public:
    ImageLoader(const string imagePath);
    ImageLoader();

    /** Must be called before other methods in this class */
    void loadImage();

    /** Must be called before other methods in this class */
    void loadImage(Mat image);

    /** Returns loaded image */
    Mat getImage() const;

    /** Returns gray and negative gray channel */
    vector<Mat> getChannels() const;

    /** Returns color and negative color channels
     * @param mode mode of channel calculations.
     * Available options are: **ERFILTER_NM_RGBLGrad** (used by default) and **ERFILTER_NM_IHSGrad**.
     */
    vector<Mat> getColorChannels(int mode = ERFILTER_NM_RGBLGrad) const;
};


#endif //TEXTRECOGNITION_IMAGELOADER_H
