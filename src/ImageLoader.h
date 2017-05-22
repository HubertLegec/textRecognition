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

class ImageLoader {
private:
    string imagePath;
    vector<Mat> channels;
    vector<Mat> colorChannels;
    Mat image;
public:
    ImageLoader(string &imagePath);

    /** Must be called before other methods in this class */
    void loadImage();

    /** Returns loaded image */
    Mat getImage() const;

    /** Returns gray and negative gray channel */
    vector<Mat> getChannels() const;

    /** Returns color and negative color channels */
    vector<Mat> getColorChannels() const;
};


#endif TEXTRECOGNITION_IMAGELOADER_H
