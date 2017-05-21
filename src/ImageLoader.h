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
    Mat image;
public:
    ImageLoader(string& imagePath);
    void loadImage();
    Mat& getImage();
    vector<Mat>& getChannels();
};


#endif //TEXTRECOGNITION_IMAGELOADER_H
