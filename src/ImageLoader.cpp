#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "ImageLoader.h"

using namespace std;

ImageLoader::ImageLoader(string &imagePath) {
    this->imagePath = imagePath;
}

void ImageLoader::loadImage() {
    cout << "--- " << "Open image: " << this->imagePath << " ---" << endl;
    this->image = imread(this->imagePath);
    cout << "width: " << image.cols << "px" << endl;
    cout << "height: " << image.rows << "px" << endl;

    Mat grey;
    cvtColor(this->image, grey, COLOR_RGB2GRAY);
    channels.push_back(grey);
    Mat greyNegative = 255 - grey;
    channels.push_back(greyNegative);
}

Mat ImageLoader::getImage() const {
    return this->image;
}

vector<Mat> ImageLoader::getChannels() const {
    return this->channels;
}
