//
// Created by Hubert Legęć on 21.05.2017.
//

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "ImageLoader.h"

using namespace std;

ImageLoader::ImageLoader(string& imagePath) {
    this->imagePath = imagePath;
}

void ImageLoader::loadImage() {
    cout << "--- " << "Open image: " << this->imagePath << " ---" << endl;
    this->image = imread(this->imagePath);
    cout << "image width: " << image.cols << endl;
    cout << "image height: " << image.rows << endl;

    Mat grey;
    cvtColor(this->image, grey, COLOR_RGB2GRAY);
    channels.push_back(grey);
    Mat greyNegative = 255 - grey;
    channels.push_back(greyNegative);
}

Mat &ImageLoader::getImage() {
    return this->image;
}

vector<Mat> &ImageLoader::getChannels() {
    return this->channels;
}
