#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "ImageLoader.h"

using namespace std;
using namespace cv::text;

ImageLoader::ImageLoader(string &imagePath) {
    this->imagePath = imagePath;
}

void ImageLoader::loadImage() {
    cout << "--- " << "Open image: " << this->imagePath << " ---" << endl;
    image = imread(this->imagePath);
    cout << "width: " << image.cols << "px" << endl;
    cout << "height: " << image.rows << "px" << endl;

    // --- gray channel ---
    Mat gray;
    cvtColor(this->image, gray, COLOR_RGB2GRAY);
    channels.push_back(gray);
    // Append negative channels to detect ER- (bright regions over dark background)
    Mat greyNegative = 255 - gray;
    channels.push_back(greyNegative);
    // --- color channels ---
    computeNMChannels(image, colorChannels);
    size_t cn = colorChannels.size();
    for (int i = 0; i < cn - 1; i++) {
        Mat negChannel = 255 - colorChannels[i];
        colorChannels.push_back(negChannel);
    }
}

Mat ImageLoader::getImage() const {
    return this->image;
}

vector<Mat> ImageLoader::getChannels() const {
    return this->channels;
}

vector<Mat> ImageLoader::getColorChannels() const {
    return this->colorChannels;
}
