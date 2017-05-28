#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "ImageLoader.h"

using namespace std;
using namespace cv::text;

ImageLoader::ImageLoader(const string imagePath) {
    this->imagePath = imagePath;
}

void ImageLoader::loadImage() {
    cout << "--- Open image: " << this->imagePath << " ---" << endl;
    image = imread(this->imagePath);
    cout << "width: " << image.cols << "px" << endl;
    cout << "height: " << image.rows << "px" << endl;
}

Mat ImageLoader::getImage() const {
    return this->image;
}

vector<Mat> ImageLoader::getChannels() const {
    vector<Mat> channels;
    Mat gray;
    cvtColor(this->image, gray, COLOR_RGB2GRAY);
    channels.push_back(gray);
    // Append negative channels to detect ER- (bright regions over dark background)
    Mat greyNegative = 255 - gray;
    channels.push_back(greyNegative);
    return channels;
}

vector<Mat> ImageLoader::getColorChannels(int mode) const {
    vector<Mat> channels;
    computeNMChannels(image, channels, mode);
    size_t cn = channels.size();
    for (int i = 0; i < cn ; i++) {
        Mat negChannel = 255 - channels[i];
        channels.push_back(negChannel);
    }
    return channels;
}
