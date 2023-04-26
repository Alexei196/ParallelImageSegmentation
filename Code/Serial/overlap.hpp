#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat overlap(const Mat &sobel_img, const Mat& orig_image) {

    // Convert sobel image type to same type as original image to run bitwise_or
    cv::Mat simg_16(sobel_img.rows, sobel_img.cols, CV_8UC3);
    int from_to[] = {0, 0, 0, 1, 0, 2};
    cv::mixChannels(&sobel_img, 1, &simg_16, 1, from_to, 3);
    cv::Mat sobel_updated;
    simg_16.convertTo(sobel_updated, CV_16U);
    
    cv::Mat overlappedImage;
    cv::bitwise_or(simg_16, orig_image, overlappedImage);
    return overlappedImage;
}