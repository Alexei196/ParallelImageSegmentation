#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat overlap(const Mat &sobel_img, const Mat &orig_image) {
    // Convert sobel image type to same type as original image to run bitwise_or
    cv::Mat outputMatrix(sobel_img.rows, sobel_img.cols, CV_8UC1);
    #pragma omp parallel for collapse(2)
    for (int row = 0; row < sobel_img.rows; row++)
        for (int col = 0; col < sobel_img.cols; col++)
        {
            if(sobel_img.at<unsigned char>(row, col) == 255) {
                outputMatrix.at<unsigned char>(row, col) = 255;
            } else {
                outputMatrix.at<unsigned char>(row, col) = orig_image.at<unsigned char>(row, col);
            }
        }
    return outputMatrix;
}