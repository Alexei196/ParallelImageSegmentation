#include <stdio.h>
#include <opencv2/core/mat.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

Mat overlap(Mat &sobel_img, Mat &orig_image) {
    Mat copy;
    cvtColor(orig_image, copy, COLOR_GRAY2RGB);
    #pragma omp parallel for collapse(2)
    for (int row = 0; row < sobel_img.rows; row++)
        for (int col = 0; col < sobel_img.cols; col++)
        {
            if(sobel_img.at<unsigned char>(row, col) > 0){       
                copy.at<Vec3b>(row, col).val[0] = (unsigned char) 0;
                copy.at<Vec3b>(row, col).val[1] = (unsigned char) 0;
                copy.at<Vec3b>(row, col).val[2] = (unsigned char) 255;
            }
        }
        
    return copy;
}