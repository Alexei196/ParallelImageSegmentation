#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

//Only works on square images
Mat sobel(const Mat &gray_img, int threshold) {
    Mat sobel_img(gray_img.rows, gray_img.cols, CV_8UC1);
    if(threshold < 256) {
        threshold*= threshold;
    }

    for (int row = 0; row < gray_img.rows; row++)
        for (int col = 0; col < gray_img.cols; col++)
        {
            try {if (row >= gray_img.rows - 2 || col >= gray_img.cols - 2){
                sobel_img.at<unsigned char>(col,row) = 0;
                continue;
            }
            Mat G;
            gray_img(cv::Rect(row, col, 3, 3)).copyTo(G);
            G.convertTo(G, CV_32SC1);
            Mat x = (Mat_<int>(3, 3) << 1, 0, -1, 2, 0, -2, 1, 0, -1);
            Mat y = (Mat_<int>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
            double G_x = sum(G.mul(x))[0];
            double G_y = sum(G.mul(y))[0];
            double pixel = pow(G_x, 2) + pow(G_y, 2);
            if (pixel <= threshold)
                pixel = 0;
            else 
                pixel = 128;
            sobel_img.at<unsigned char>(col,row) = pixel;
            } catch(int error) {
                fprintf(stderr, "ERROR:%d\n", error);
                continue;
            }
        }
    return sobel_img;
}