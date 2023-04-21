#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

Mat sobel(const Mat&, int);

int main(int argc, char **argv)
{   
    if(argc < 2) {
        printf("Usage: ./sobel <IMAGE_PATH>\n");
        return 1;
    }
    Mat gray_img = imread(argv[1], IMREAD_GRAYSCALE);
    if(!gray_img.data) {
        printf("Cannot find image1\n");
        return 1;
    }
    Mat sobel_img = sobel(gray_img, 60);
    cv::namedWindow("Sobel");
    cv::imshow("Sobel",sobel_img);
    cv::waitKey(0);
    return 0;
}

Mat sobel(const Mat &gray_img, int threshold) {
    Mat sobel_img(gray_img.rows-2, gray_img.cols-2, CV_8UC1);
    if(threshold < 256) {
        threshold*= threshold;
    }

    for (int row = 0; row < gray_img.rows - 2; row++)
        for (int col = 0; col < gray_img.cols - 2; col++)
        {
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
                pixel = 255;
            sobel_img.at<unsigned char>(col,row) = pixel;
        }
    return sobel_img;
}