#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main(int argc, char **argv)
{   
    Mat image = imread("C:/Users/mdp72/Documents/VS Code Projects/openCV/image(2).jpg");
    Mat gray_img;
    cvtColor(image, gray_img, COLOR_BGR2GRAY);
    Mat sobel_img(gray_img.rows-2, gray_img.cols-2, CV_8UC1);
    double threshold = 60;

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
            double pixel = sqrt(pow(G_x, 2) + pow(G_y, 2));
            if (pixel <= threshold)
                pixel = 0;
            sobel_img.at<unsigned char>(col,row) = pixel;
        }
    cv::namedWindow("Sobel");
    cv::imshow("Sobel",sobel_img);
    cv::waitKey(0);
}