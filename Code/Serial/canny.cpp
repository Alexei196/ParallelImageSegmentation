#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main(int argc, char** argv )
{
    Mat image = imread("C:/Users/mdp72/Documents/VS Code Projects/openCV/image(2).jpg");
    cv::Mat contours;
    cv::Mat gray_image;

    //cvtColor( image, gray_image, COLOR_BGR2GRAY );

    cv::Canny(image,contours,35,90);

    cv::namedWindow("Image");
    cv::imshow("Image",image);

    //cv::namedWindow("Gray");
    //cv::imshow("Gray",gray_image);

    cv::namedWindow("Canny");
    cv::imshow("Canny",contours);
    cv::waitKey(0);
}