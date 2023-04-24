#include "kMeans.hpp"
#include "sobel.hpp"
#include<stdio.h>
#include<opencv2/opencv.hpp>


using namespace cv;

int main(int argc, char** argv){
    Mat image = imread(argv[1]);
    if(!image.data) { 
        fprintf(stderr, "Cannot read file\n");
        return 1;
    }

    image = kMeans(image, 3, 3, 2);
    
    namedWindow("IMage", WINDOW_AUTOSIZE);
    imshow("After Image", image);
    waitKey(0);
    return 0;
}