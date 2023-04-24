#include "kMeans.hpp"
#include "sobel.hpp"
#include<stdio.h>
#include<opencv2/opencv.hpp>


using namespace cv;

int main(int argc, char** argv){
    Mat image = imread(argv[1], IMREAD_GRAYSCALE);
    if(!image.data) { 
        fprintf(stderr, "Cannot read file\n");
        return 1;
    }

    Mat kimage = kMeans(image, 3,5, 2);
    Mat simage = sobel(kimage, 60);

    imshow("After Image", image);
    waitKey(0);
    return 0;
}