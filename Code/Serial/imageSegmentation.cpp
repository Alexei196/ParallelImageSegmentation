#include "kMeans.hpp"
#include "sobel.hpp"
#include<stdio.h>

using namespace cv;

int main(int argc, char** argv){
    Mat image = imread(argv[1]);
    if(image == 0) { 
        fprintf(stderr, "Cannot read file1\n");
        return 1;
    }

    image = kMeans(image, 3, 3, 2);

    imshow(image);
    waitkey(0);
    return 0;
}