#include<iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    //Received matrix from Edge detection and the grey image
    Mat originalImage, greyImage, edgesImage;

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(
        edgesImage,
        contours,
        hierarchy,
        RETR_CCOMP,
        CHAIN_APPROX_SIMPLE
    );

    Mat markers(edgesImage.size(), CV_32S);
    markers = Scalar::all(0);

    int computationCounter = 0;
    for(int idx = 0; idx >=0; idx = hierarchy[idx][0], ++computationCounter){
        drawContours(
            markers, 
            contours, 
            idx, 
            Scalar::all(computationCounter + 1),
            -1, //Line Thickness
            8, //Line Type
            hierarchy,
            INT_MAX
            );
    }

    watershed(greyImage, markers);
    
    Mat watershedImage(markers.size(), CV_8UC3);

    for( int i = 0; i < markers.rows; i++ )
        for( int j = 0; j < markers.cols; j++ )
        {
            int index = markers.at<int>(i,j);
            if( index == -1 )
                watershedImage.at<Vec3b>(i,j) = Vec3b(255,255,255);
            else if( index <= 0 || index > computationCounter )
                watershedImage.at<Vec3b>(i,j) = Vec3b(0,0,0);
            else
                watershedImage.at<Vec3b>(i,j) = colorTab[index - 1];
        }
    watershedImage = watershedImage*0.5 + greyImage*0.5;

    imshow("Watershed Algorithm Results", watershedImage);
}