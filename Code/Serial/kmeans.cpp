#include <iostream>
#include<vector>
#include <opencv2/opencv.hpp>

using namespace cv;

int distance(const int&, const int&);

int main(int argc, char** argv ) {
    if ( argc != 2 ) {
        printf("usage: kmeans.exe <Image_Path>\n");
        return -1;
    }

    //load image into matrix
    Mat image = imread( argv[1], 1 );
    if ( !image.data ) {
        printf("No image data \n");
        return -1;
    }
    //Setup for k means analysis
    Scalar colorTab[] = {
        Scalar(255, 0, 0),
        Scalar(0, 255, 0),
        Scalar(0, 0, 255)
    };
    //set up for kmeans
    const int clustersCount = 3, iterations = 10;

    //1. Define random centroids for k clusters
    int centroids[clustersCount];
    long long int centroidSum[clustersCount];
    int centroidCount[clustersCount];
    for(int i = 0; i < clustersCount; ++i) {
        centroids[i] = rand() % 256;
        centroidSum[i] = 0;
        centroidCount[i] = 0;
    }
    //2. Assign data to closest centroid
    int lowestDistance = 65536, closestCentroid = 0;
    //For each iteration of the k-means alg
    for(int i = 0; i < iterations; ++i) {
        //For each pixel in image
        for(int y = 0; y < image.rows(); ++y) {
            for(int x = 0; x < image.cols(); ++x) {     
                //For each centroid in existence
                //TODO reset distance and closest centroid
                for(int c = 0; c < clustersCount; ++c) {
                    if(distance((int)image.at<int>(y,x), centroids[c]) < lowestDistance) {
                        closestCentroid = c;
                        lowestDistance = distance((int)image.at<int>(y,x), centroids[c]);
                    }
                }
                centroidSum[closestCentroid] += (long long int)image.at<int>(y,x);
                centroidCount[closestCentroid] += 1;
            }
        }
        //3. Assign centroid to average of each grouped data
        for(int c = 0; c < clustersCount; ++c) {
            centroids[c] = centroidSum[c] / centroidCount[c];
        }
    }
     //4. perform 2 and 3 i amount of times   

    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", clusteredImage);

    waitKey(0);
    return 0;
}

int distance(const int &l1, const int &l2) {
    return l2 - l1;
}