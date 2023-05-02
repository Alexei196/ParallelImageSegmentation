#include<iostream>
#include<vector>
#include<omp.h>
#include<opencv2/opencv.hpp>

using namespace cv;

int distance(const int &l1, const int &l2) {
    return (l2 - l1) < 0 ? -1*(l2-l1) : (l2-l1);
}

Mat kMeans(const Mat& image, const int& clustersCount, const int& iterations, int threadCount) {
    //1. Define random centroids for k clusters
    Mat centroidAssigned;
    long long int centroidSum[clustersCount];
    int centroids[clustersCount], centroidCount[clustersCount];
    for(int i = 0; i < clustersCount; ++i) {
        centroids[i] = (int) (rand() % 256);
        centroidSum[i] = 0ll;
        centroidCount[i] = 0;
    }

    //2. Assign data to closest centroid
    centroidAssigned = Mat(image.rows, image.cols, CV_8UC1);
    int lowestDistance, closestCentroid, c;
    int colorScale = 255 / (clustersCount);
    long int y, x;
    //For each iteration of the k-means alg
    for(int i = 0; i < iterations; ++i) {
        //For each pixel in image
        #pragma omp parallel for num_threads(threadCount) \
        default(none) shared(image, centroidAssigned, colorScale, centroids, centroidSum, centroidCount, clustersCount) private(y,x, c, closestCentroid, lowestDistance) 
        for(y = 0; y < image.rows; ++y) {
            for(x = 0; x < image.cols; ++x) {
                // option for centroids to ignore all low value/black pixels
                if((int)image.at<unsigned char>(y,x) < 12) {
                    centroidAssigned.at<unsigned char>(y,x) = (unsigned char) 0;
                }
                //For each centroid in existence
                closestCentroid = 0;
                lowestDistance = distance((int)image.at<unsigned char>(y,x), centroids[0]);
                for(c = 1; c < clustersCount; ++c) {
                    int space = distance((int)image.at<unsigned char>(y,x), centroids[c]);
                    if(space < lowestDistance) {
                        closestCentroid = c;
                        lowestDistance = space;
                    }
                }
                //Now that centroids are found, replace the pixels with the color of the centroid
                centroidAssigned.at<unsigned char>(y,x) = (unsigned char) (closestCentroid+1) * colorScale;
                centroidSum[closestCentroid] += (long long int)image.at<unsigned char>(y,x);
                centroidCount[closestCentroid] += 1;
            }
        }
        //3. Assign centroid to the average of each grouped data
        for(int c = 0; c < clustersCount; ++c) {
            if(centroidCount[c] == 0) {
                //In event centroid is not counted
                fprintf(stderr, "Centroid %d did not gain any points!\n", c);
                continue;
            }
            centroids[c] = (int) (centroidSum[c] / (long long int) centroidCount[c]);
        }
    }
    //4. perform 2 and 3 i amount of times   
    return centroidAssigned;
}