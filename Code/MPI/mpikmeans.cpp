#include<stdio.h>
#include<iostream>
#include<filesystem>
#include<openmpi/mpi.h>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;
using namespace cv;

int main(int argc, char** argv) {
    int comm_sz, my_rank;
    fs::path folderPath;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    //Read folder, 
    if(my_rank == 0) {
        folderPath = argv[1];
        if(!fs::exists(folderPath)) {
            fprintf(stderr, "Specified path does not exist!\n");
            MPI_Abort();
            return 1;
        } else {
            //found folder result
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    //foreach loop to look at each image
    for(auto const& imageFile : fs::directory_iterator{folderPath}) {
        Mat image = imread(imageFile);
        std::cout << "Channels: " << image.channels() << std::endl;
        size_t imageSize = image.step[0] * image.rows;
        size_t sectionSize = imageSize / comm_sz;
        size_t remainder = imageSize - (sectionSize * comm_sz);
        int centriodCount = 3, interations = 7;
        int* centroids;

        if(my_rank <= remainder) {
            sectionSize +=1;
        }
        //init buffer for image buffer
        unsigned char* sectionBuffer = malloc(sectionSize*sizeof(unsigned char));
        //distribute image data across the world
        MPI_Scatter(image.data, sectionSize, MPI_UNSIGNED_CHAR, sectionBuffer, sectionSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
        
        if(my_rank == 0) {
            centroids = (int*) malloc(centriodCount * sizeof(int)); 
            for(int i = 0; i < centriodCount; ++i) {
                centroids[i] = (int) (rand() % 256);
            }
        }
        //For each iteration
        for(int iter = 0; iter < iterations; ++iter) {
            //broadcast centroids
            MPI_Bcast(); 
            //for each pixel in buffer
            for(size_t index = 0; index < sectionSize; ++index) {
                unsigned char* pixel = &sectionBuffer[index];
                //assign to a centroid
                
            }

        }
        //After the iterations, assign the pixels to their centroids and return to root

        //root collects results and outputs as image
        MPI_Barrier(MPI_COMM_WORLD);
        if(my_rank == 0) {
            //and output the image as jpg.   
            char fileName[64];
            sprintf(fileName, "brainRegions%d.jpg", imageCount++);
            imwrite(fileName, image);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    //processes then merge their chunks together on rank 0 
    //TODO order concatinations
    MPI_Finalize();
    return 0;
}

int distance(const int &l1, const int &l2) {
    return (l2 - l1) < 0 ? -1*(l2-l1) : (l2-l1);
}

Mat kMeans(const Mat& image, const int& clustersCount, const int& iterations, int threadCount) {
    //1. Define random centroids for k clusters
    long long int centroidSum[clustersCount];
    int centroids[clustersCount], centroidCount[clustersCount];
    for(int i = 0; i < clustersCount; ++i) {
        centroids[i] = (int) (rand() % 256);
        centroidSum[i] = 0ll;
        centroidCount[i] = 0;
    }

    //2. Assign data to closest centroid
    Mat centroidAssigned = image.clone();
    int lowestDistance, closestCentroid, c;
    int colorScale = 256 / (clustersCount);
    long int y, x;
    //For each iteration of the k-means alg
    for(int i = 0; i < iterations; ++i) {
        //For each pixel in image
        #pragma omp parallel for num_threads(threadCount) \
        default(none) shared(image, centroidAssigned, colorScale, centroids, centroidSum, centroidCount, clustersCount) private(y,x, c, closestCentroid, lowestDistance) 
        for(y = 0; y < image.rows; ++y) {
            if(omp_get_thread_num() > 0) printf("Thread %d reporting\n", omp_get_thread_num());
            for(x = 0; x < image.cols; ++x) {
                // option for centroids to ignore all low value/black pixels
                if((int)image.at<unsigned char>(y,x) < 24) {
                    //continue;
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
                centroidAssigned.at<unsigned char>(y,x) = closestCentroid * colorScale;
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
            centroids[c] = (long long int) (centroidSum[c] / (long long int) centroidCount[c]);
        }
    }
    //4. perform 2 and 3 i amount of times   
    return centroidAssigned;
}

Mat sobel(const Mat &gray_img, int threshold) {
    Mat sobel_img(gray_img.rows, gray_img.cols, CV_8UC1);
    if(threshold < 256) {
        threshold*= threshold;
    }
    #include <stdio.h>
#include <opencv2/opencv.hpp>
#include <omp.h>

using namespace cv;
using namespace std;


Mat sobel(const Mat &gray_img, int threshold) {
    Mat sobel_img(gray_img.rows, gray_img.cols, CV_8UC1);
    if(threshold < 256) {
        threshold*= threshold;
    }
#pragma omp parallel for collapse(2)
    for (int row = 0; row < gray_img.rows; row++)
        for (int col = 0; col < gray_img.cols; col++)
        {
            if (row >= gray_img.rows - 2 || col >= gray_img.cols - 2){
                sobel_img.at<unsigned char>(row,col) = 0;
                continue;
            }
            Mat G;
            gray_img(cv::Rect(col, row, 3, 3)).copyTo(G);
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
            sobel_img.at<unsigned char>(row,col) = pixel;
             
        }
    return sobel_img;
}
    for (int row = 0; row < gray_img.rows; row++)
        for (int col = 0; col < gray_img.cols; col++)
        {
            if (row >= gray_img.rows - 2 || col >= gray_img.cols - 2){
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
                pixel = 255;
            sobel_img.at<unsigned char>(col,row) = pixel;
        }
    return sobel_img;
}
