#include<stdio.h>
#include<filesystem>
#include<openmpi/mpi.h>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;
using namespace cv;

int main(int argc, char** argv) {
    int comm_sz, my_rank;    
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    //Read folder, 
    if(my_rank == 0) {
        fs::path folderPath = new fs::path();
        if(!fs::exists(folderPath)) {
            fprintf(stderr, "Specified path does not exist!\n");
            MPI_Abort();
            return 1;
        } else {
            //TODO get directory iterator
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    //TODO setup foreach loop to look at each image
 
    //For each file in folder
    Mat image = imread(/*IMAGEPATH*/);

    //Split file into equal chunks specified by the amount of processes
    const int portions = comm_sz; 
    Mat imageSlices[portions];

    for(int i = 0; i < portions; ++i) {
        imageSlices[i] = Mat(Range((image.rows * i / comm_sz), (image.rows * (i+1) / comm_sz) - 1), Range(0, image.cols));
    }
    //for each chunk perform k-means
    //TODO kmeans must be split up to work between processes
    Mat kMeansImg = kMeans(imageSlices[my_rank], 3, 3, 4);
    printf("kmeans, %d", image.type());
    //after k means is finished, each process calls sobel on their chunk
    //Mat sobelImg = sobel(kMeansImg, 40);
    printf("sobel \n");
    imageSlices[my_rank] = kMeansImg;
    MPI_Barrier();

    //processes then merge their chunks together on rank 0 
    //TODO order concatinations
    if(my_rank != 0) {
        vconcat(imageSlices[0], imageSlices[my_rank], imageSlices[0]);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    //and output the image as jpg.   
    if(my_rank == 0) {
        char fileName[64];
        sprintf(fileName, "brainRegions%d.jpg", 1);
        imwrite(fileName, image);
    }
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