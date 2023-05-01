#include <stdio.h>
#include <iostream>
#include <filesystem>
#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <climits>
#include <limits.h>
#include <time.h>
#include <sys/resource.h>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

int brightness_distance(const int &l1, const int &l2);

int main(int argc, char **argv)
{
    int comm_sz, my_rank;
    const fs::path imagesFolder{argv[1]};
    std:string outputFolderPath;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    // Read folder
    if (my_rank == 0)
    {
        if (!fs::exists(imagesFolder))
        {
            fprintf(stderr, "Specified path does not exist!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        outputFolderPath = imagesFolder.filename().u8string() + "_KMeans_Images";
        fs::create_directory(outputFolderPath);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // foreach loop to look at each image
    for (auto const &imageFile : fs::directory_iterator{imagesFolder})
    {
        unsigned char * recvBuffer;
        unsigned char * sendBuffer;
        int centroidCount = 3, iterations = 7;
        int *centroids;
        int imageCount;
        int * displs = (int*)malloc(comm_sz * sizeof(int));
        int sectionSize;
        int * sectionSizePerThread = (int*)malloc(comm_sz * sizeof(int));
        long long int imageSize;
        Mat image;
        if(my_rank == 0) {
            image = imread(imageFile.path().u8string(), IMREAD_GRAYSCALE);
            imageSize = image.step[0] * image.rows;
            
            sectionSize = imageSize / comm_sz; // Broadcast this
            size_t remainder = imageSize - (sectionSize * comm_sz);
            imageCount++;
            // Displacements for MPI_Gatherv at the end
            displs[0] = 0;
            for(int i = 0; i < comm_sz; ++i) {
                sectionSizePerThread[i] = (i < remainder) ? sectionSize + 1 : sectionSize;
                displs[i] = (i <=remainder) ? (sectionSize+1)*i : (i*sectionSize) + remainder;
            }

            // fill this buffer with image pixels
            sendBuffer = image.data;
        }

        MPI_Bcast(&imageSize, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&sectionSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

        recvBuffer = (unsigned char *) malloc((imageSize) * sizeof(unsigned char));

        // init buffer for image buffer
        unsigned char *sectionBuffer = new unsigned char[imageSize];
        // distribute image data across the world

        MPI_Barrier(MPI_COMM_WORLD);       
        MPI_Scatterv(sendBuffer, sectionSizePerThread, displs, MPI_UNSIGNED_CHAR, sectionBuffer, imageSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);  
        MPI_Barrier(MPI_COMM_WORLD);
        centroids = (int *)malloc(centroidCount * sizeof(int));
        if (my_rank == 0)
        {
            for (int i = 0; i < centroidCount; ++i)
            {
                centroids[i] = (int)((rand() * (my_rank+1)) % 256);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        // For each iteration
        long long int * globalCentroidSum = (long long int *)malloc(centroidCount * sizeof(long long int));
        long long int * globalCentroidCounter = (long long int *)malloc(centroidCount * sizeof(long long int));
        long long int * localCentroidSum = (long long int *)malloc(centroidCount * sizeof(long long int));
        long long int * localCentroidCounter = (long long int *)malloc(centroidCount * sizeof(long long int));
        for (int iter = 0; iter < iterations; ++iter)
        {
            for(int i = 0; i < centroidCount; ++i) {
                globalCentroidSum[i] = 0;
                globalCentroidCounter[i] = 0;
                localCentroidSum[i] = 0;
                localCentroidCounter[i] = 0;
            }
            // broadcast centroids
            MPI_Bcast(centroids, centroidCount, MPI_INT, 0, MPI_COMM_WORLD);
            // for each pixel in buffer
            // #pragma omp parallel for num_threads(threadCount)
            
            for (size_t index = 0; index < sectionSize; ++index)
            {
                unsigned char *pixel = &sectionBuffer[index];

                // Step 1: get closest centroid of current pixel
                int current_pixel = *pixel;
                if(current_pixel < 12) {continue;}
                int closest_centroid = 0; // first centroid default is min

                int min_brightness_diff = INT_MAX;
                // Finds the closest centroid
                for (int centroid_index = 0; centroid_index < centroidCount; centroid_index++)
                {
                    // compute difference in brightness for the current pixel and each centroid
                    int current_brightness_diff = brightness_distance(current_pixel, centroids[centroid_index]);
                    if (current_brightness_diff < min_brightness_diff)
                    {
                        min_brightness_diff = current_brightness_diff;
                        closest_centroid = centroid_index;
                    }
                }

                // Step 2: add pixel value to centroid sum
                localCentroidSum[closest_centroid] += current_pixel;
                localCentroidCounter[closest_centroid] = localCentroidCounter[closest_centroid] + 1;
            }
            for (int centroid_index = 0; centroid_index < centroidCount; centroid_index++)
            {
                MPI_Reduce(&localCentroidSum[centroid_index], &globalCentroidSum[centroid_index], 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
                MPI_Reduce(&localCentroidCounter[centroid_index], &globalCentroidCounter[centroid_index], 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
            }
            // Step 3: after all pixels are added, calculate new centroid
            if(my_rank == 0) {
                for (int centroid_index = 0; centroid_index < centroidCount; centroid_index++)
                {
                    int new_centroid = globalCentroidSum[centroid_index] / globalCentroidCounter[centroid_index];
                    centroids[centroid_index] = new_centroid;
                } 
            }
        }

        // root collects results and outputs as image
        MPI_Barrier(MPI_COMM_WORLD);

        // Step 4: assign each pixel the value of its assigned centroid -> how to upscale to  0 to 255
        for (size_t index = 0; index < sectionSize; ++index)
            {
                unsigned char *pixel = &sectionBuffer[index];

                // Step 1: get closest centroid of current pixel
                int current_pixel = *pixel;
                if(current_pixel < 12) {continue;}
                int closest_centroid = centroids[0]; // first centroid default is min
                int closest_centroid_idx = 0;

                int min_brightness_diff = INT_MAX;
                // Finds the closest centroid
                for (int centroid_index = 0; centroid_index < centroidCount; centroid_index++)
                {
                    // compute difference in brightness for the current pixel and each centroid
                    int current_centroid = centroids[centroid_index];
                    int current_brightness_diff = brightness_distance(current_pixel, current_centroid);
                    if (current_brightness_diff < min_brightness_diff)
                    {
                        min_brightness_diff = current_brightness_diff;
                        closest_centroid = current_centroid;
                        closest_centroid_idx = centroid_index;
                    }
                }

                // assign each pixel the value of its closest centroid
                sectionBuffer[index] = (unsigned char) (closest_centroid_idx * (256 / centroidCount));
            }

        // Step 6: process 0 retrieves all

        MPI_Gatherv(sectionBuffer, sectionSize, MPI_UNSIGNED_CHAR, sendBuffer, sectionSizePerThread, displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

        if (my_rank == 0)
        {
            // Make Mat same size as original image
            //cv::Size size = image.size(); // get original size of image
            //cv::Mat outputMatrix(size, image.type()); // get new mat
            cv::Mat outputMatrix(image.rows, image.cols, image.type(), sendBuffer);

            // and output the image as jpg.
            std::string outputFilePath = outputFolderPath + "/" + imageFile.path().filename().u8string(); 
            if(!imwrite(outputFilePath, outputMatrix)) {
                std::cerr << "error writing to \"" << outputFilePath << "\"\n";
                continue;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}

int brightness_distance(const int &l1, const int &l2)
{
    return (l2 - l1) < 0 ? -1 * (l2 - l1) : (l2 - l1);
}

Mat sobel(const Mat &gray_img, int threshold)
{
    Mat sobel_img(gray_img.rows, gray_img.cols, CV_8UC1);
    if (threshold < 256)
    {
        threshold *= threshold;
    }

#pragma omp parallel for collapse(2)
    for (int row = 0; row < gray_img.rows; row++)
        for (int col = 0; col < gray_img.cols; col++)
        {
            if (row >= gray_img.rows - 2 || col >= gray_img.cols - 2)
            {
                sobel_img.at<unsigned char>(row, col) = 0;
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
            sobel_img.at<unsigned char>(row, col) = pixel;
        }
    return sobel_img;
}

Mat overlap(const Mat &sobel_img, const Mat &orig_image)
{

    // Convert sobel image type to same type as original image to run bitwise_or
    cv::Mat simg_16(sobel_img.rows, sobel_img.cols, CV_8UC3);
    int from_to[] = {0, 0, 0, 1, 0, 2};
    cv::mixChannels(&sobel_img, 1, &simg_16, 1, from_to, 3);
    cv::Mat sobel_updated;
    simg_16.convertTo(sobel_updated, CV_16U);

    cv::Mat overlappedImage;
    cv::bitwise_or(simg_16, orig_image, overlappedImage);
    return overlappedImage;
}
