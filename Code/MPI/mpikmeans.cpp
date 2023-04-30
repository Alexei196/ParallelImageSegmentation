#include <stdio.h>
#include <iostream>
#include <filesystem>
#include <openmpi/mpi.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <climits>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    int comm_sz, my_rank;
    fs::path folderPath;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    // Read folder
    if (my_rank == 0)
    {
        folderPath = argv[1];
        if (!fs::exists(folderPath))
        {
            fprintf(stderr, "Specified path does not exist!\n");
            MPI_Abort();
            return 1;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // foreach loop to look at each image
    for (auto const &imageFile : fs::directory_iterator{folderPath})
    {

        Mat image = imread(imageFile);
        std::cout << "Channels: " << image.channels() << std::endl;
        size_t imageSize = image.step[0] * image.rows;
        size_t sectionSize = imageSize / comm_sz;
        size_t remainder = imageSize - (sectionSize * comm_sz);
        int centroidCount = 3, iterations = 7;
        int *centroids;

        if (my_rank <= remainder)
        {
            sectionSize += 1;
        }
        // init buffer for image buffer
        unsigned char *sectionBuffer = malloc(sectionSize * sizeof(unsigned char));
        // distribute image data across the world
        MPI_Scatterv(image.data, sectionSize, MPI_UNSIGNED_CHAR, sectionBuffer, sectionSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

        if (my_rank == 0)
        {
            centroids = (int *)malloc(centroidCount * sizeof(int));
            for (int i = 0; i < centroidCount; ++i)
            {
                centroids[i] = (int)(rand() % 256);
            }
        }

        

        // For each iteration
        for (int iter = 0; iter < iterations; ++iter)
        {
            long long int globalCentroidSum[] = new long long int[centroidCount];
            long long int globalCentroidCounter[] = new long long int[centroidCount];

            // broadcast centroids
            MPI_Bcast(centroids, centroidCount, MPI_INT, 0, MPI_COMM_WORLD);
            // for each pixel in buffer
            // #pragma omp parallel for num_threads(threadCount)
            long long int localCentroidSum[centroidCount];
            long long int localCentroidCounter[centroidCount];
            for (size_t index = 0; index < sectionSize; ++index)
            {
                unsigned char *pixel = &sectionBuffer[index];

                // Step 1: get closest centroid of current pixel
                int current_pixel = *pixel;
                if(current_pixel < 12) {continue;}
                int closest_centroid = centroids[0]; // first centroid default is min

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
                    }
                }

                // Step 2: add pixel value to centroid sum
                localCentroidSum[closest_centroid] += current_pixel;
                localCentroidCounter[closest_centroid]++;
            }

            for (int centroid_index = 0; centroid_index < centroidCount; centroid_index++)
            {
                MPI_Reduce(&localCentroidSum[centroid_index], &globalCentroidSum[centroid_index], 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
                MPI_Reduce(&localCentroidCounter[centroid_index], &globalCentroidCounter[centroid_index], 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
            }
            // Step 3: after all pixels are added, calculate new centroid
            for (int centroid_index = 0; centroid_index < centroidCount; centroid_index++)
            {
                int new_centroid = globalCentroidSum[centroid_index] / globalCentroidCounter[centroid_index];
                centroids[centroid_index] = new_centroid;
            } 
        }

        // root collects results and outputs as image
        MPI_Barrier(MPI_COMM_WORLD);

        // Step 4: Make Mat same size as original image
        cv::Size size = image.size(); // get original size of image
        cv::Mat outputMatrix(size, image.type()); // get new mat

        // Step 5: assign each pixel the value of its assigned centroid -> how to upscale to  0 to 255

        // Step 6: process 0 retrieves all
        // for each pixel in output Matrix, assign each pixel it's centroid
        unsigned char * recvBuffer = malloc(sectionSize * sizeof(unsigned char));

        // TODO: 
        // we need to figure out the displacement array for gatherv
        int MPI_Gatherv(sectionBuffer, sectionSize, MPI_UNSIGNED_CHAR, recvBuffer, sectionSize, , MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

        if (my_rank == 0)
        {
            // Step 6: assign each pixel a the centroid value

            // and output the image as jpg.
            char fileName[64];
            sprintf(fileName, "brainRegions%d.jpg", imageCount++);
            imwrite(fileName, image);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    // processes then merge their chunks together on rank 0
    // TODO order concatinations
    MPI_Finalize();
    return 0;
}

int brightness_distance(const int &l1, const int &l2)
{
    return (l2 - l1) < 0 ? -1 * (l2 - l1) : (l2 - l1);
}

Mat kMeans(const Mat &image, const int &clustersCount, const int &iterations, int threadCount)
{
    // 1. Define random centroids for k clusters
    long long int centroidSum[clustersCount];
    int centroids[clustersCount], centroidCount[clustersCount];
    for (int i = 0; i < clustersCount; ++i)
    {
        centroids[i] = (int)(rand() % 256);
        centroidSum[i] = 0ll;
        centroidCount[i] = 0;
    }

    // 2. Assign data to closest centroid
    Mat centroidAssigned = image.clone();
    int lowestDistance, closestCentroid, c;
    int colorScale = 256 / (clustersCount);
    long int y, x;
    // For each iteration of the k-means alg
    for (int i = 0; i < iterations; ++i)
    {
// For each pixel in image
#pragma omp parallel for num_threads(threadCount) default(none) shared(image, centroidAssigned, colorScale, centroids, centroidSum, centroidCount, clustersCount) private(y, x, c, closestCentroid, lowestDistance)
        for (y = 0; y < image.rows; ++y)
        {
            if (omp_get_thread_num() > 0)
                printf("Thread %d reporting\n", omp_get_thread_num());
            for (x = 0; x < image.cols; ++x)
            {
                // option for centroids to ignore all low value/black pixels
                if ((int)image.at<unsigned char>(y, x) < 24)
                {
                    // continue;
                }
                // For each centroid in existence
                closestCentroid = 0;
                lowestDistance = distance((int)image.at<unsigned char>(y, x), centroids[0]);
                for (c = 1; c < clustersCount; ++c)
                {
                    int space = distance((int)image.at<unsigned char>(y, x), centroids[c]);
                    if (space < lowestDistance)
                    {
                        closestCentroid = c;
                        lowestDistance = space;
                    }
                }
                // Now that centroids are found, replace the pixels with the color of the centroid
                centroidAssigned.at<unsigned char>(y, x) = closestCentroid * colorScale;
                centroidSum[closestCentroid] += (long long int)image.at<unsigned char>(y, x);
                centroidCount[closestCentroid] += 1;
            }
        }
        // 3. Assign centroid to the average of each grouped data
        for (int c = 0; c < clustersCount; ++c)
        {
            if (centroidCount[c] == 0)
            {
                // In event centroid is not counted
                fprintf(stderr, "Centroid %d did not gain any points!\n", c);
                continue;
            }
            centroids[c] = (long long int)(centroidSum[c] / (long long int)centroidCount[c]);
        }
    }
    // 4. perform 2 and 3 i amount of times
    return centroidAssigned;
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
