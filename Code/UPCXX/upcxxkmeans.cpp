#include<iostream>
#include<filesystem>
#include<string>
#include<upcxx/upcxx.hpp>
#include<opencv2/opencv.hpp>
#include "sobel.hpp"
#include "overlap.hpp"

#define IS_MASTER_PROCESS my_rank == 0

namespace fs = std::filesystem;
using namespace cv;

int distance(const unsigned char&, const unsigned char&);
long long int reduceToRank0(long long int);

int main(int argc, char** argv) {
    //init upc++
    upcxx::init();
    int comm_sz = upcxx::rank_n(), my_rank = upcxx::rank_me();
    std::string outputFolderPath;
    if(argc < 2) {
        if(my_rank == 0) std::cerr << "Usage: upcKmeans <DIRPATH>\n";
        upcxx::finalize();
        return 1;
    }
    //designate folder of images and create folder for output
    const fs::path imagesFolder{argv[1]};
    if(IS_MASTER_PROCESS) {
        outputFolderPath = imagesFolder.filename().u8string() + "_KMeans_Images";
        fs::create_directory(outputFolderPath);
    }
    //TODO ensure each process has access to imageEntry
    //for each entry in the directory
    std::cout << "Looking through the folder...\n";
    for (auto const& imageEntry : fs::directory_iterator{imagesFolder}) {
        fs::path imagePath = imageEntry.path();

        const int iterations = 7, clustersCount = 3;
        upcxx::global_ptr<unsigned char> globalImage_uchar = nullptr;
        upcxx::global_ptr<int> centroids = nullptr;
        size_t imageSize;
        size_t* localImageSize;
        //master checks path for errors
        if(IS_MASTER_PROCESS) {
            //check for jpg
            if(imagePath.extension().u8string().compare(".jpg")) {
                std::cerr << "file " << imagePath.filename().u8string() << " is not a JPEG!" << std::endl; 
                continue;
            }
            //read in JPEG
            Mat image = imread(imagePath.u8string(), IMREAD_GRAYSCALE);
            if(!image.data) { 
                std::cerr << "Cannot read file \"" << imagePath << "\"\n";
                continue;
            }
            //TODO share imagesize between processes
            imageSize = image.step[0] * image.rows;
            std::cout << "imageSize : " << imageSize << std::endl;
            imageGlobalSize = upcxx::new_<size_t>(imageSize);

            centroids = upcxx::new_array<int>(clustersCount);
            for(int i = 0; i < clustersCount; ++i) {
                int randomNumber = (int) rand() % 256;
                upcxx::rput(randomNumber, centroids + i);
                std::cout << "centroid : " << randomNumber << std::endl;
            }

            globalImage_uchar = upcxx::new_array<unsigned char>( imageSize);
            upcxx::rput(image.data, globalImage_uchar, imageSize);
            std::cout << "Read the image " << imagePath << std::endl;
        }
        //Do kmeans
        imageGlobalSize = upcxx::broadcast(imageGlobalSize, 0).wait();
        size_t sectionSize = ((size_t) localImageSize) / comm_sz;
        size_t remainder = ((size_t)localImageSize) - (sectionSize * comm_sz);
        globalImage_uchar = upcxx::broadcast(globalImage_uchar, 0).wait();
        //distribute centroids
        centroids = upcxx::broadcast(centroids, 0).wait();
        
        unsigned char* localPixels = globalImage_uchar.local();
        int* localCentroids = centroids.local();
        long long int centroidSum[clustersCount], centroidCounter[clustersCount];
        int sectionStart = my_rank * sectionSize;
        int sectionEnd = my_rank =< remainder ? ((1 + my_rank) * sectionSize) + 1 : ((1 + my_rank) * sectionSize) + 2;
        upcxx::barrier();
        std::cout << "starting iterations... \n";
        for(int iter = 0; iter < iterations; ++iter ) {
            //reset centroid sum and counter
            std::cout << "Centroids:" << std::endl;
            for(int i = 0; i < clustersCount; ++i) {
                centroidSum[i] = 0ll;
                centroidCounter[i] = 0;
                std::cout << " " << localCentroids[i];
            }

            std::cout << std::endl << "init centroid sums and counters\n";
            //processes assign pixels to centroids
            for(int pIndex = sectionStart; pIndex < sectionEnd; ++pIndex) {
                std::cout << "considering pixel " << pIndex << std::endl; 
                int closestCentroidIndex = 0;
                int closestCentroidDistance = distance(localPixels[pIndex], localCentroids[closestCentroidIndex]);
                for(int centroidIndex = 0; centroidIndex < clustersCount; ++centroidIndex) {
                    int centroidDistance = distance(localPixels[pIndex], localCentroids[centroidIndex]);
                    if(centroidDistance < closestCentroidDistance) {
                        closestCentroidIndex = centroidIndex;
                        closestCentroidDistance = centroidDistance;
                    }
                }
                std::cout << "Centroid assigned to " << closestCentroidIndex << std::endl;
                //TODO assign centroid
                centroidSum[closestCentroidIndex] += (int) localPixels[pIndex];
                centroidCounter[closestCentroidIndex] += (int) localPixels[pIndex];
            }
            //processes share their centroid sums and counts via reduction
            for(int i = 0; i < clustersCount; ++i) {
                long int sums = 0ll, counts = 0ll; 
                sums = upcxx::reduce_all(centroidSum[i], upcxx::op_fast_add).wait();
                counts = upcxx::reduce_all(centroidCounter[i], upcxx::op_fast_add).wait();
                localCentroids[i] = sums / counts;
            }
            std::cout << "assigned centroids for iter " << iter << std::endl;
        }
        //after all iterations complete, return a new image of the assigned pixels

        //apply sobel and overlay on original image and 

        //TODO masterprocess writes image to file
        
    }

    upcxx::barrier();

    upcxx::finalize();
    return 0;
}

//helper function distance between points l1 and l2
int distance(const unsigned char &l1, const unsigned char &l2) {
    return (int) (l2 - l1) < 0 ? (int) -1*(l2-l1) : (int) (l2-l1);
}