#include<iostream>
#include<filesystem>
#include<string>
#include<upcxx/upcxx.hpp>
#include<opencv2/opencv.hpp>
#include<sys/resource.h>
#include<time.h>
#include<sys/time.h>
#include "sobel.hpp"
#include "overlap.hpp"

#define IS_MASTER_PROCESS my_rank == 0
#define TIME(timeStruct) (double) timeStruct.tv_sec + (double) timeStruct.tv_usec * 0.000001

namespace fs = std::filesystem;
using namespace cv;

int distance(const unsigned char&, const unsigned char&);
long long int reduceToRank0(long long int);

int main(int argc, char** argv) {
    //init upc++
    upcxx::init();
    int comm_sz = upcxx::rank_n(), my_rank = upcxx::rank_me();
    
    if(argc < 2) {
        if(IS_MASTER_PROCESS) std::cerr << "Usage: upcKmeans <DIRPATH>\n";
        upcxx::finalize();
        return 1;
    }
    const fs::path imagesFolder{argv[1]};
    std::string outputFolderPath;
    
    if(!fs::exists(imagesFolder)) {
        if(IS_MASTER_PROCESS) std::cerr << "Cannot reach folder specified by path!" << std::endl;
        upcxx::finalize();
        return 1;
    }
    //designate folder of images and create folder for output
    
    if(IS_MASTER_PROCESS) {
        outputFolderPath = imagesFolder.filename().u8string() + "_KMeans_Images";
        fs::create_directory(outputFolderPath);
        srand(time(NULL));
    }
    //for each entry in the directory
    upcxx::global_ptr<unsigned char> globalImage_uchar = nullptr;
    upcxx::global_ptr<unsigned char> centroids = nullptr;
    upcxx::global_ptr<size_t> imageSize = nullptr;
    
    upcxx::barrier();
    struct timeval startTime;
    struct rusage resource_usage;
    gettimeofday(&startTime, NULL);
    for (auto const& imageEntry : fs::directory_iterator{imagesFolder}) {
        fs::path imagePath = imageEntry.path();
        Mat image;
        const int iterations = 7, clustersCount = 4;
        
        //master checks path for errors
        if(IS_MASTER_PROCESS) {
            //check for jpg
            if(imagePath.extension().u8string().compare(".jpg")) {
                std::cerr << "file " << imagePath.filename().u8string() << " is not a JPEG!" << std::endl; 
                continue;
            }
            //read in JPEG
            image = imread(imagePath.u8string(), IMREAD_GRAYSCALE);

            if(!image.data) { 
                std::cerr << "Cannot read file \"" << imagePath << "\"\n";
                continue;
            }
            
            imageSize = upcxx::new_<size_t>((size_t) image.step[0] * image.rows);
            globalImage_uchar = upcxx::new_array<unsigned char>( imageSize.local()[0]);
            centroids = upcxx::new_array<unsigned char>(clustersCount);

            for(int i = 0; i < clustersCount; ++i) {
                unsigned char randomChar = (unsigned char) (rand() * (i * 31)) % 256;
                upcxx::rput(randomChar, centroids + i).wait();
            }
            
            upcxx::rput(image.data, globalImage_uchar, imageSize.local()[0]).wait();
        }
        
        //broadcast the master determined variables
        globalImage_uchar = upcxx::broadcast(globalImage_uchar, 0).wait();
        centroids = upcxx::broadcast(centroids, 0).wait();
        imageSize = upcxx::broadcast(imageSize, 0).wait();

        unsigned char* localPixels = globalImage_uchar.local();
        unsigned char* localCentroids = centroids.local();

        size_t sectionSize = imageSize.local()[0]  / comm_sz;
        size_t remainder = imageSize.local()[0] - (sectionSize * comm_sz);

        long long int centroidSum[clustersCount], centroidCounter[clustersCount];
        int sectionStart = my_rank < remainder ? my_rank * (sectionSize+1) : (my_rank * sectionSize) + remainder;
        int sectionEnd = (my_rank < remainder) ? ((1 + my_rank) * sectionSize) + my_rank: ((1 + my_rank) * sectionSize) + remainder - 1;

        for(int iter = 0; iter < iterations; ++iter ) {
            //reset centroid sum and counter
            for(int i = 0; i < clustersCount; ++i) {
                centroidSum[i] = 0ll;
                centroidCounter[i] = 0;
            }
            //processes assign pixels to centroids
            for(int pIndex = sectionStart; pIndex < sectionEnd; ++pIndex) {
                int closestCentroidIndex = 0;
                int closestCentroidDistance = distance(localPixels[pIndex], localCentroids[closestCentroidIndex]);
                for(int centroidIndex = 0; centroidIndex < clustersCount; ++centroidIndex) {
                    int centroidDistance = distance(localPixels[pIndex], localCentroids[centroidIndex]);
                    if(centroidDistance < closestCentroidDistance) {
                        closestCentroidIndex = centroidIndex;
                        closestCentroidDistance = centroidDistance;
                    }
                }
                centroidSum[closestCentroidIndex] += (int) localPixels[pIndex];
                centroidCounter[closestCentroidIndex] += 1;
            }
            //processes share their centroid sums and counts via reduction
            for(int i = 0; i < clustersCount; ++i) {
                long int sums = 0ll, counts = 0ll;
                upcxx::future<long long int> sumFuture = upcxx::reduce_all(centroidSum[i], upcxx::op_fast_add);
                upcxx::future<long long int> countFuture = upcxx::reduce_all(centroidCounter[i], upcxx::op_fast_add);
                //skip 0 counter centroids
                if(IS_MASTER_PROCESS) {
                    if(counts == 0l) {
                        continue;
                    }                    
                    localCentroids[i] = (unsigned char) ((double)sumFuture.wait() / countFuture.wait());
                }
            }
        }
        //After all iterations, pass once more over the array to assign pixels their centroids
        for(int pIndex = sectionStart; pIndex < sectionEnd; ++pIndex) {
            int closestCentroidIndex = 0;
            int closestCentroidDistance = distance(localPixels[pIndex], localCentroids[closestCentroidIndex]);
            for(int centroidIndex = 1; centroidIndex < clustersCount; ++centroidIndex) {
                int centroidDistance = distance(localPixels[pIndex], localCentroids[centroidIndex]);
                if(centroidDistance < closestCentroidDistance) {
                    closestCentroidIndex = centroidIndex;
                    closestCentroidDistance = centroidDistance;
                }
            }
            localPixels[pIndex] = (unsigned char) ((closestCentroidIndex + 1) * (255 / clustersCount));
        }
        //after all iterations complete, return a new image of the assigned pixels
        upcxx::barrier();
        getrusage(RUSAGE_SELF, &resource_usage);
        struct timeval timeNow;
        int local_r = resource_usage.ru_maxrss;
        gettimeofday(&timeNow, NULL);
        double timeCount = TIME(timeNow);
        double global_r = upcxx::reduce_all(local_r, upcxx::op_fast_max).wait();
        double global_timeNow = upcxx::reduce_all(timeCount, upcxx::op_fast_max).wait();
        if(IS_MASTER_PROCESS) {
            cout << "Elaspsed time : " << global_timeNow - TIME(startTime) << endl;
            cout << "Memory usage : " << global_r << endl;
            Mat kmeansImage = Mat(image.rows, image.cols, CV_8UC1, localPixels);
            //apply sobel and overlay on original image and write to folder
            Mat sobelImage = sobel(kmeansImage, 60);

            Mat returnImage = overlap(sobelImage, image);

            std::string outputFilePath = outputFolderPath + "/" + imagePath.filename().u8string();
            if(!imwrite(outputFilePath, returnImage)) {
                std::cerr << "Could not write image to \"" << outputFolderPath << "\"" << endl;
                continue;
            } 
            upcxx::delete_(imageSize);
            upcxx::delete_array(globalImage_uchar);
            upcxx::delete_array(centroids);
        }
        
    }
    upcxx::finalize();
    return 0;
}

//helper function distance between points l1 and l2
int distance(const unsigned char &l1, const unsigned char &l2) {
    return (int) (l2 - l1) < 0 ? (int) -1*(l2-l1) : (int) (l2-l1);
}