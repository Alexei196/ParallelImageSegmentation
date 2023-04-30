#include<iostream>
#include<filesystem>
#include<string>
#include<upcxx/upcxx.h>

#define IS_MASTER_PROCESS my_rank == 0

namespace fs = std::filesystem;
using namespace cv;

int distance(const unsigned char&, const unsigned char&);

int main(int argc, char** argv) {
    //init upc++
    upcxx::upcxx_init();
    int comm_sz = upcxx::rank_n(), my_rank = upcxx::rank_me();
    std:string outputFolderPath;
    //designate folder of images and create folder for output
    const fs::path imagesFolder{argv[1]};
    if(IS_MASTER_PROCESS) {
        outputFolderPath = imagesFolder.filename().u8string() + "_KMeans_Images";
        fs::create_directory(outputFolderPath);
    }
    //TODO ensure each process has access to imageEntry
    //for each entry in the directory
    for (auto const& imageEntry : fs::directory_iterator{imagesFolder}) {
        fs::path imagePath = imageEntry.path();
        size_t imageSize;
        size_t sectionSize;
        size_t remainder;
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
            imageSize = image.step[0] * image.rows;
            sectionSize = imageSize / comm_sz;
            remainder = imageSize - (sectionSize * comm_sz);
        }
        //Do kmeans
        const int iterations = 7, clustersCount = 3;
        //TODO Distribute image bytes

        upcxx::global_ptr<unsigned char> globalImage_uChar = nullptr;
        if(my_rank == 0) {
            globalImage_uchar = upcxx::new_array<unsigned char>(imageSize);
            for(int i = 0; i < image.size(); ++i) {
                upcxx::rput(image.data[i], globalImage_uchar + i );
            }
        }
        globalImage_uchar = upcxx::broadcast(centroids, 0).wait();
        //distribute centroids
        upcxx::global_ptr<unsigned char> centroids = nullptr;
        if(my_rank == 0) {
            centroids = upcxx::new_array<unsigned char>(clustersCount);
            for(int i = 0; i < clustersCount; ++i) {
                upcxx::rput((unsigned char) rand() % 256, centroids + i);
            }
        }
        centroids = upcxx::broadcast(centroids, 0).wait();
        
        unsigned char* localPixels = globalImage_uchar.local();
        unsigned char* localCentroids = centroids.local();
        long long int centroidSum[clustersCount], centroidCounter[clustersCount];
        int sectionStart = my_rank * sectionSize;
        int sectionEnd = my_rank < remainder ? ((1 + my_rank) * sectionSize) + 1 : ((1 + my_rank) * sectionSize) + 2;

        for(int iter = 0; iter < iterations; ++iter ) {
            //distribute k centroids to processes
            
            
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
                    int centroidDistance = distance(pixel, localCentroids[centroidIndex]);
                    if(centroidDistance < closestCentroidDistance) {
                        closestCentroidIndex = centroidIndex;
                        closestCentroidDistance = centroidDistance;
                    }
                }
                //TODO assign centroid
                centroidSum[closestCentroidIndex] += (int) pixel;
                centroidCounter[closestCentroidIndex] += (int) pixel;
            }
            //processes share their centroid sums and counts via reduction
            for(int i = 0; i < clustersCount; ++i) {
                localCentroids[i] = (unsigned char) (reduceToRank0(centroidSum[i]) / reduceToRank0(centroidCounter[i])); 
            }
        }
        //after all iterations complete, return a new image of the assigned pixels

        //apply sobel and overlay on original image and 

        //masterprocess writes image to file
        if(IS_MASTER_PROCESS) {
            std::string outputFilePath = outputFolderPath + "/" + imagePath.filename().u8string(); 
            if(!imwrite(outputFilePath, simage)) {
                std::cerr << "error writing to \"" << outputFilePath << "\"\n";
                continue;
            }
        }
    }

    upcxx::barrier();

    upcxx::upcxx_finalize();
    return 0;
}

//helper function distance between points l1 and l2
int distance(const unsigned char &l1, const unsigned char &l2) {
    return (int) (l2 - l1) < 0 ? (int) -1*(l2-l1) : (int) (l2-l1);
}

void upcxxKmeans(int clusters, int iterations) {

}

long long int reduceToRank0(long long int count) {
    return upcxx::allreduce(count, plus<long long int>()).wait();
}