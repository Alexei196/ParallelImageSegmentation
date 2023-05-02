#include "kMeans.hpp"
#include "sobel.hpp"
#include "overlap.hpp"
#include<stdio.h>
#include<filesystem>
#include<string>
#include<sys/resource.h>
#include<time.h>
#include<sys/time.h>
#include<opencv2/opencv.hpp>

#define TIME(timeStruct) (double) timeStruct.tv_sec + (double) timeStruct.tv_usec * 0.000001

using namespace cv;
namespace fs = std::filesystem;
//Arg1 = Pathname of image folder
int main(int argc, char** argv){
    if(argc < 2) {
        fprintf(stderr, "Not enough arguments!\n");
        return 1;
    }
    //user designates folder path to read from 
    const fs::path imagesFolder{argv[1]};
    //load output folder in exe folder, create one if not present
    std::cout << "Entered folderpath is " << imagesFolder.filename().u8string() << std::endl;
    std:string outputFolderPath = imagesFolder.filename().u8string() + "_KMeans_Images";
    fs::create_directory(outputFolderPath);
    //For each directory_entry in specified folder
    struct timeval startTime;
    struct rusage resource_usage;
    gettimeofday(&startTime, NULL);
    for (auto const& imageEntry : fs::directory_iterator{imagesFolder}) {
        fs::path imagePath = imageEntry.path();
        //check for jpg
        if(imagePath.extension().u8string().compare(".jpg")) {
            std::cerr << "file " << imagePath.filename().u8string() << " is not a JPEG!" << std::endl; 
            continue;
        }
        //Do work on image
        Mat image = imread(imagePath.u8string(), IMREAD_GRAYSCALE);
        if(!image.data) { 
            std::cerr << "Cannot read file \"" << imagePath << "\"\n";
            continue;
        }

        Mat kimage = kMeans(image, 4,10, 2);
        
        getrusage(RUSAGE_SELF, &resource_usage);
        int reservedMemory = resource_usage.ru_maxrss;
        struct timeval timeNow;
        gettimeofday(&timeNow, NULL);

        Mat simage = sobel(kimage, 60);
        Mat overlappedImage = overlap(simage, image);
        //output store image result in folder
        cout << "Elaspsed time : " << TIME(timeNow) - TIME(startTime) << endl;
        cout << "Memory usage : " << reservedMemory << endl;
        std::string outputFilePath = outputFolderPath + "/" + imagePath.filename().u8string(); 
        if(!imwrite(outputFilePath, overlappedImage)) {
            std::cerr << "error writing to \"" << outputFilePath << "\"\n";
            continue;
        }
    }
    
    return 0;
}
