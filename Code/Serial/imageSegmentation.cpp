#include "kMeans.hpp"
#include "sobel.hpp"
#include "overlap.hpp"
#include<stdio.h>
#include<filesystem>
#include<string>
#include<opencv2/opencv.hpp>

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
    char outputFolderName[128]; 
    sprintf(outputFolderName, "%s_KMeans_Images", imagesFolder.filename().u8string() );
    fs::create_directory(outputFolderName);
    //For each directory_entry in specified folder
    for (auto const& imageEntry : fs::directory_iterator{imagesFolder}) {
        fs::path imagePath = imageEntry.path();
        std::cout << imagePath << '\n';
        //check for jpg
        if(imagePath.extension().u8string().compare(".jpg")) {
            fprintf(stderr, "file %s is not a JPEG!\n", imagePath.filename());
            continue;
        }
        //Do work on image
        Mat image = imread(imagePath.u8string(), IMREAD_GRAYSCALE);
        if(!image.data) { 
            fprintf(stderr, "Cannot read file \"%s\"\n", imagePath);
            continue;
        }

        Mat kimage = kMeans(image, 3,5, 2);
        Mat simage = sobel(kimage, 60);
        Mat overlappedImage = overlap(simage, image);
        //output store image result in folder
        char outputImagePath[128];
        sprintf(outputImagePath, "%s/%s.jpg", outputFolderName, imagePath.filename());
        if(!imwrite(outputImagePath, overlappedImage) ) {
            fprintf(stderr, "error writing \"%s\"", outputImagePath);
            continue;
        }
    }
    
    return 0;
}
