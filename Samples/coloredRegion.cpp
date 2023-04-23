#include <iostream>
#include<vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/core_c.h>

using namespace cv;

int main(int argc, char** argv ) {

    // <REPLACE THIS WITH YOUR PATH>
    Mat image = imread("");

    // KMEANS CODE GOES HERE
    //Setup for k means analysis
    Scalar colorTab[] = {
        Scalar(255, 0, 0),
        Scalar(0, 255, 0),
        Scalar(0, 0, 255)
    };

    // Convert to vector datatype for the color stuff later
    Vec3b colorVec3b[] = {
    Vec3b(colorTab[0][0], colorTab[0][1], colorTab[0][2]),
    Vec3b(colorTab[1][0], colorTab[1][1], colorTab[1][2]),
    Vec3b(colorTab[2][0], colorTab[2][1], colorTab[2][2])
    };

    //set up for kmeans
    int clustersCount = 3, iterations = 5;

    Mat samples(image.total(), 3, CV_32F);
    auto samples_ptr = samples.ptr<float>(0);
    
    for(int row = 0; row != image.rows; ++row) {
        auto src_begin = image.ptr<uchar>(row);
        auto src_end = src_begin + image.cols * image.channels();

        while(src_begin != src_end) {
            samples_ptr[0]= src_begin[0];
            samples_ptr[1]= src_begin[1];
            samples_ptr[2]= src_begin[2];
            samples_ptr+=3;
            src_begin+=3; 
        }
    }

    Mat labels, centers;
    kmeans(samples, clustersCount, labels, 
    TermCriteria( CV_TERMCRIT_ITER | TermCriteria::MAX_ITER|TermCriteria::EPS, 10, 0.01 ),
    iterations, KMEANS_PP_CENTERS, centers);
    //show clusters on image
    Mat clusteredImage( image.size(), image.type() );
    for( int row = 0; row != image.rows; ++row ){
        auto clusteredImageBegin = clusteredImage.ptr<uchar>(row);
        auto clusteredImageEnd = clusteredImageBegin + clusteredImage.cols * 3;
        auto labels_ptr = labels.ptr<int>(row * image.cols);
    
        //while the end of the image hasn't been reached...
        while( clusteredImageBegin != clusteredImageEnd ){
            //current label index:
            int const cluster_idx = *labels_ptr;
            //get the center of that index:
            auto centers_ptr = centers.ptr<float>(cluster_idx);
            clusteredImageBegin[0] = centers_ptr[0];
            clusteredImageBegin[1] = centers_ptr[1];
            clusteredImageBegin[2] = centers_ptr[2];
    
            clusteredImageBegin += 3; ++labels_ptr;
        }
    }

    // Display kmeans image before we feed it to Canny
    imshow("Kmeans Image", clusteredImage);
    // END OF KMEANS CODE

    // CANNY CODE STARTS HERE
    // Input kmeans matrix into canny
    cv::Mat contours;
    cv::Canny(clusteredImage,contours,5,200);
    // END OF CANNY CODE

    // SEGMENTING REGIONS AND FILLING WITH COLOR CODE
    for (int row = 0; row < clusteredImage.rows; row++) {
        for (int col = 0; col < clusteredImage.cols; col++) {
            int label = labels.at<int>(row * clusteredImage.cols + col, 0);
            clusteredImage.at<Vec3b>(row, col) = colorVec3b[label];
        }
    } // END OF COLOR CODE

    // Display original image
    cv::imshow("Original Image", image);

    // Display canny edges image
    cv::imshow("Canny image", contours);

    // Display segmented colored image
    cv::imshow("Canny colored Image", clusteredImage);

    waitKey(0);
    return EXIT_SUCCESS;
}