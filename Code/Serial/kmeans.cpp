#include <iostream>
#include<vector>
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char** argv ) {
    if ( argc != 2 ) {
        printf("usage: kmeans.exe <Image_Path>\n");
        return -1;
    }

    //load image into matrix
    Mat image = imread( argv[1], 1 );
    if ( !image.data ) {
        printf("No image data \n");
        return -1;
    }
    //Setup for k means analysis
    Scalar colorTab[] = {
        Scalar(255, 0, 0),
        Scalar(0, 255, 0),
        Scalar(0, 0, 255)
    };
    //set up for kmeans
    int clustersCount = 3, iterations = 5;

    Mat samples(image.total(), 3, CV_32F);
    auto samples_ptr = samples.ptr<float>(0);
    
    for(int row = 0; row != image.rows; ++rows) {
        auto src_begin = imputImage.ptr<uchar>(row);
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
    TermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10, 0.01 ),
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

    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", clusteredImage);

    waitKey(0);
    return 0;
}