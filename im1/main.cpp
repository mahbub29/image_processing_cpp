#include <iostream>
#include "im1.hpp"
#include "get.hpp"
#include <opencv2/opencv.hpp>


int main()
{   
    std::string filePath = "/home/mahbub/ImageProcessing/im1/sample_image_t1.jpg";
    cv::Mat imgOriginal = cv::imread(filePath, cv::IMREAD_COLOR);


    imageProcess imp_ (filePath);

    cv::Mat outGray = imp_.medianFilterGray (21);
    cv::Mat outColor = imp_.medianFilterRGB (21);


    cv::imshow("Original Image", imgOriginal);
    imshow("Processed Gray Image", outGray);
    imshow("Processed Color Image", outColor);
    cv::waitKey(0); // Wait for a keystroke in the window
    cv::destroyAllWindows();

    return 0;
}