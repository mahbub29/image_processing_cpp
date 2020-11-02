#include <iostream>
#include "im1.hpp"
#include "get.hpp"
#include "im_kernels.hpp"
#include <opencv2/opencv.hpp>


int main()
{   
    std::string filePath = "/home/mahbub/ImageProcessing/im1/sample_image_t1.jpg";
    cv::Mat img = cv::imread(filePath, cv::IMREAD_COLOR);

    get get_;
    imageProcess imp_ (filePath);
    imageKernel kernel_ (filePath);

    // int windowSize;
    // bool noWindowChosen = true;

    // while (noWindowChosen) {
    //     std::cout << "Enter a window-size:" << " ";
    //     std::cin >> windowSize;

    //     if (windowSize%2!=1 || windowSize>img.rows || windowSize>img.cols) {
    //         std::cout << "ERROR: Window-size must be odd INT, and must not be "
    //                       "larger than the largest dimension of the selected image." << "\n";
    //     } else {
    //         noWindowChosen = false;
    //     }
    // }


    // cv::Mat outGray = imp_.adaptiveFilterGray (windowSize);
    // cv::Mat outColor = imp_.adaptiveFilterColor (windowSize);

    // cv::Mat out = kernel_.outline();
    
    // imp_.kmeansSegmentation (); //******************

    // cv::imshow("Original Image", img);
    // cv::imshow("Processed Gray Image", out);
    // // imshow("Processed Color Image", outColor);
    // cv::waitKey(0); // Wait for a keystroke in the window
    // cv::destroyAllWindows();

    return 0;
}