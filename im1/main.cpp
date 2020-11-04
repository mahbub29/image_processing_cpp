#include <iostream>
#include "im1.hpp"
#include "get.hpp"
#include "im_kernels.hpp"
#include <opencv2/opencv.hpp>


int main()
{   
    std::string filePath = "/home/mahbub/ImageProcessing/im1/sample_image_t1.jpg";
    cv::Mat img = cv::imread(filePath, cv::IMREAD_GRAYSCALE);

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

    // img.convertTo(img, CV_64FC1);
    // std::cout << img << "\n";
    // cv::Mat t = cv::Mat::zeros(img.size(), CV_64FC1);
    // for (int i=0; i<t.cols; i++) {
    //     // std::cout << img.col(0) << "\n";
    //     img.col(0).copyTo(t.col(i));
    // }
    // std::cout << t << "\n";
    // std::cout << img - t << "\n";
    // // cv::Mat outGray = imp_.adaptiveFilterGray (windowSize);
    // // cv::Mat outColor = imp_.adaptiveFilterColor (windowSize);

    // cv::Mat out = kernel_.outline();
    
    imp_.kmeansSegmentation (); //******************
    // cv::Mat y = img-t;
    // std::cout << y << "\n";
    // y.convertTo(y, CV_8UC1);
    // // img.convertTo(img, CV_8UC1);
    // cv::imshow("Original Image", img);
    // // cv::imshow("Processed Gray Image", out);
    // // // imshow("Processed Color Image", outColor);
    // cv::waitKey(0); // Wait for a keystroke in the window
    // cv::destroyAllWindows();

    return 0;
}