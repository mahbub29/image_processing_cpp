#include <iostream>
#include "im1.hpp"
#include <opencv2/opencv.hpp>


int main()
{
	windowProcess wp;

	std::string file = "/home/mahbub/ImageProcessing/im1/sample_image_t1.jpg";
	// cv::Mat img = cv::imread(file, cv::IMREAD_COLOR);		// color image (BGR)
	cv::Mat img = cv::imread(file, cv::IMREAD_GRAYSCALE);	// grascale image
	
	// cv::Mat bgr[3];
	// cv::split(cimg,bgr); // split the color channels int the image

	if(img.empty())
    {
        std::cout << "Could not read the image: " << file << std::endl;
        return 1;
    }

    // std::cout << img-100 << "\n";
    // std::cout << img.cols << "\n";
    // std::cout << img.rows << "\n";
    // std::cout << img.colRange(0,10).rowRange(0,10) << "\n";

    // for (int i=0; i<img.rows; i++) {
    // 	for (int j=0; j<img.cols; j++) {
    // 		std::cout << i << "\n";
    // 		std::cout << j << "\n";
    // 		std::cout << wp.getWindow(img, 3, i, j) << "\n";
    // 	}
    // }

    cv::Mat w = cv::Mat::zeros(1,10,CV_64FC1);
    w = img.rowRange(0,1).colRange(0,10);
    std::cout << w << "\n";
    cv::sort(w, w, CV_SORT_ASCENDING);
    std::cout << w << "\n";
    std::cout << w.size() << "\n";
    std::cout << img.size() << "\n";

    imshow("Display window", img);
    cv::waitKey(0); // Wait for a keystroke in the window
    cv::destroyAllWindows();

    return 0;
}