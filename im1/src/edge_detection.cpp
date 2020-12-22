#include <iostream>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

#include <edge_detection.hpp>
#include <get.hpp>
#include <im_kernels.hpp>



edge_detection::edge_detection (std::string IMAGE_FILE_PATH)
    : grayImg (get::getGrayImg(IMAGE_FILE_PATH))
    {}


cv::Mat edge_detection::sobelEdgeDetect (bool redOverlay, double threshold)
{
    imageKernel kernel_;

     // *********************
    cv::Mat rgbGray(grayImg.size(), CV_8UC3), chans[3];
    cv::cvtColor (grayImg, rgbGray, CV_GRAY2RGB);
    rgbGray.convertTo (rgbGray, CV_64FC3); cv::split (rgbGray, chans);
    // *********************

    cv::Mat top, bottom, left, right, combo;

    top = kernel_.SelectAnOption(grayImg, 4);
    bottom = kernel_.SelectAnOption(grayImg, 5);
    left = kernel_.SelectAnOption(grayImg, 6);
    right = kernel_.SelectAnOption(grayImg, 7);

    cv::max (top, bottom, combo);
    cv::max (left, combo, combo);
    cv::max (right, combo, combo);

    combo.convertTo (combo, CV_64FC1);
    
    if (redOverlay) {
        for (int i=0; i<combo.rows; i++) {
            for (int j=0; j<combo.cols; j++) {
                if (i==0 || i==combo.rows-1 || j==0 || j==combo.cols-1) continue;
                if (combo.at<double>(i,j)>=threshold) {
                    chans[0].row(i).col(j) = 0;
                    chans[1].row(i).col(j) = 0;
                    chans[2].row(i).col(j) = 255;
                }
            }
        }
    }
    else {
        for (int i=0; i<combo.rows; i++) {
            for (int j=0; j<combo.cols; j++) {
                if (i==0 || i==combo.rows-1 || j==0 || j==combo.cols-1) combo.row(i).col(j) = 0;
                else if (combo.at<double>(i,j)<threshold) combo.row(i).col(j) = 0;
                else combo.row(i).col(j) = 255;
            }
        }
    }

    cv::Mat imageOut; 
    
    if (redOverlay) {
        std::vector<cv::Mat> chanVec = {chans[0], chans[1], chans[2]};
        cv::merge (chanVec, imageOut);
        imageOut.convertTo (imageOut, CV_8UC3);
        std::cout << "done\n";
    } else {
        combo.convertTo (imageOut, CV_8UC1);
    }

    return imageOut;
}

