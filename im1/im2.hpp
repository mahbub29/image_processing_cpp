#ifndef IM2_H_
#define IM2_H_

#include <iostream>
#include <opencv2/opencv.hpp>


class histogramProcess
{
  public:
    cv::Mat GrayImg;
    cv::Mat ColorImg;

    histogramProcess (std::string IMAGE_FILE_PATH);

    void
    getImageHistogram ();
};
#endif
