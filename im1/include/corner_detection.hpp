#ifndef CORNER_DETECTION_H_
#define CORNER_DETECTION_H_

#include <iostream>
#include <opencv2/core/mat.hpp>

class corner_detection
{
    public:
        cv::Mat grayImg;

        corner_detection (std::string IMAGE_FILE_PATH);

        cv::Mat
        movarecDetect (int r, double sigma, double threshold);
};
#endif