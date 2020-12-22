#ifndef EDGE_DETECTION_H_
#define EDGE_DETECTION_H_

#include <iostream>
#include <opencv2/core/mat.hpp>



class edge_detection
{
    public:
        cv::Mat grayImg;

        edge_detection (std::string IMAGE_FILE_PATH);

        cv::Mat
        sobelEdgeDetect (bool redOverlay=true, double threshold=50);

};
#endif