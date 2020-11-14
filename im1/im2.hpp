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

    std::vector<int>
    getImageHistogram (cv::Mat image=NULL_MAT, bool showPlot=false, bool saveFig=false);

    std::vector<int>
    getCumulativeImageHist (bool showPlot=false, bool saveFig=false);

    cv::Mat
    getEqualizedImage (cv::Mat image=NULL_MAT, bool showPlot=false, bool saveFig=false);

    cv::Mat
    getEqlColor ();

  private:
    static cv::Mat NULL_MAT;

};
#endif
