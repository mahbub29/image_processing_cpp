#ifndef NLM_H_
#define NLM_H_


#include <iostream>
#include <opencv2/core/mat.hpp>




class NLM
{
  public:
    cv::Mat
    nonLocalMeans (cv::Mat img, double h, double sigma, int patchRadius, int windowRadius, int best=10);

  private:
    cv::Mat
    nonLocalMeans_singleChannel (cv::Mat img, double h, double sigma, int patchRadius, int windowRadius, int best);

    std::vector<cv::Mat>
    naiveTemplateMatching (cv::Mat img, int row, int col, int patchRadius, int windowRadius);

    double
    computeWeighting (cv::Mat distance, cv::Mat templateWindow, double h, double sigma, int best);

};
#endif
