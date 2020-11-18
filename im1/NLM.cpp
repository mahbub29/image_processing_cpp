#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>
#include "NLM.hpp"
#include "get.hpp"
#include <chrono>



cv::Mat NLM::nonLocalMeans (cv::Mat img, double h, double sigma, int patchRadius, int windowRadius)
{
  auto start_time = std::chrono::high_resolution_clock::now(); // start timer

  int cn = img.channels();
  cv::Mat imageOut;
  if (cn==3)
  {
    // if color image
    cv::Mat bgr[3], b, g, r;
    cv::split (img, bgr);
    std::cout << "Blue Channel\n";
    b = this->nonLocalMeans_singleChannel(bgr[0], h, sigma, patchRadius, windowRadius);
    std::cout << "Green Channel\n";
    g = this->nonLocalMeans_singleChannel(bgr[1], h, sigma, patchRadius, windowRadius);
    std::cout << "Red Channel\n";
    r = this->nonLocalMeans_singleChannel(bgr[2], h, sigma, patchRadius, windowRadius);
    std::vector<cv::Mat> vec = {b,g,r};
    cv::merge (vec, imageOut);
  }
  else {
    // if grayscale image
    imageOut = this->nonLocalMeans_singleChannel(img, h, sigma, patchRadius, windowRadius);
  }

  auto end_time = std::chrono::high_resolution_clock::now(); // end timer
  auto time_taken = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
  std::cout << "Time taken: " << time_taken.count() << " s\n";

  return imageOut;
}


cv::Mat NLM::nonLocalMeans_singleChannel (cv::Mat img, double h, double sigma, int patchRadius, int windowRadius)
{
  // pad image with zeros depending on window size and patch size
  cv::Mat paddedImg = get::getPaddedImage (img, patchRadius+windowRadius);

  // set boundaries for iteration
  int beginRow, beginCol, endRow, endCol, padding;
  padding = patchRadius+windowRadius;
  beginRow = beginCol = padding;
  endRow = img.rows+padding;
  endCol = img.cols+padding;

  // set space for output variables
  std::vector<cv::Mat> var;
  cv::Mat searchWindow, distance;
  double pixel;
  cv::Mat imgOut64 = cv::Mat::zeros (img.size(), CV_64FC1);

  // set up variables to print progress
  int count = 0;
  double total = img.rows*img.cols;
  double progress;

  for (int i=beginRow; i<endRow; i++) {
    for (int j=beginCol; j<endCol; j++) {
      var = this->naiveTemplateMatching (paddedImg, i, j, patchRadius, windowRadius);
      searchWindow = var[0]; distance = var[1];
      pixel = this->computeWeighting (distance, searchWindow, h, sigma);
      imgOut64.row(i-padding).col(j-padding) = pixel;

      count++;
      progress = round(count/total*10000)/100;
      std::cout << "\r" << progress << "%" << std::flush;
    }
  }

  // convert to unit8
  cv::Mat imgOut; imgOut64.convertTo (imgOut, CV_8UC1);
  std::cout << "   [DONE]\n";

  return imgOut;
}


std::vector<cv::Mat> NLM::naiveTemplateMatching (cv::Mat paddedImg, int row, int col, int patchRadius, int windowRadius)
{
  int patchSize = patchRadius*2+1;
  int windowSize = windowRadius*2+1;

  cv::Mat img64; paddedImg.convertTo (img64, CV_64FC1); // convert padded image from uint8 to double
  cv::Mat distance = cv::Mat::zeros (windowSize, windowSize, CV_64FC1); // to store the ssd distances

  // Define the template patch to be tested
  cv::Mat templatePatch = img64.rowRange(row-patchRadius,row+patchRadius+1)
                               .colRange(col-patchRadius,col+patchRadius+1);

  // Define the search window within the image (this includes that
  // the patch can centre on the outer pixels and therefore overlap the
  // edges of the search window box as well)
  cv::Mat templateWindow = img64.rowRange(row-windowRadius,row+windowRadius+1)
                                .colRange(col-windowRadius,col+windowRadius+1);
  cv::Mat searchWindow   = img64.rowRange(row-patchRadius-windowRadius,row+patchRadius+windowRadius+1)
                                .colRange(col-patchRadius-windowRadius,col+patchRadius+windowRadius+1);

  // iterate over each of the pixels within the desired search window and
  // get the sum of the squared differences
  cv::Mat testPatch; // variable to store test patch
  cv::Mat sqrDiff;   // variable to store square distances
  double sqrDiffSum; // variable to store sum of squared differences
  for (int i=patchRadius; i<windowSize; i++) {
    for (int j=patchRadius; j<windowSize; j++) {
      // designate test patch matrix centered at p(i,j)
      testPatch = searchWindow.rowRange(i-patchRadius,i+patchRadius+1)
                              .colRange(j-patchRadius,j+patchRadius+1);
      cv::pow (testPatch-templatePatch, 2, sqrDiff); // store the square difference
                                                     // between corresponding pixels
      sqrDiffSum = cv::sum (sqrDiff)[0]; // calculate sum of square differences between patches
      distance.row(i).col(j) = sqrDiffSum; // store it in distances
    }
  }

  cv::Mat k = this->getWeightingKernel (windowRadius); // initialize weighting kernel
  cv::multiply (k, distance, distance);

  std::vector<cv::Mat> searchWindow_n_distance;
  searchWindow_n_distance = {templateWindow, distance};

  return searchWindow_n_distance;
}


double NLM::computeWeighting (cv::Mat distance, cv::Mat templateWindow, double h, double sigma)
{
  cv::Mat ZERO_MAT = cv::Mat::zeros (distance.size(), CV_64FC1);
  cv::Mat distDiff = distance-2*sigma*sigma;
  cv::Mat num; cv::max (ZERO_MAT, distDiff, num);
  cv::Mat x = -1/(h*h)*num;
  cv::Mat w; cv::exp (x, w);

  double C = cv::sum (w)[0];
  cv::Mat B_Mat; cv::multiply (templateWindow, w, B_Mat);
  double B = (1/C)*cv::sum(B_Mat)[0];

  return B;
}


cv::Mat NLM::getWeightingKernel (int windowRadius)
{
  int f = windowRadius;
  cv::Mat k = cv::Mat::zeros (2*f+1, 2*f+1, CV_64FC1);
  double val;

  for (int d=1; d<f+1; d++) {
    val = 1/pow(2*d+1,2);

    for (int i=-d; i<d+1; i++) {
      for (int j=-d; j<d+1; j++) {
        k.row(f-i).col(f-j) += val;
  }}}
  k = k/f;

  return k;
}