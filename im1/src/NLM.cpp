#include <iostream>
#include <math.h>
#include <chrono>
#include <fstream>

#include <opencv2/core/mat.hpp>

#include <NLM.hpp>
#include <get.hpp>





cv::Mat NLM::nonLocalMeans (cv::Mat img, double h, double sigma, int patchRadius, int windowRadius, int best)
{
  std::cout << "Denoising input image\n";
  auto start_time = std::chrono::high_resolution_clock::now(); // start timer

  int cn = img.channels();
  cv::Mat imageOut;
  if (cn==3)
  {
    // if color image
    cv::Mat bgr[3], b, g, r;
    cv::split (img, bgr);
    std::cout << "Blue Channel\n";
    b = this->nonLocalMeans_singleChannel(bgr[0], h, sigma, patchRadius, windowRadius, best);
    std::cout << "Green Channel\n";
    g = this->nonLocalMeans_singleChannel(bgr[1], h, sigma, patchRadius, windowRadius, best);
    std::cout << "Red Channel\n";
    r = this->nonLocalMeans_singleChannel(bgr[2], h, sigma, patchRadius, windowRadius, best);
    std::vector<cv::Mat> vec = {b,g,r};
    cv::merge (vec, imageOut);
  }
  else {
    // if grayscale image
    imageOut = this->nonLocalMeans_singleChannel(img, h, sigma, patchRadius, windowRadius, best);
  }

  auto end_time = std::chrono::high_resolution_clock::now(); // end timer
  auto time_taken = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
  std::cout << "Time taken: " << time_taken.count() << " s\n";

  return imageOut;
}


cv::Mat NLM::nonLocalMeans_singleChannel (cv::Mat img, double h, double sigma, int patchRadius, int windowRadius, int best)
{
  // pad image with zeros depending on window size and patch size
  cv::Mat paddedImg = get::getPaddedImage (img, windowRadius);

  // set boundaries for iteration
  int beginRow, beginCol, endRow, endCol;
  beginRow = beginCol = windowRadius;
  endRow = paddedImg.rows-windowRadius;
  endCol = paddedImg.cols-windowRadius;

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
      pixel = this->computeWeighting (distance, searchWindow, h, sigma, best);
      imgOut64.row(i-windowRadius).col(j-windowRadius) = pixel;

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
  cv::Mat distance = cv::Mat::zeros (windowSize-patchSize+1, windowSize-patchSize+1, CV_64FC1); // to store the ssd distances

  // Define the template patch to be tested
  cv::Mat templatePatch = img64.rowRange(row-patchRadius,row+patchRadius+1)
                               .colRange(col-patchRadius,col+patchRadius+1);

  // Define the search window within the image (this includes that
  // the patch can centre on the outer pixels and therefore overlap the
  // edges of the search window box as well)
  cv::Mat searchWindow; img64.rowRange(row-windowRadius,row+windowRadius+1)
                             .colRange(col-windowRadius,col+windowRadius+1)
                             .copyTo (searchWindow);
  cv::Mat _searchWindow_; img64.rowRange(row-windowRadius+patchRadius,row+windowRadius-patchRadius+1)
                             .colRange(col-windowRadius+patchRadius,col+windowRadius-patchRadius+1)
                             .copyTo (_searchWindow_);

  // iterate over each of the pixels within the desired search window and
  // get the sum of the squared differences
  cv::Mat testPatch; // variable to store test patch
  cv::Mat sqrDiff;   // variable to store square distances
  double sqrDiffSum; // variable to store sum of squared differences
  int count=0;

  for (int i=patchRadius; i<windowSize-patchRadius; i++) {
    for (int j=patchRadius; j<windowSize-patchRadius; j++) {
      // designate test patch matrix centered at p(i,j)
      testPatch = searchWindow.rowRange(i-patchRadius,i+patchRadius+1)
                              .colRange(j-patchRadius,j+patchRadius+1);
      cv::pow (testPatch-templatePatch, 2, sqrDiff); // store the square difference
                                                     // between corresponding pixels
      sqrDiffSum = cv::sum (sqrDiff)[0]; // calculate sum of square differences between patches
      distance.row(i-patchRadius).col(j-patchRadius) = sqrDiffSum/pow(patchSize,2); // store it in distances
    }
  }

  std::vector<cv::Mat> searchWindow_n_distance;
  searchWindow_n_distance = {_searchWindow_, distance};

  return searchWindow_n_distance;
}


double NLM::computeWeighting (cv::Mat distance, cv::Mat searchWindow, double h, double sigma, int best)
{
  // reshape distances and search window into two lines of numbers
  distance = distance.reshape (1,1);
  searchWindow = searchWindow.reshape (1,1);

  // sort the distances in order from lowest to highest and line get
  // the corresponding pixels
  cv::Mat distIds;
  cv::sortIdx (distance, distIds, cv::SORT_EVERY_ROW+cv::SORT_ASCENDING);
  distIds.convertTo (distIds, CV_64FC1);
  int idx;
  cv::Mat distance_sorted = cv::Mat::zeros (distance.size(), CV_64FC1);
  cv::Mat corresponding_pixels = cv::Mat::zeros (distance.size(), CV_64FC1);
  for (int i=0; i<distIds.cols; i++) {
    idx = distIds.at<double>(0,i);
    distance_sorted.row(0).col(i) = distance.at<double>(0,idx);
    corresponding_pixels.row(0).col(i) = searchWindow.at<double>(0,idx);
  }

  // get the weight of the top pixels and multiply the weights by
  // the correspodning pixels
  cv::Mat weights = cv::Mat::zeros (distance_sorted.size(),CV_64FC1);
  double d2, w;
  for (int i=0; i<best; i++) {
    d2 = distance_sorted.at<double>(0,i);
    w = exp(-std::max(d2-2*sigma*sigma, 0.0)/(h*h));
    weights.row(0).col(i) = w;
  }
  weights = weights*(1/cv::sum(weights)[0]);
  cv::Mat product; cv::multiply (weights, corresponding_pixels, product);

  double denoisedVal = cv::sum (product)[0];

  return denoisedVal;
}

