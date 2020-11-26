#include <iostream>
#include <fstream>
#include "im2.hpp"
#include "get.hpp"
#include <Python.h>
#include <opencv2/core/mat.hpp>




cv::Mat histogramProcess::NULL_MAT;


histogramProcess::histogramProcess (std::string IMAGE_FILE_PATH)
  : GrayImg(get::getGrayImg(IMAGE_FILE_PATH)), ColorImg(get::getColorImg(IMAGE_FILE_PATH))
	{}


std::vector<int> histogramProcess::getImageHistogram (cv::Mat image, bool showPlot, bool saveFig) {
  if (image.empty()) image = GrayImg;

  cv::Mat imgThread = image.reshape(1,1);
  imgThread.convertTo(imgThread, CV_64FC1);

  std::vector<int> intensityFreqs(256);
  for (int i=0; i<intensityFreqs.size(); i++) {intensityFreqs[i] = 0;}

  int p;
  for (int i=0; i<imgThread.cols; i++) {
    p = imgThread.at<double>(0,i);
    intensityFreqs[p]++;
  }

  if (showPlot) {
    std::vector<int> intensity(256);
    for (int i=0; i<intensityFreqs.size(); i++) {intensity[i] = i;}

    // create csv file to store intensity values and frequency of their occurences
    std::ofstream fileOut;
    std::string fileName = "data.csv";
    fileOut.open (fileName);
    for (int i=0; i<intensity.size(); i++) {
      fileOut << intensity[i] << "," << intensityFreqs[i] << "\n";
    }
    fileOut.close();

    // init python shell to run python script
    Py_Initialize();
    char pyFile[] = "plot_histogram.py";
    FILE *fp = _Py_fopen(pyFile, "r");
    // run the script to use matplotlib to plot histogran
    PyRun_SimpleFile(fp, pyFile);
    PyRun_SimpleString("plt.bar(I,f,width=0.5)\n"
                       "plt.xlabel('Intensity')\n"
                       "plt.ylabel('No. of pixels')\n");

    if (saveFig) {
      PyRun_SimpleString(
                         "plt.savefig('image_histogram.png')\n");
      std::cout << "Plot Figure saved.\n";
    }

    PyRun_SimpleString("plt.show()");

    // close python shell
    Py_Finalize();

    //delete csv file
    std::remove ("data.csv");
  }

  return intensityFreqs;
}


std::vector<int> histogramProcess::getCumulativeImageHist (cv::Mat cn, bool showPlot, bool saveFig) {
  if (cn.empty()) cn = GrayImg;
  
  std::vector<int> intensityFreqs = this->getImageHistogram (cn);
  std::vector<int> cumulativeFreq(256);

  for (int i=0; i<cumulativeFreq.size(); i++) {
    if (i==0)
      cumulativeFreq[i] = intensityFreqs[i];
    else {
      cumulativeFreq[i] = cumulativeFreq[i-1] + intensityFreqs[i];
    }
  }

  if (showPlot) {
    std::vector<int> intensity(256);
    for (int i=0; i<intensityFreqs.size(); i++) {intensity[i] = i;}

    std::ofstream fileOut;
    std::string fileName = "data.csv";
    fileOut.open (fileName);
    for (int i=0; i<cumulativeFreq.size(); i++) {
      fileOut << intensity[i] << "," << cumulativeFreq[i] << "\n";
    }
    fileOut.close();

    // init python shell to run python script
    Py_Initialize();
    char pyFile[] = "plot_histogram.py";
    FILE *fp = _Py_fopen(pyFile, "r");
    // run the script to use matplotlib to plot histogran
    PyRun_SimpleFile(fp, pyFile);
    PyRun_SimpleString("plt.plot(I,f, linewidth=0.5)\n"
                       "plt.xlabel('Intensity')\n"
                       "plt.ylabel('Cumulative No. of pixels')\n");

    if (saveFig) {
      PyRun_SimpleString("plt.savefig('image_cumulative_histogram.png')");
      std::cout << "Plot Figure saved.\n";
    }
    PyRun_SimpleString("plt.show()");

    // close python shell
    Py_Finalize();

    //delete csv file
    std::remove ("data.csv");
  }

  return cumulativeFreq;
}


cv::Mat histogramProcess::getEqualizedImage (cv::Mat image, bool showPlot, bool saveFig) {
  if (image.empty()) image = GrayImg;

  std::vector<int> cdf = this->getCumulativeImageHist ();
  cv::Mat copy = image.reshape(1,1);
  copy.convertTo (copy, CV_64FC1);

  int p;
  for (int i=0; i<copy.cols; i++) {
    p = copy.at<double>(0,i);
    copy.row(0).col(i) = cdf[p];
  }

  double minVal, maxVal;
  cv::Point2i minIdx, maxIdx;
  cv::minMaxLoc (copy, &minVal, &maxVal, &minIdx, &maxIdx);
  copy = copy/maxVal * 255;

  std::vector<int> intensityFreqs(256);
  for (int i=0; i<intensityFreqs.size(); i++) {intensityFreqs[i] = 0;}

  for (int i=0; i<copy.cols; i++) {
    p = copy.at<double>(0,i);
    intensityFreqs[p]++;
  }

  if (showPlot) {
    std::vector<int> intensity(256);
    for (int i=0; i<intensityFreqs.size(); i++) {intensity[i] = i;}

    // create csv file to store intensity values and frequency of their occurences
    std::ofstream fileOut;
    std::string fileName = "data.csv";
    fileOut.open (fileName);
    for (int i=0; i<intensity.size(); i++) {
      fileOut << intensity[i] << "," << intensityFreqs[i] << "\n";
    }
    fileOut.close();

    // init python shell to run python script
    Py_Initialize();
    char pyFile[] = "plot_histogram.py";
    FILE *fp = _Py_fopen(pyFile, "r");
    // run the script to use matplotlib to plot histogran
    PyRun_SimpleFile(fp, pyFile);
    PyRun_SimpleString("plt.bar(I,f,width=0.5)\n"
                       "plt.xlabel('Equalized Intensity')\n"
                       "plt.ylabel('No. of pixels')\n");

    if (saveFig) {
      PyRun_SimpleString(
                         "plt.savefig('image_histogram.png')\n");
      std::cout << "Plot Figure saved.\n";
    }

    PyRun_SimpleString("plt.show()");

    // close python shell
    Py_Finalize();

    // delete csv file
    std::remove ("data.csv");
  }

  copy.convertTo (copy, CV_8UC1);

  copy = copy.reshape(1,image.rows);

  return copy;
}


cv::Mat histogramProcess::getEqlColor () {
  cv::Mat bgr[3];
  cv::split(ColorImg, bgr);

  cv::Mat b, g, r;
  b = this->getEqualizedImage (bgr[0]);
  g = this->getEqualizedImage (bgr[1]);
  r = this->getEqualizedImage (bgr[2]);

  std::vector<cv::Mat> bgrOut = {b,g,r};
  cv::Mat imageOut; cv::merge (bgrOut, imageOut);

  return imageOut;
}


cv::Mat histogramProcess::matchHistogram_cn (cv::Mat src, cv::Mat tgt) {
  // get cdf for source image and target image
  std::vector<int> cdfSrc = this->getCumulativeImageHist (src);
  std::vector<int> cdfTgt = this->getCumulativeImageHist (tgt);

  // normalise the cdf values using the value in the last bin
  // and use the max values of each pixel intensity, i.e. out of cdfSrc and cdfTgt
  std::vector<double> cdf_Out(256);
  double cdfS_max = cdfSrc[255]; double cdfT_max = cdfTgt[255];
  for (int i=0; i<cdfSrc.size(); i++) { cdf_Out[i] = std::max(cdfSrc[i]/cdfS_max, cdfTgt[i]/cdfT_max); }

  cv::Mat tgt_copy = tgt.reshape (1,1);
  tgt_copy.convertTo (tgt_copy, CV_64FC1);
  
  int p;
  for (int i=0; i<tgt_copy.cols; i++) {
    p = tgt_copy.at<double>(0,i);
    tgt_copy.row(0).col(i) = cdf_Out[p]*255;
  }

  tgt_copy = tgt_copy.reshape (1, tgt.rows);
  tgt_copy.convertTo (tgt_copy, CV_8UC1);

  return tgt_copy;
}


cv::Mat histogramProcess::matchHistogram (std::string tgtPath, bool rgb) {
  cv::Mat imageOut;
  
  if (rgb)
  {
    cv::Mat tgt = cv::imread (tgtPath, cv::IMREAD_COLOR);
    cv::Mat bgr_src[3]; cv::split (ColorImg, bgr_src);
    cv::Mat bgr_tgt[3]; cv::split (tgt, bgr_tgt);

    cv::Mat tgt_b, tgt_g, tgt_r;
    tgt_b = this->matchHistogram_cn (bgr_src[0], bgr_tgt[0]);
    tgt_g = this->matchHistogram_cn (bgr_src[1], bgr_tgt[1]);
    tgt_r = this->matchHistogram_cn (bgr_src[2], bgr_tgt[2]);

    std::vector<cv::Mat> v = {tgt_b, tgt_g, tgt_r};
    cv::merge (v, imageOut);
  }
  else
  {
    cv::Mat tgt = cv::imread (tgtPath, cv::IMREAD_GRAYSCALE);
    imageOut = this->matchHistogram_cn (GrayImg, tgt);
  }

  return imageOut;
}