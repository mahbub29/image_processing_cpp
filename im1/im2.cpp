#include <iostream>
#include <fstream>
#include "im2.hpp"
#include "get.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>
#include <Python.h>



histogramProcess::histogramProcess (std::string IMAGE_FILE_PATH)
  : GrayImg(get::getGrayImg(IMAGE_FILE_PATH)), ColorImg(get::getColorImg(IMAGE_FILE_PATH))
	{}


void histogramProcess::getImageHistogram () {
  cv::Mat GrayThread = GrayImg.reshape(1,1);
  GrayThread.convertTo(GrayThread, CV_64FC1);

  std::vector<double> intensityFreqs(256);
  for (int i=0; i<intensityFreqs.size(); i++) {intensityFreqs[i] = 0;}

  int p;
  for (int i=0; i<GrayThread.cols; i++) {
    p = GrayThread.at<double>(0,i);
    intensityFreqs[p]++;
  }

  std::vector<int> intensity(256);
  for (int i=0; i<intensityFreqs.size(); i++) {intensity[i] = i;}

  // create csv file to store intensity values and frequency of their occurences
  std::ofstream fileOut;
  std::string fileName = "intensity.csv";
  fileOut.open(fileName);
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
  // close python shell
  Py_Finalize();

  //delete csv file
  std::remove ("intensity.csv");
  
  return;
}
