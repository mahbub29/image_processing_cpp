
#include <iostream>
#include "get.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>


// function to get row range and column range of image window in getWindow() function
cv::Mat getRowCol (cv::Mat M, int i1, int i2, int j1, int j2)
{
	cv::Mat w = M.rowRange(i1, i2).colRange(j1, j2);
	return w;
}


// Window grabbing function - presents image window as windowSize x windowSize matrix
cv::Mat get::getWindow (cv::Mat image, int windowSize, int i, int j)
{
	int height = image.rows;
	int width = image.cols;
	int limit = floor(windowSize/2);

	cv::Mat window = cv::Mat::zeros(windowSize, windowSize, CV_64FC1);
	/** 
	CV_##AC#
	CV_   is compulsory prefix
	##    is the number of bits per matrix element
	A     is the type of base element
	C#	  # is the number of channels
	**/

	if (i<limit) {
		if (j<limit) {
			window = getRowCol(image, 0, i+limit+1, 0, j+limit+1);
		} else if (j>width-limit-1) {
			window = getRowCol(image, 0, i+limit+1, j-limit, width);
		} else {
			window = getRowCol(image, 0, i+limit+1, j-limit, j+limit+1);
		}
	}
	else if (j<limit) {
		if (i<limit) {
			window = getRowCol(image, 0, i+limit+1, 0, j+limit+1);
		} else if (i>height-limit-1) {
			window = getRowCol(image, i-limit, height, 0, j+limit+1);
		} else {
			window = getRowCol(image, i-limit, i+limit+1, 0, j+limit+1);
		}
	}
	else if (i>height-limit-1) {
		if (j<limit) {
			window = getRowCol(image, i-limit, height, 0, j+limit+1);
		} else if (j>width-limit-1) {
			window = getRowCol(image, i-limit, height, j-limit, width);
		} else {
			window = getRowCol(image, i-limit, height, j-limit, j+limit+1);
		}
	}
	else if (j>width-limit-1) {
		if (i<limit) {
			window = getRowCol(image, 0, i+limit+1, j-limit, width);
		} else if (i>height-limit-1) {
			window = getRowCol(image, i-limit, height, j-limit, width);
		} else {
			window = getRowCol(image, i-limit, i+limit+1, j-limit, width);
		}
	}
	else {
		window = getRowCol(image, i-limit, i+limit+1, j-limit, j+limit+1);
	}

	return window;
}


std::string get::type2str(int type) {
	std::string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch ( depth ) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans+'0');

	return r;
}