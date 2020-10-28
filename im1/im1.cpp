

#include "im1.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>


// function to get row range and column range of image window in getWindow() function
cv::Mat getRowCol (cv::Mat M, int i1, int i2, int j1, int j2)
{
	return M.rowRange(i1, i2).colRange(j1, j2);
}


int getMedian (cv::Mat window)
{
	cv::Mat window_thread = cv::Mat::zeros(1, window.rows*window.cols, CV_64FC1);
	window_thread = window.reshape(1, window.rows*window.cols);
	cv::sort(window_thread, window_thread, CV_SORT_ASCENDING);

	int median
	if ((window.rows*window.cols)%2==1) {
		median = window_thread.row(0).col(floor(window_thread.cols/2));
	} else {
		median = (window_thread.row(0).col(window_thread.cols/2) + 
				  window_thread.row(0).col(window_thread.cols/2))/2
	}

	return median;
}


// Window grabbing function - presents image window as windowSize x windowSize matrix
cv::Mat windowProcess::getWindow (cv::Mat image, int windowSize, int i, int j)
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



imageProcess::imageProcess (cv::Mat new_image)
	: image(new_image) {}


imageProcess::medianFilter ()
{

}