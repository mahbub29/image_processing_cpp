
#include <iostream>
#include "im1.hpp"
#include "get.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>


int getMedian (cv::Mat window)
{	
	cv::Mat window_thread = cv::Mat::zeros(1, window.rows*window.cols, CV_8UC1);
	window_thread = window.clone().reshape(0, window.rows*window.cols);
	cv::sort(window_thread, window_thread, CV_SORT_ASCENDING);

	int median;
	if ((window.rows*window.cols)%2==1) {
		median = window_thread.at<uchar>(floor(window_thread.rows/2)-1);
	} else {
		median = (window_thread.at<uchar>(floor(window_thread.rows/2)-1) + 
				  window_thread.at<uchar>(floor(window_thread.rows/2)))/2;
	}

	return median;
}


cv::Mat medianFilter (cv::Mat image, int windowSize)
{	
	if (windowSize%2!=1) {
		std::cout << "ERROR: windowSize must be odd int" << std::endl;
	} else {
		int height = image.rows;
		int width = image.cols;

		cv::Mat window = cv::Mat::zeros(windowSize, windowSize, CV_64FC1);
		cv::Mat imageOut = cv::Mat::zeros(height, width, CV_64FC1);
		double count = 0;
		double progress;
		double last_progress = 1000;
		int total = image.rows*image.cols;

		for (int i=0; i<height; i++) {
			for (int j=0; j<width; j++) {
				window = get::getWindow(image, windowSize, i, j);
				imageOut.row(i).col(j) = getMedian(window);
				count++;
				progress = floor(count/56852*100);
				if (progress != last_progress){
					if (std::fmod(progress,5)==0) {
						std::cout << progress << "%" << "\n";
						last_progress = progress;
					}
				}
			}
		}

		imageOut.convertTo(imageOut, CV_8UC1);
		return imageOut;
	}
}


imageProcess::imageProcess (std::string IMAGE_FILE_PATH)
	: file_path(IMAGE_FILE_PATH) {}


cv::Mat imageProcess::medianFilterGray (int windowSize)
{
	// Read as grayscale image
	cv::Mat image = cv::imread(file_path, cv::IMREAD_GRAYSCALE);

	if(image.empty())
    {
        std::cout << "Could not read the image: " << file_path << std::endl;
    }

	std::cout << "Processing GRAY Median Filter" << std::endl;
	cv::Mat imageOut = medianFilter(image, windowSize);

	return imageOut;
}


cv::Mat imageProcess::medianFilterRGB (int windowSize)
{	
	// Read as RGB image
	cv::Mat image = cv::imread(file_path, cv::IMREAD_COLOR);
	
	if(image.empty())
    {
        std::cout << "Could not read the image: " << file_path << std::endl;
    }

	// Split image into BGR color channels
	cv::Mat bgr[3];
	cv::split (image, bgr); // split the color channels int the image
	cv::Mat blue = bgr[0];
	cv::Mat green = bgr[1];
	cv::Mat red = bgr[2];

	std::cout << "Processing RGB Median Filter" << std::endl;
	std::cout << "Blue  Channel" << "\n";
	cv::Mat blueOut = medianFilter(blue, windowSize);
	std::cout << "Green Channel" << "\n";
	cv::Mat greenOut = medianFilter(green, windowSize);
	std::cout << "Red Channel" << "\n";
	cv::Mat redOut = medianFilter(red, windowSize);

	cv::Mat imageChannels[3] = {blueOut, greenOut, redOut};
	cv::Mat imageOut(bgr[0].size(), CV_8UC3);
	cv::merge (imageChannels, 3, imageOut);

	return imageOut;
}