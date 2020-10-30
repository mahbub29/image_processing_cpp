#ifndef GET_H_
#define GET_H_

#include <iostream>
#include <opencv2/opencv.hpp>


class get
{
	public:
		// Window grabbing function - presents image window as
		//windowSize x windowSize matrix
		static cv::Mat
		getWindow (cv::Mat image, int windowSize, int i, int j);

		// outputs string describing type of cv::Mat contents
		static std::string
		getType (int type);

		// gets means of image window area across every pixel of image
		static cv::Mat
		getMeans (cv::Mat image, int windowSize);

		// get variances of image window area across every pixel of image
		static cv::Mat
		getVariances (cv::Mat image, int windowSize, cv::Mat means);
};
#endif

