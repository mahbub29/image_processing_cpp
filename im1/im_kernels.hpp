#ifndef IM_KERNEL_H_
#define IM_KERNEL_H_


#include <iostream>
#include <opencv2/opencv.hpp>


class imageKernel
{
	public:
		cv::Mat GrayImg;

		imageKernel (std::string IMAGE_FILE_PATH);

		cv::Mat
		sharpen ();

		cv::Mat
		blur ();

		cv::Mat
		emboss ();

		cv::Mat
		topSobel ();

		cv::Mat
		bottomSobel ();

		cv::Mat
		leftSobel ();

		cv::Mat
		rightSobel ();

		cv::Mat
		outline ();

		cv::Mat
		smooth ();

};
#endif