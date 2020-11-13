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

		// gets gray image conversion of input image file path
		static cv::Mat
		getGrayImg (std::string file_path);

		// gets 3 channel RGB conversion of input image file path
		static cv::Mat
		getColorImg (std::string file_path);

		// gets Grayscale, RGB or RGBij intensity vectors
		static cv::Mat
		get_nD_intensities (cv::Mat IMG, std::vector< std::vector<int> > selection, int dims=3);
};
#endif

