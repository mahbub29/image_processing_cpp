#ifndef GET_H_
#define GET_H_

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/mat.hpp>



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

		// pads the image with a specified number of zeros
		static cv::Mat
		getPaddedImage (cv::Mat image, int padding);
		
		// writes a csv file with for the specified matrix
		static void
		writeCSV(std::string filename, cv::Mat m);

		// gets gaussian distributed weighting kernel
		static cv::Mat
		getGaussianWeightingKernel (int windowRadius);
};
#endif
