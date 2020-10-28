


#ifndef IM1_H_
#define IM1_H_

#include <iostream>
#include <opencv2/opencv.hpp>


class windowProcess
{
	public:
		// Window grabbing function - presents image window as windowSize x 
		// windowSize matrix
		cv::Mat
		getWindow (cv::Mat image, int windowSize, int i, int j);
		
};

class imageProcess
{	
	cv::Mat image;

	public:
		imageProcess (cv::Mat new_image);

		cv::Mat
		medianFilter ();


};
#endif