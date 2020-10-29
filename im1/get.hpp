#ifndef GET_H_
#define GET_H_

#include <iostream>
#include <opencv2/opencv.hpp>


class get
{
	public:
		// Window grabbing function - presents image window as windowSize x 
		// windowSize matrix
		static cv::Mat
		getWindow (cv::Mat image, int windowSize, int i, int j);

		static std::string
		type2str(int type);
		
};
#endif

