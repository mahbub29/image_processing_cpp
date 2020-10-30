


#ifndef IM1_H_
#define IM1_H_

#include <iostream>
#include <opencv2/opencv.hpp>


class imageProcess
{	
	std::string file_path;

	public:
		imageProcess (std::string IMAGE_FILE_PATH);

		cv::Mat
		medianFilterGray (int windowSize);

		cv::Mat
		medianFilterRGB (int windowSize);

		cv::Mat
		adaptiveFilterGray (int windowSize);

		cv::Mat
		adaptiveFilterColor (int windowSize);

};
#endif