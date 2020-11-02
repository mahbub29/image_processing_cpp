#ifndef IM1_H_
#define IM1_H_

#include <iostream>
#include <opencv2/opencv.hpp>


class imageProcess
{	
	public:
		cv::Mat GrayImg;
		cv::Mat ColorImg;

		imageProcess (std::string IMAGE_FILE_PATH);

		cv::Mat
		medianFilterGray (int windowSize);

		cv::Mat
		medianFilterRGB (int windowSize);

		cv::Mat
		adaptiveFilterGray (int windowSize);

		cv::Mat
		adaptiveFilterColor (int windowSize);

		void
		kmeansSegmentation ();

	private:
		int
		getMedian (cv::Mat window);

		cv::Mat
		medianFilter (cv::Mat image, int windowSize);

		cv::Mat
		adaptiveFilter (cv::Mat image, int windowSize);

		static void
		leftMouseClick (int event, int i, int j, int flags, void *params);

		std::vector<cv::Mat>
		kmeansColor_nD_segmentation (cv::Mat i_vec, int ndims);

};
#endif