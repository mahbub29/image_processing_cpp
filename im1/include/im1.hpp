#ifndef IM1_H_
#define IM1_H_

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/mat.hpp>





class imageProcess
{	
	public:
		cv::Mat GrayImg;
		cv::Mat ColorImg;

		int GRAYSCALE = 1;
		int COLOR = 2;

		imageProcess (std::string IMAGE_FILE_PATH);

		cv::Mat
		medianFilterGray (int windowSize);

		cv::Mat
		medianFilterRGB (int windowSize);

		cv::Mat
		adaptiveFilterGray (int windowSize);

		cv::Mat
		adaptiveFilterColor (int windowSize);

		std::vector<cv::Mat>
		kmeansSegmentation (int selection, bool choose);

		std::vector<cv::Mat>
		kmeans_nD_segmentation (int channels, cv::Mat i_vec, int ndims);

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
		kmeansConvergence (int channels, cv::Mat i_vec, int ndims);

		cv::Mat
		makeColorPalette (int channels, cv::Mat i_vec);

};
#endif