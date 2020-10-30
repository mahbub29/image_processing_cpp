#include <iostream>
#include "get.hpp"
#include "im_kernels.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>


imageKernel::imageKernel (std::string IMAGE_FILE_PATH)
	: GrayImg(get::getGrayImg(IMAGE_FILE_PATH)) 
	{}


cv::Mat imageKernel::applyKernel (cv::Mat kernel)
{
	cv::Mat imageOut = cv::Mat::zeros(GrayImg.rows, GrayImg.cols, CV_64FC1);
	cv::Mat_<double> window;
	int p;

	for (int i=1; i<GrayImg.rows-1; i++) {
		for (int j=1; j<GrayImg.cols-1; j++) {
			window = get::getWindow (GrayImg, 3, i, j);
			p = cv::sum(window*kernel)[0];
			imageOut.row(i).col(j) = p;
		}
	}

	// Find the maximum value of the image regularize the Matrix relative to 255
	double MIN, MAX;
	cv::Point pMin, pMax;
	cv::minMaxLoc (imageOut, &MIN, &MAX, &pMin, &pMax);
	imageOut = imageOut / MAX * 255;

	// Convert image to uint8
	imageOut.convertTo(imageOut, CV_8UC1);

	return imageOut;
}


cv::Mat imageKernel::sharpen ()
{	
	std::cout << "Applying Sharpening Kernel..." << "\n";
	cv::Mat k = (cv::Mat_<double>(3,3) << 0,-1,0, -1,5,-1, 0,-1,0);
	cv::Mat imageOut = this->applyKernel (k);
	std::cout << "FINISHED" << "\n";
	return imageOut;
}


cv::Mat imageKernel::blur ()
{	
	std::cout << "Applying Blurring Kernel..." << "\n";
	cv::Mat k = (cv::Mat_<double>(3,3) << 0.0625,0.125,0.0625, 0.125,0.25,0.125, 0.0625,0.125,0.0625);
	cv::Mat imageOut = this->applyKernel (k);
	std::cout << "FINISHED" << "\n";
	return imageOut;
}


cv::Mat imageKernel::emboss ()
{	
	std::cout << "Applying Embossing Kernel..." << "\n";
	cv::Mat k = (cv::Mat_<double>(3,3) << -2,-1,0, -1,1,1, 0,1,2);
	cv::Mat imageOut = this->applyKernel (k);
	std::cout << "FINISHED" << "\n";
	return imageOut;
}


cv::Mat imageKernel::topSobel ()
{	
	std::cout << "Applying Sobel (Top) Kernel..." << "\n";
	cv::Mat k = (cv::Mat_<double>(3,3) << 1,2,1, 0,0,0, -1,-2,-1);
	cv::Mat imageOut = this->applyKernel (k);
	std::cout << "FINISHED" << "\n";
	return imageOut;
}


cv::Mat imageKernel::bottomSobel ()
{	
	std::cout << "Applying Sobel (Bottom) Kernel..." << "\n";
	cv::Mat k = (cv::Mat_<double>(3,3) << -1,-2,-1, 0,0,0, 1,2,1);
	cv::Mat imageOut = this->applyKernel (k);
	std::cout << "FINISHED" << "\n";
	return imageOut;
}


cv::Mat imageKernel::leftSobel ()
{	
	std::cout << "Applying Sobel (Left) Kernel..." << "\n";
	cv::Mat k = (cv::Mat_<double>(3,3) << 1,0,-1, 2,0,-2, 1,0,-1);
	cv::Mat imageOut = this->applyKernel (k);
	std::cout << "FINISHED" << "\n";
	return imageOut;
}


cv::Mat imageKernel::rightSobel ()
{	
	std::cout << "Applying Sobel (Right) Kernel..." << "\n";
	cv::Mat k = (cv::Mat_<double>(3,3) << -1,0,1, -2,0,2, -1,0,1);
	cv::Mat imageOut = this->applyKernel (k);
	std::cout << "FINISHED" << "\n";
	return imageOut;
}


cv::Mat imageKernel::outline ()
{	
	std::cout << "Applying Outlining Kernel..." << "\n";
	cv::Mat k = (cv::Mat_<double>(3,3) << -1,-1,-1, -1,8,-1, -1,-1,-1);
	cv::Mat imageOut = this->applyKernel (k);
	std::cout << "FINISHED" << "\n";
	return imageOut;
}


cv::Mat imageKernel::smooth ()
{	
	std::cout << "Applying Smoothing Kernel..." << "\n";
	cv::Mat k = (cv::Mat_<double>(3,3) << 1/9,1/9,1/9, 1/9,1/9,1/9, 1/9,1/9,1/9);
	cv::Mat imageOut = this->applyKernel (k);
	std::cout << "FINISHED" << "\n";
	return imageOut;
}


cv::Mat imageKernel::customInput ()
{
	std::cout << "Enter custom kernel matrix. Values should be separated by a SPACE, "
				 "where 1st value corresponds to m(0,0), 2nd value to m(0,1), 3rd to m(0,2) "
				 "and so on up to 9th value which corresponds to m(2,2):" << "\n";

	int i=0;
	double n[9];
	while (i<9) {
		std::cin >> n[i];
		i++;
	}

	cv::Mat k = (cv::Mat_<double>(3,3) << n[0],n[1],n[2], n[3],n[4],n[5], n[6],n[7],n[8]);
	std::cout << "Applying the following custom kernel:" << "\n";
	std::cout << k << "..." << "\n";
	cv::Mat imageOut = this->applyKernel (k);
	std::cout << "FINISHED" << "\n";
	return imageOut;
}



