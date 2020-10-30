#include <iostream>
#include "get.hpp"
#include "im_kernels.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>


imageKernel::imageKernel (std::string IMAGE_FILE_PATH)
	: GrayImg(get::getGrayImg(IMAGE_FILE_PATH)) 
	{}