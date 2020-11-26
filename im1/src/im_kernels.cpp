#include <iostream>
#include <math.h>

#include <opencv2/core/mat.hpp>

#include <get.hpp>
#include <im_kernels.hpp>





imageKernel::imageKernel (std::string IMAGE_FILE_PATH)
	: GrayImg(get::getGrayImg(IMAGE_FILE_PATH))
	{}


cv::Mat imageKernel::applyKernel (cv::Mat kernel)
{
	int sz = kernel.rows; int r = (sz-1)/2;
	cv::Mat imageOut = cv::Mat::zeros (GrayImg.rows, GrayImg.cols, CV_64FC1);
	cv::Mat imgPad = get::getPaddedImage (GrayImg, r);

	cv::Mat_<double> window, k_window;
	int p;
	double total = GrayImg.rows * GrayImg.cols;

	int k_diff_row = GrayImg.rows-r;
	int k_diff_col = GrayImg.cols-r;
	
	cv::Mat mulmat;

	int count=0;
	double progress;
	
	for (int i=0; i<GrayImg.rows; i++) {
		for (int j=0; j<GrayImg.cols; j++) {
			// window = get::getWindow (imgPad, sz, i, j);
			// p = cv::sum(window*kernel)[0];
			// imageOut.row(i).col(j) = p;

			window = get::getWindow (GrayImg, sz, i, j);

			if (i<r) {
				if (j<r) { k_window = get::getWindow (kernel, sz, sz-i-1, sz-j-1); }
				else if (j>GrayImg.cols-r-1) {
					k_diff_col = GrayImg.cols-j-1;
					k_window = get::getWindow (kernel, sz, sz-i-1, k_diff_col);
				}
				else { k_window = get::getWindow (kernel, sz, sz-i-1, r); }
			}
			else if (i>GrayImg.rows-r-1) {
				k_diff_row = GrayImg.rows-i-1;
				if (j<r) { k_window = get::getWindow (kernel, sz, k_diff_row, sz-j-1); }
				else if (j>GrayImg.cols-r-1) {
					k_diff_col = GrayImg.cols-j-1;
					k_window = get::getWindow (kernel, sz, k_diff_row, k_diff_col);
				}
				else { k_window = get::getWindow (kernel, sz, k_diff_row, r); }
			}
			else {
				if (j<r) { k_window = get::getWindow (kernel, sz, r, sz-j-1); }
				else if (j>GrayImg.cols-r-1) {
					k_diff_col = GrayImg.cols-j-1;
					k_window = get::getWindow (kernel, sz, r, k_diff_col);
				}
				else { k_window = get::getWindow (kernel, sz, r, r); }
			}

			cv::multiply (window, k_window, mulmat);
			p = cv::sum (mulmat)[0];
			imageOut.row(i).col(j) = p;

			count++;
			progress = round(count/total*10000)/100;
			std::cout << "\r" << progress << "%" << std::flush;
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
	std::cout << "Applying Sharpening Kernel...\n";
	cv::Mat k = (cv::Mat_<double>(3,3) << 0,-1, 0,
										 -1, 5,-1,
										  0,-1, 0);
	cv::Mat imageOut = this->applyKernel (k);
	std::cout << "   [DONE]\n";
	return imageOut;
}


cv::Mat imageKernel::blur ()
{
	std::cout << "Applying Blurring Kernel...\n";
	cv::Mat k = (cv::Mat_<double>(3,3) << 0.0625,0.125,0.0625,
										   0.125, 0.25, 0.125,
										  0.0625,0.125,0.0625);
	cv::Mat imageOut = this->applyKernel (k);
	std::cout << "   [DONE]\n";
	return imageOut;
}


cv::Mat imageKernel::emboss ()
{
	std::cout << "Applying Embossing Kernel...\n";
	cv::Mat k = (cv::Mat_<double>(3,3) << -2,-1, 0,
										  -1, 1, 1,
										   0, 1, 2);
	cv::Mat imageOut = this->applyKernel (k);
	std::cout << "   [DONE]\n";
	return imageOut;
}


cv::Mat imageKernel::topSobel ()
{
	std::cout << "Applying Sobel (Top) Kernel...\n";
	cv::Mat k = (cv::Mat_<double>(3,3) << 1, 2, 1,
										  0, 0, 0,
										 -1,-2,-1);
	cv::Mat imageOut = this->applyKernel (k);
	std::cout << "   [DONE]\n";
	return imageOut;
}


cv::Mat imageKernel::bottomSobel ()
{
	std::cout << "Applying Sobel (Bottom) Kernel...\n";
	cv::Mat k = (cv::Mat_<double>(3,3) << -1,-2,-1,
										   0, 0, 0,
										   1, 2, 1);
	cv::Mat imageOut = this->applyKernel (k);
	std::cout << "   [DONE]\n";
	return imageOut;
}


cv::Mat imageKernel::leftSobel ()
{
	std::cout << "Applying Sobel (Left) Kernel...\n";
	cv::Mat k = (cv::Mat_<double>(3,3) << 1, 0,-1,
										  2, 0,-2,
										  1, 0,-1);
	cv::Mat imageOut = this->applyKernel (k);
	std::cout << "   [DONE]\n";
	return imageOut;
}


cv::Mat imageKernel::rightSobel ()
{
	std::cout << "Applying Sobel (Right) Kernel...\n";
	cv::Mat k = (cv::Mat_<double>(3,3) << -1, 0, 1,
										  -2, 0, 2,
										  -1, 0, 1);
	cv::Mat imageOut = this->applyKernel (k);
	std::cout << "   [DONE]\n";
	return imageOut;
}


cv::Mat imageKernel::outline ()
{
	std::cout << "Applying Outlining Kernel...\n";
	cv::Mat k = (cv::Mat_<double>(3,3) << -1,-1,-1,
										  -1, 8,-1,
										  -1,-1,-1);
	cv::Mat imageOut = this->applyKernel (k);
	std::cout << "FINISHED\n";
	return imageOut;
}


cv::Mat imageKernel::smooth ()
{
	std::cout << "Applying Smoothing Kernel..." << "\n";
	cv::Mat k = (cv::Mat_<double>(3,3) << 1/9,1/9,1/9,
										  1/9,1/9,1/9,
										  1/9,1/9,1/9);
	cv::Mat imageOut = this->applyKernel (k);
	std::cout << "   [DONE]\n";
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

	cv::Mat k = (cv::Mat_<double>(3,3) << n[0],n[1],n[2],
										  n[3],n[4],n[5],
										  n[6],n[7],n[8]);
	std::cout << "Applying the following custom kernel:\n";
	std::cout << k << "...\n";
	cv::Mat imageOut = this->applyKernel (k);
	std::cout << "   [DONE]\n";

	return imageOut;
}


cv::Mat imageKernel::Gaussian (double sigma, int r) {
	int s = 2*r+1;
	std::cout << "Applying " << s << "x" << s << " Gaussian kernel...\n";
	
	double G, n, x, y;
	cv::Mat k = cv::Mat::zeros (s,s,CV_64FC1);
	for (int i=0; i<s; i++) {
		for (int j=0; j<s; j++) {
			x=i-r; y=j-r;
			n = -(x*x + y*y)/(2*sigma*sigma);
			G = 1/(2*M_PI*sigma*sigma) * exp(n);
			k.row(i).col(j) = G;
		}
	}

	cv::Mat imageOut = this->applyKernel (k);
	std::cout << "   [DONE]\n";

	return imageOut;
}


cv::Mat imageKernel::SelectAnOption (int option)
{
	cv::Mat imageOut;

	switch (option) {
		case 1:
			imageOut = this->sharpen();
			break;
		case 2:
			imageOut = this->blur();
			break;
		case 3:
			imageOut = this->emboss();
			break;
		case 4:
			imageOut = this->topSobel();
			break;
		case 5:
			imageOut = this->bottomSobel();
			break;
		case 6:
			imageOut = this->leftSobel();
			break;
		case 7:
			imageOut = this->rightSobel();
			break;
		case 8:
			imageOut = this->outline();
			break;
		case 9:
			imageOut = this->smooth();
			break;
		case 10:
			int r;
			double sigma;
			std::cout << "Enter window radius: ";
			std::cin >> r;
			std::cout << "Enter sigma value: ";
			std::cin >> sigma;
			imageOut = this->Gaussian(sigma, r);
			break;
		case 11:
			imageOut = this->customInput();
			break;
		default:
			std::cout << "ERROR: That is not an option.\n";
	}

	return imageOut;
}
