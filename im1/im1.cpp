
#include <iostream>
#include "im1.hpp"
#include "get.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>


std::vector< std::vector<int> > LEFT_CLICKS;
int n=0;

int imageProcess::getMedian (cv::Mat window)
{	
	cv::Mat window_thread = cv::Mat::zeros(1, window.rows*window.cols, CV_8UC1);
	window_thread = window.clone().reshape(0, window.rows*window.cols);
	cv::sort(window_thread, window_thread, CV_SORT_ASCENDING);

	int median;
	if ((window.rows*window.cols)%2==1) {
		median = window_thread.at<uchar>(floor(window_thread.rows/2)-1);
	} else {
		median = (window_thread.at<uchar>(floor(window_thread.rows/2)-1) + 
				  window_thread.at<uchar>(floor(window_thread.rows/2)))/2;
	}

	return median;
}


cv::Mat imageProcess::medianFilter (cv::Mat image, int windowSize)
{	
	if (windowSize%2!=1) {
		std::cout << "ERROR: windowSize must be odd int" << std::endl;
	} else {
		int height = image.rows;
		int width = image.cols;

		cv::Mat window = cv::Mat::zeros(windowSize, windowSize, CV_64FC1);
		cv::Mat imageOut = cv::Mat::zeros(height, width, CV_64FC1);
		double count = 0;
		double progress;
		double last_progress = 1000;
		int total = image.rows*image.cols;

		std::cout << "Applying Median Filter to channel...\n";

		for (int i=0; i<height; i++) {
			for (int j=0; j<width; j++) {
				window = get::getWindow(image, windowSize, i, j);
				imageOut.row(i).col(j) = this->getMedian(window);
				
				count++;
				progress = floor(count/total*100);
				std::cout << "\r" << progress << "%" << std::flush;
				last_progress = progress;
			}
		}
		std::cout << " - DONE\n";

		imageOut.convertTo(imageOut, CV_8UC1);
		return imageOut;
	}
}


cv::Mat imageProcess::adaptiveFilter(cv::Mat image, int windowSize)
{
	int height = image.rows;
	int width = image.cols;

	cv::Mat mean = get::getMeans (image, windowSize);
	cv::Mat variance = get::getVariances (image, windowSize, mean);

	cv::Mat_<double> window;
	cv::Mat var_window = cv::Mat::zeros (windowSize, windowSize, CV_64FC1);
	cv::Mat imageOut = cv::Mat::zeros (image.rows, image.cols, CV_64FC1);

	int p; // pixel(i,j) i.e. p(i,j)
	double m, // mean at window centered at p(i,j)
		   v, // variance at window centered at p(i,j)
	 	  lv; // average local variance of window centered aroung p(i,j)

	double count = 0;
	double progress;
	double last_progress = 1000;
	int total = image.rows*image.cols;
	std::string dots = "";

	std::cout << "Appying Adaptive Filter to channel\n";

	for (int i=0; i<height; i++) {
		for (int j=0; j<width; j++) {
			var_window = get::getWindow (variance, windowSize, i, j);

			p = image.at<uchar>(i,j);
			m = mean.at<double>(i,j);
			v = variance.at<double>(i,j);

			lv = cv::sum(var_window)[0]/(var_window.rows*var_window.cols);

			imageOut.row(i).col(j) = round(m + (v-lv)/v * (p-m));

			count++;
			progress = floor(count/total*100);
			if (progress != last_progress){
				if (std::fmod(progress,5)==0) {dots += ".";}
				std::cout << "\r" << progress << "%" << dots << std::flush;
				last_progress = progress;
			}
		}
	}
	std::cout << "DONE\n";

	imageOut.convertTo(imageOut, CV_8UC1);

	return imageOut;
}


imageProcess::imageProcess (std::string IMAGE_FILE_PATH)
	: GrayImg(get::getGrayImg(IMAGE_FILE_PATH)), ColorImg(get::getColorImg(IMAGE_FILE_PATH))
	{
		int height = GrayImg.rows;
		int width = GrayImg.cols;
	}


cv::Mat imageProcess::medianFilterGray (int windowSize)
{
	std::cout << "Processing GRAY Median Filter" << std::endl;
	cv::Mat imageOut = medianFilter(GrayImg, windowSize);
	std::cout << "FINISHED" << "\n";

	return imageOut;
}


cv::Mat imageProcess::medianFilterRGB (int windowSize)
{	
	cv::Mat bgr[3]; // Split image into BGR color channels
	cv::split (ColorImg, bgr); // split the color channels int the image
	cv::Mat blue = bgr[0];
	cv::Mat green = bgr[1];
	cv::Mat red = bgr[2];

	std::cout << "Processing RGB Median Filter" << std::endl;
	std::cout << "Blue Channel" << "\n";
	cv::Mat blueOut = this->medianFilter(blue, windowSize);
	std::cout << "Green Channel" << "\n";
	cv::Mat greenOut = this->medianFilter(green, windowSize);
	std::cout << "Red Channel" << "\n";
	cv::Mat redOut = this->medianFilter(red, windowSize);
	std::cout << "FINISHED" << "\n";

	cv::Mat imageChannels[3] = {blueOut, greenOut, redOut};
	cv::Mat imageOut(bgr[0].size(), CV_8UC3);
	cv::merge (imageChannels, 3, imageOut);

	return imageOut;
}


cv::Mat imageProcess::adaptiveFilterGray (int windowSize)
{
    std::cout << "Processing GRAY Adaptive Filter" << std::endl;
    cv::Mat imageOut = this->adaptiveFilter(GrayImg, windowSize);
    std::cout << "FINISHED" << "\n";

    return imageOut;
}


cv::Mat imageProcess::adaptiveFilterColor (int windowSize)
{
	cv::Mat bgr[3]; // Split image into BGR color channels
	cv::split (ColorImg, bgr); // split the color channels int the image
	cv::Mat blue = bgr[0];
	cv::Mat green = bgr[1];
	cv::Mat red = bgr[2];

	std::cout << "Processing RGB Adaptive Filter" << std::endl;
	std::cout << "Blue Channel" << "\n";
	cv::Mat blueOut = this->adaptiveFilter(blue, windowSize);
	std::cout << "Green Channel" << "\n";
	cv::Mat greenOut = this->adaptiveFilter(green, windowSize);
	std::cout << "Red Channel" << "\n";
	cv::Mat redOut = this->adaptiveFilter(red, windowSize);
	std::cout << "FINISHED" << "\n";

	cv::Mat imageChannels[3] = {blueOut, greenOut, redOut};
	cv::Mat imageOut(bgr[0].size(), CV_8UC3);
	cv::merge (imageChannels, 3, imageOut);

	return imageOut;	
}


void imageProcess::kmeansSegmentation (int selection)
{	
	cv::Mat IMG;

	if (selection == 0) {
		std::cout << "Press 1 for Grayscale Segmentation.\n"
				 "Press 2 for Color Segmentation.\n";

		bool keepAsking = true;
		
		while (keepAsking) {
			char option;
			std::cin >> option;

			if (option=='1') {
				IMG=GrayImg;
				keepAsking = false;
			}
			else if (option=='2') {
				IMG=ColorImg;
				keepAsking = false;
			}
			else {
				std::cout << "ERROR: That is not an option. Please press 1 for Grayscale "
							 "Segmentation or 2 for Color Segmentation.\n";
			}
		}
	} else if (selection == 1) {
		IMG = GrayImg;
	} else if (selection == 2) {
		IMG = ColorImg;
	}	

	cv::namedWindow ("Select Pixels", cv::WINDOW_NORMAL);
	cv::resizeWindow ("Select Pixels", 640, 480);
	cv::Mat xyz;
	cv::setMouseCallback ("Select Pixels", this->leftMouseClick, &xyz);
	cv::imshow ("Select Pixels", IMG);

	std::cout << "Select a minimum of 2 points.\n"
				 "Press R to reset and clear your selection.\n"
				 "Press V to verify your selection.\n"
				 "Press Q to quit.\n\n";

	char key1;
	while (1) {
		key1 = cv::waitKey(1);

		if (key1=='q') {
			cv::destroyAllWindows();
			break;
		}
		else if (key1=='r') {
			LEFT_CLICKS.clear();
			std::cout << "Pixel selections cleared.\n"
						 "Please select minimum of 2 points in the image window, "
						 "or press Q to quit." << "\n";
		} else if (key1=='v') {
			if (LEFT_CLICKS.size()<2) {
				std::cout << "ERROR: Select at least 2 points, or press R to clear "
				"the list or \"Q\" to quit." << "\n";
			} else {
				std::cout << "You have selected " << LEFT_CLICKS.size() << " points. "
							 "Is this correct?\nPress Y to confirm, R to clear the list, "
							 "or continue clicking to add to your selection." << "\n";
			}
		} else if (key1=='y' && LEFT_CLICKS.size()>1) {

			cv::Mat i_vec; // Matrix of intensity vectors
			int ndims;

			std::cout << "Starting k-Means Segmentation..." << "\n";
			if (IMG.channels() == 1) {
				// retrieve Grayscale intensities
				i_vec = get::get_nD_intensities (IMG, LEFT_CLICKS, 1);
				bool selectingDim = true;
				while (selectingDim) {
					std::cout << "Enter 1 for GRAY (1D) k-Means, or enter 3 for GRAYij (GRAY-3D) k-Means: ";
					std::cin >> ndims;
					if (ndims==1 || ndims==3) {
						selectingDim = false;
					} else {
						std::cout << "ERROR: That is not an option. Please enter either 1 or 3.\n";
					}

				}
			}
			else {
				bool selectingDim = true;
				while (selectingDim) {
					std::cout << "Enter 3 for RGB (COLOR-3D) k-Means, or enter 5 (COLOR-5D) for RGBij k-Means: ";
					std::cin >> ndims;
					if (ndims==3 || ndims==5) {
						selectingDim = false;
					} else {
						std::cout << "ERROR: That is not an option. Please enter either 3 or 5.\n";
					}
				}
			}

			// retrieve RGB or RGBij intensities
			i_vec = get::get_nD_intensities (IMG, LEFT_CLICKS, ndims);
			std::cout << i_vec << "\n";

			// conduct segmentation
			std::cout << "Counducting segmentation...\n";
			cv::Mat imageOut = this->kmeansConvergence (IMG.channels(), i_vec, ndims);

			cv::namedWindow("Output", cv::WINDOW_NORMAL);
			cv::resizeWindow("Output", 640, 480);
			cv::imshow("Output", imageOut);
			key1 = cv::waitKey(0);
			cv::destroyAllWindows();
			break;
		}
	}

	return;
}


void imageProcess::leftMouseClick (int event, int j, int i, int flags, void *param)
{
	cv::Mat &xyz = *((cv::Mat*)param);
	if (event==cv::EVENT_LBUTTONDOWN) {
		std::cout << "row=" << i << ", col=" << j << "\n";
		std::vector<int> v;
		v.push_back(i);
		v.push_back(j);
		LEFT_CLICKS.push_back(v);
		return;
	}
}


std::vector<cv::Mat> imageProcess::kmeans_nD_segmentation (int channels, cv::Mat i_vec, int ndims)
{
	std::vector<cv::Mat> iVec_n_imageOut;

	if (channels == 3) // *** FOR COLOR ***
	{
		// number of k values
		int k_num = i_vec.cols;

		// split image matrix to BGR channels
		cv::Mat imageThread = ColorImg.clone().reshape(3,1);
		
		cv::Mat bgrThread[ndims];
		cv::split(imageThread, bgrThread);

		// convert image matrix to doubles
		for (int i=0; i<3; i++) {
	        bgrThread[i].convertTo(bgrThread[i], CV_64FC1);
	    }

	    // add i and j dimensions if ndims requested is 5D
	    if (ndims==5) {
	    	bgrThread[3] = cv::Mat::zeros(bgrThread[0].size(), CV_64FC1);
	    	bgrThread[4] = cv::Mat::zeros(bgrThread[0].size(), CV_64FC1);
	    	int count = 0;
	    	for (int i=0; i<ColorImg.rows; i++) {
		        for (int j=0; j<ColorImg.cols; j++) {
		            bgrThread[3].at<double>(0,count) = i;
		            bgrThread[4].at<double>(0,count) = j;
		            count++;
		        }
		    }
	    }

	    cv::Mat imagePixelVectors;
	    std::vector<cv::Mat> m;
	    if (ndims==5) {
	    	m = {bgrThread[0], // B displacement
	    		 bgrThread[1], // G displacement
	    		 bgrThread[2], // R displacement
	    		 bgrThread[3], // i displacement
	    		 bgrThread[4]};// j displacement
	    } else {
	    	m = {bgrThread[0], // B displacement
	    		 bgrThread[1], // G displacement
	    		 bgrThread[2]};// R displacement
	    }
	    cv::vconcat(m, imagePixelVectors);

	    // Make vectors to store the cluster sums as well as the total nymber
		// belonging to each cluster 
		std::vector< cv::Mat_<double> > k_sums(k_num);
		std::vector<int> k_tots(k_num);

		// Matrix to store the difference between each image pixel and the currently
		// calculated k-means values
		cv::Mat_<double> delta = cv::Mat::zeros (k_num, imageThread.cols, CV_64FC1);

		// calculate the root square distance (in terms of RGBij) each pixel in the image
	   	// to each of the selected pixel RGB values
	   	// i.e. d = sqrt(r^2 + g^2 + b^2 + i^2 + j^2)
	    // assign a k cluster number to each set of RGB differences

		cv::Mat sqrDiff; // matrix to save the squared differeneces
		cv::Mat pixelLabels = cv::Mat::ones(k_num, imageThread.cols, CV_32FC1); // array containing the k number of labels

		std::cout << "Calculating vector distances...";
		for (int i=0; i<k_num; i++) {
			if (ndims==5) {
				k_sums[i] = cv::Mat::zeros (5,1,CV_64FC1);
			} else {
				k_sums[i] = cv::Mat::zeros (3,1,CV_64FC1);
			}
			k_tots[i] = 0;

			cv::Mat I = cv::Mat::zeros(imagePixelVectors.size(), CV_64FC1); 
			for (int n=0; n<imagePixelVectors.cols; n++) {
				i_vec.col(i).copyTo(I.col(n));
			}

			cv::pow((imagePixelVectors-I), 2, sqrDiff); // calculate squared differences
			cv::reduce(sqrDiff, delta.row(i), 0, cv::REDUCE_SUM, CV_64FC1); // sum the squared differnces
			cv::sqrt(delta.row(i), delta.row(i)); // square-root of the sum
			pixelLabels.row(i) = pixelLabels.row(i)*(i+1); // initialise a set of labels for each row of delta
														   // to use to identify the lowest value in each column later
		}
		std::cout << "DONE\n";

		std::cout << "Labelling Pixels...";
		// initialise an array to contain the final pixel cluster identities
		cv::Mat pixelLabelsFinal = cv::Mat::zeros (1, imageThread.cols, CV_32F);
		// find the lowest delta_rgbij out of the k groups and label the pixel as
	    // belonging to the k value with the lowest delta rgb value
	    double minVal, maxVal;
	    cv::Point2i minIdx, maxIdx;
	    int IDX;
	    for (int i=0; i<delta.cols; i++) {
	    	cv::minMaxLoc (delta.col(i), &minVal, &maxVal, &minIdx, &maxIdx);
	    	IDX = minIdx.y;
	    	pixelLabelsFinal.at<int>(0,i) = IDX;

	    	// increment the total number of pixels in the cluster by one
	    	k_tots[IDX]++;

	    	// add the actual pixel intensity to the cluster dictionary item
	    	k_sums[IDX] += imagePixelVectors.col(i);
	    }
	    std::cout << "DONE\n";

		// calculate the new average pixel intensity vector
		for (int i=0; i<i_vec.cols; i++) {
			i_vec.col(i) = k_sums[i] / k_tots[i];
		}

		// get the output image
		std::cout << "Making output image...";
		int pL;
		int count = 0;
		cv::Mat bgrOut = cv::Mat::zeros (3, imageThread.cols, CV_64FC1);
		for (int i=0; i<imageThread.cols; i++) {
			pL = pixelLabelsFinal.at<int>(0,i);
			i_vec.col(pL).rowRange(0,3).copyTo(bgrOut.col(i));
		}

		// get the output image
		cv::Mat imageChannels[3];
		imageChannels[0] = bgrOut.row(0); // blue
		imageChannels[1] = bgrOut.row(1); // green
		imageChannels[2] = bgrOut.row(2); // red

		cv::Mat imageOut;
		cv::merge (imageChannels, 3, imageOut);
		imageOut = imageOut.reshape(3, ColorImg.rows);
		imageOut.convertTo(imageOut, CV_8UC3);

		// create an array of matrices to store the new intensity vector and the output image
		iVec_n_imageOut = {i_vec, imageOut};
	}

	//---------------------------------------------------------------------------------------------

	else // *** FOR GRAYSCALE ***
	{
		// number of k values
		int k_num = i_vec.cols;

		// split image matrix to BGR channels
		cv::Mat grayThread[ndims] = GrayImg.clone().reshape(1,1);

		// convert image matrix to doubles
	    grayThread[0].convertTo(grayThread[0], CV_64FC1);

	    // add i and j dimensions if ndims requested is 3D
	    if (ndims==3) {
	    	grayThread[1] = cv::Mat::zeros(grayThread[0].size(), CV_64FC1);
	    	grayThread[2] = cv::Mat::zeros(grayThread[0].size(), CV_64FC1);
	    	int count = 0;
	    	for (int i=0; i<GrayImg.rows; i++) {
		        for (int j=0; j<GrayImg.cols; j++) {
		            grayThread[1].at<double>(0,count) = i;
		            grayThread[2].at<double>(0,count) = j;
		            count++;
		        }
		    }
	    }

		cv::Mat imagePixelVectors;
	    std::vector<cv::Mat> m;
	    if (ndims==3) {
	    	m = {grayThread[0], // GRAY displacement
	    		 grayThread[1], // i displacement
	    		 grayThread[2]};// j displacement
	    } else {
	    	m = {grayThread[0]};// GRAY displacement
	    }
	    cv::vconcat(m, imagePixelVectors);

	    // Make vectors to store the cluster sums as well as the total nymber
		// belonging to each cluster 
		std::vector< cv::Mat_<double> > k_sums(k_num);
		std::vector<int> k_tots(k_num);

		// Matrix to store the difference between each image pixel and the currently
		// calculated k-means values
		cv::Mat_<double> delta = cv::Mat::zeros (k_num, grayThread[0].cols, CV_64FC1);
	    
		cv::Mat sqrDiff; // matrix to save the squared differeneces
		cv::Mat pixelLabels = cv::Mat::ones(k_num, grayThread[0].cols, CV_32FC1); // array containing the k number of labels

		std::cout << "Calculating vector distances...";
		for (int i=0; i<k_num; i++) {
			if (ndims==3) {
				k_sums[i] = cv::Mat::zeros (3,1,CV_64FC1);
			} else {
				k_sums[i] = cv::Mat::zeros (1,1,CV_64FC1);
			}
			k_tots[i] = 0;

			cv::Mat I = cv::Mat::zeros(imagePixelVectors.size(), CV_64FC1); 
			for (int n=0; n<imagePixelVectors.cols; n++) {
				i_vec.col(i).copyTo(I.col(n));
			}

			cv::pow((imagePixelVectors-I), 2, sqrDiff); // calculate squared differences
			cv::reduce(sqrDiff, delta.row(i), 0, cv::REDUCE_SUM, CV_64FC1); // sum the squared differnces
			cv::sqrt(delta.row(i), delta.row(i)); // square-root of the sum
			pixelLabels.row(i) = pixelLabels.row(i)*(i+1); // initialise a set of labels for each row of delta
														   // to use to identify the lowest value in each column later
		}
		std::cout << "DONE\n";

		std::cout << "Labelling Pixels...";
		// initialise an array to contain the final pixel cluster identities
		cv::Mat pixelLabelsFinal = cv::Mat::zeros (1, grayThread[0].cols, CV_32F);
		// find the lowest delta_rgbij out of the k groups and label the pixel as
	    // belonging to the k value with the lowest delta rgb value
	    double minVal, maxVal;
	    cv::Point2i minIdx, maxIdx;
	    int IDX;
	    for (int i=0; i<delta.cols; i++) {
	    	cv::minMaxLoc (delta.col(i), &minVal, &maxVal, &minIdx, &maxIdx);
	    	IDX = minIdx.y;
	    	pixelLabelsFinal.at<int>(0,i) = IDX;

	    	// increment the total number of pixels in the cluster by one
	    	k_tots[IDX]++;

	    	// add the actual pixel intensity to the cluster dictionary item
	    	k_sums[IDX] += imagePixelVectors.col(i);
	    }
	    std::cout << "DONE\n";

	    // calculate the new average pixel intensity vector
		for (int i=0; i<i_vec.cols; i++) {
			i_vec.col(i) = k_sums[i] / k_tots[i];
		}

		// get the output image
		std::cout << "Making output image...";
		int pL;
		int count = 0;
		cv::Mat grayOut = cv::Mat::zeros (1, grayThread[0].cols, CV_64FC1);
		for (int i=0; i<grayThread[0].cols; i++) {
			pL = pixelLabelsFinal.at<int>(0,i);
			i_vec.col(pL).rowRange(0,1).copyTo(grayOut.col(i));
		}

		cv::Mat imageOut;
		grayOut = grayOut.reshape(1, GrayImg.rows);
		grayOut.convertTo(imageOut, CV_8UC1);

		// create an array of matrices to store the new intensity vector and the output image
		iVec_n_imageOut = {i_vec, imageOut};
	}

	std::cout << "DONE\n";

	return iVec_n_imageOut;
}


cv::Mat imageProcess::kmeansConvergence (int channels, cv::Mat init_i_vec, int ndims)
{
	// set arbitray initial previous intensity vector to compare to
	cv::Mat prev_iVec = cv::Mat::ones (init_i_vec.size(), CV_64FC1)*1000;
	cv::Mat curr_iVec = init_i_vec;

	std::vector<cv::Mat> iVec_n_imageOut;
	cv::Mat imageOut;
	double SUMcurr, SUMprev;
	SUMcurr = cv::sum(curr_iVec)[0];
	SUMprev = cv::sum(prev_iVec)[0];

	while (SUMcurr != SUMprev) {
		prev_iVec = curr_iVec;
		iVec_n_imageOut = this->kmeans_nD_segmentation (channels, curr_iVec, ndims);
		curr_iVec = iVec_n_imageOut[0];
		imageOut = iVec_n_imageOut[1];
		SUMcurr = cv::sum(curr_iVec)[0];
		SUMprev = cv::sum(prev_iVec)[0];
	}

	if (channels == 3) {
		for (int i=0; i<curr_iVec.cols; i++) {
			std::cout << "Color " << i+1 << " = R:" << round(curr_iVec.at<double>(2,i))
										   << " G:" << round(curr_iVec.at<double>(1,i))
										   << " B:" << round(curr_iVec.at<double>(0,i)) << "\n";
		}
	} else {
		for (int i=0; i<curr_iVec.cols; i++) {
			std::cout << "Grayscale intensity: " << i+1 << " = " << round(curr_iVec.at<double>(0,i)) << "\n";
		}
	}

	std::cout << "FINISHED\n"
				 "Click console and press Ctrl+C to quit.\n";

	return imageOut;
}