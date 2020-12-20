#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/mat.hpp>

#include <corner_detection.hpp>
#include <get.hpp>
#include <im_kernels.hpp>



corner_detection::corner_detection (std::string IMAGE_FILE_PATH)
    : grayImg (get::getGrayImg(IMAGE_FILE_PATH))
    {}


cv::Mat corner_detection::movarecDetect (int r, double sigma, double threshold)
{   
    cv::Mat img; grayImg.convertTo (img, CV_64FC1);

    cv::Mat k = imageKernel::Gaussian (sigma, r); k = k/cv::sum(k)[0];
    cv::Mat cornerMap = cv::Mat::zeros (img.size(), CV_64FC1);
    
    int bound = k.rows-1;

    cv::Mat window, p, q, ssdMat;
    double ssd;

    cv::Mat SSD = cv::Mat::zeros (1, k.rows*k.cols, CV_64FC1);
    cv::Mat SSDsqr;
    
    double minVal, maxVal;
	cv::Point2i minIdx, maxIdx;
    int count;
    double cornerness;

    double total = (img.rows-2*bound)*(img.cols-2*bound);
    double progress;
    int progress_count = 0;

    for (int i=bound; i<img.rows-bound; i++) {
        for (int j=bound; j<img.cols-bound; j++) {
            window = get::getWindow (img, k.rows+bound, i, j);
            
            p = get::getWindow (window, k.rows, bound, bound);
            count = 0;
            for (int x=bound/2; x<window.rows-bound/2; x++) {
                for (int y=bound/2; y<window.cols-bound/2; y++) {
                    if (x==bound && y==bound) {
                        SSD.row(0).col(count) = 1000000;
                    }
                    else{
                        q = get::getWindow (window, k.rows, x, y);
                        cv::pow (p-q, 2, ssdMat);
                        cv::multiply (k, ssdMat, ssdMat);
                        ssd = cv::sum (ssdMat)[0];
                        SSD.row(0).col(count) = ssd;
                    }

                    count++;
                }
            }

            cv::minMaxLoc (SSD, &minVal, &maxVal, &minIdx, &maxIdx);
            cornerness = minVal;
            if (cornerness>threshold) cornerMap.row(i).col(j) = 255;
        
            progress_count++;
			progress = round(progress_count/total*10000)/100;
			std::cout << "\r" << progress << "%" << std::flush;
        }
    }

    cv::Mat out; cornerMap.convertTo (out, CV_8UC1);
    
    return out;
}