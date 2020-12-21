#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

#include <corner_detection.hpp>
#include <get.hpp>
#include <im_kernels.hpp>



corner_detection::corner_detection (std::string IMAGE_FILE_PATH)
    : grayImg (get::getGrayImg(IMAGE_FILE_PATH))
    {}


cv::Mat corner_detection::movarecDetect (int r, bool redOverlay, double threshold)
{   
    cv::Mat img; grayImg.convertTo (img, CV_64FC1);

    cv::Mat k = get::getGaussianWeightingKernel (r);
    cv::Mat cornerMap = cv::Mat::zeros (img.size(), CV_64FC1);

    // *********************
    cv::Mat rgbGray(grayImg.size(), CV_8UC3), chans[3];
    cv::cvtColor (grayImg, rgbGray, CV_GRAY2RGB);
    rgbGray.convertTo (rgbGray, CV_64FC3); cv::split (rgbGray, chans);
    // *********************
    
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
            if (cornerness>threshold) {
                cornerMap.row(i).col(j) = 255;

                if (redOverlay) {
                    chans[0].row(i).col(j) = 0; chans[0].row(i-1).col(j) = 0; chans[0].row(i+1).col(j) = 0; chans[0].row(i).col(j-1) = 0; chans[0].row(i).col(j+1) = 0;
                    chans[1].row(i).col(j) = 0; chans[1].row(i-1).col(j) = 0; chans[1].row(i+1).col(j) = 0; chans[1].row(i).col(j-1) = 0; chans[1].row(i).col(j+1) = 0;
                    chans[2].row(i).col(j) = 255; chans[2].row(i-1).col(j) = 255; chans[2].row(i+1).col(j) = 255; chans[2].row(i).col(j-1) = 255; chans[2].row(0).col(j+1) = 255;
                }
            }
        
            progress_count++;
			progress = round(progress_count/total*10000)/100;
			std::cout << "\r" << progress << "%" << std::flush;
        }
    }

    cv::Mat out; 
    if (redOverlay) {
        std::vector<cv::Mat> chanVec = {chans[0], chans[1], chans[2]};
        cv::Mat cornerMap2; cv::merge (chanVec, cornerMap2);
        cornerMap2.convertTo (out, CV_8UC3);
    } else {
        cornerMap.convertTo (out, CV_8UC1);
    }

    return out;
}



cv::Mat corner_detection::harrisDetect (int r, bool redOverlay,  double threshold)
{
    cv::Mat img; grayImg.convertTo (img, CV_64FC1);

    cv::Mat cornerMap = cv::Mat::zeros (img.size(), CV_64FC1);

    // CORNER OVERLAY MAP *********
    cv::Mat rgbGray(grayImg.size(), CV_8UC3), chans[3];
    cv::cvtColor (grayImg, rgbGray, CV_GRAY2RGB);
    rgbGray.convertTo (rgbGray, CV_64FC3); cv::split (rgbGray, chans);
    // ****************************

    cv::Mat Ix, Iy, Ixy, p, Mx, My;
    double I_x2, I_y2, I_xy, cornerness;
    Mx = cv::Mat::zeros (2*r+1, 2*r+1, CV_64FC1);
    My = cv::Mat::zeros (2*r+1, 2*r+1, CV_64FC1);

    int bound = r+1;
    int windowSize = 2*(r+1)+1;

    int count = 0;
    double progress;
    double total = (img.rows-2*bound)*(img.cols-2*bound);
    int num_elements = windowSize*windowSize;

    for (int i=bound; i<img.rows-bound; i++) {
        for (int j=bound; j<img.cols-bound; j++) {
            p = get::getWindow (img, windowSize, i, j);

            if (abs(img.at<double>(i,j) - cv::sum(p)[0]/num_elements)>100) count++;
            else {
                for (int y=1; y<p.rows-1; y++) {
                    for (int x=1; x<p.cols-1; x++) {
                        Mx.row(y-1).col(x-1) = 0.5*(p.row(y).col(x+1) - p.row(y).col(x-1));
                        My.row(y-1).col(x-1) = 0.5*(p.row(y+1).col(x) - p.row(y-1).col(x));
                    }
                }

                cv::multiply (Mx, Mx, Ix);  I_x2 = cv::sum(Ix)[0];
                cv::multiply (My, My, Iy);  I_y2 = cv::sum(Iy)[0];
                cv::multiply (Mx, My, Ixy); I_xy = cv::sum(Ixy)[0];
        
                if (I_x2 + I_y2 == 0) cornerness = 0;
                else cornerness = (I_x2*I_y2 - I_xy*I_xy) / (I_x2 + I_y2);

                if (cornerness > threshold) {
                    cornerMap.row(i).col(j) = 255;

                    if (redOverlay) {
                        chans[0].row(i).col(j) = 0; chans[0].row(i-1).col(j) = 0; chans[0].row(i+1).col(j) = 0; chans[0].row(i).col(j-1) = 0; chans[0].row(i).col(j+1) = 0;
                        chans[1].row(i).col(j) = 0; chans[1].row(i-1).col(j) = 0; chans[1].row(i+1).col(j) = 0; chans[1].row(i).col(j-1) = 0; chans[1].row(i).col(j+1) = 0;
                        chans[2].row(i).col(j) = 255; chans[2].row(i-1).col(j) = 255; chans[2].row(i+1).col(j) = 255; chans[2].row(i).col(j-1) = 255; chans[2].row(0).col(j+1) = 255;
                    }
                }

                count++;
            }

            
			progress = round(count/total*10000)/100;
			std::cout << "\r" << progress << "%" << std::flush;
        }
    }
    std::cout << count << "\n";

    cv::Mat out;
    if (redOverlay) {
        std::vector<cv::Mat> chanVec = {chans[0], chans[1], chans[2]};
        cv::Mat cornerMap2; cv::merge (chanVec, cornerMap2);
        cornerMap2.convertTo (out, CV_8UC3);
    } else {
        cornerMap.convertTo (out, CV_8UC1);
    }
            
    return out;

}