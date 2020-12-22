#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/mat.hpp>

#include <im1.hpp>
#include <im2.hpp>
#include <get.hpp>
#include <NLM.hpp>
#include <im_kernels.hpp>
#include <corner_detection.hpp>
#include <edge_detection.hpp>



int main (int argc, char *argv[])
{
    std::string process = argv[1];
    std::string filePath = argv[2];

    cv::Mat in, out, colorPalette;

    
    if (process == "optimize")
    {   
        bool useColor = std::stoi(argv[3]);

        imageProcess imp_ (filePath);
        histogramProcess histo_ (filePath);
        imageKernel kernel_;
        NLM nlm_;

        char grayOrColor;
        bool noOptionSelected = true;

        std::cout << "Enter the number of the process you would like to run and press ENTER.\n"
                    "1. Apply Median Filter\n"
                    "2. Apply Adaptive Filter\n"
                    "3. Run k-Means Segmentation\n"
                    "4. Get Equalized Image\n"
                    "5. Perform Histogram matching with a target image\n"
                    "6. Peform NLM denoising\n"
                    "7. Apply a kernel (options will be provided).\n";

        int option, windowRadius, windowSize;
        noOptionSelected = true;

        while (noOptionSelected) {
            std::cin >> option;

            switch (option) {
                case 1: {
                    std::cout << "Enter window radius: ";
                    std::cin >> windowRadius; windowSize = 2*windowRadius+1;

                    if (useColor) out = imp_.medianFilterRGB (windowSize);
                    else out = imp_.medianFilterGray (windowSize);

                    noOptionSelected = false;

                    break;
                }
                case 2: {
                    std::cout << "Enter window radius: ";
                    std::cin >> windowRadius; windowSize = 2*windowRadius+1;

                    if (useColor) out = imp_.adaptiveFilterColor (windowSize);
                    else out = imp_.adaptiveFilterGray (windowSize);

                    noOptionSelected = false;

                    break;
                }
                case 3: {
                    std::vector<cv::Mat> imOut_n_paletteOut;

                    if (useColor) imOut_n_paletteOut = imp_.kmeansSegmentation (imp_.COLOR, false);
                    else imOut_n_paletteOut = imp_.kmeansSegmentation (imp_.GRAYSCALE, false);

                    out = imOut_n_paletteOut[0];
                    colorPalette = imOut_n_paletteOut[1];
                    noOptionSelected = false;

                    break;
                }
                case 4: {
                    std::cout << "Equalizing image...\n";
                    
                    if (useColor) out = histo_.getEqlColor ();
                    else out = histo_.getEqualizedImage ();
                    
                    std::cout << "Press any key to Quit.\n";
                    noOptionSelected = false;

                    break;
                }
                case 5: {
                    std::cout << "Enter the path the target image:\n";
                    std::string targetPath;
                    std::cin >> targetPath;

                    if (useColor) out = histo_.matchHistogram (targetPath, true);
                    else out = histo_.matchHistogram (targetPath, false);
                    
                    noOptionSelected = false;
                    
                    break;
                }
                case 6: {
                    std::cout << "Please enter SPACE separated values for\n"
                                "   h  sigma  patchRadius  windowRadius\n"
                                "in that order\n"
                                "Values: ";
                    std::vector<double> val(4);

                    for (int i=0; i<4; i++) { std::cin >> val[i]; }
                    
                    cv::Mat img;
                    if (useColor) img = cv::imread (filePath, cv::IMREAD_COLOR); 
                    else img = cv::imread (filePath, cv::IMREAD_GRAYSCALE);

                    double h=val[0]; double sigma=val[1]; int pR=val[2]; int wR=val[3];
                    out = nlm_.nonLocalMeans (img,h,sigma,pR,wR);

                    noOptionSelected = false;

                    break;
                }
                case 7: {
                    std::cout << "Select which kernel to apply:\n"
                                "1. Sharpen\n"
                                "2. Blur\n"
                                "3. Emboss\n"
                                "4. Top Sobel\n"
                                "5. Bottom Sobel\n"
                                "6. Left Sobel\n"
                                "7. Right Sobel\n"
                                "8. Outline\n"
                                "9. Smooth\n"
                                "10. Gaussian\n"
                                "11.Custom Input Kernel\n";
                    std::cout << "Option: ";
                    int K;
                    std::cin >> K;

                    cv::Mat img;
                    if (useColor) {
                        img = cv::imread (filePath, cv::IMREAD_COLOR);
                        out = kernel_.SelectAnOption(img, K);
                    } else {
                        img = cv::imread (filePath, cv::IMREAD_GRAYSCALE);
                        out = kernel_.SelectAnOption(img, K);
                    }

                    noOptionSelected = false;

                    break;
                }
                default: {
                    std::cout << "ERROR: "<< option << " is not an option.\n";
                }
            }
        }

        
        if (useColor) {
            in = cv::imread (filePath, cv::IMREAD_COLOR);
        } else {
            in = cv::imread (filePath, cv::IMREAD_GRAYSCALE);
        }

    }
    else if (process == "dcorner")
    {
        corner_detection corner_(filePath);

        in = cv::imread (filePath, cv::IMREAD_GRAYSCALE);
        out = corner_.harrisDetect (1);
    }
    else if (process == "dedge")
    {
        edge_detection edge_(filePath);

        in = cv::imread (filePath, cv::IMREAD_GRAYSCALE);
        out = edge_.sobelEdgeDetect (false);
    }


    // output windows
    if (!out.empty()) {
        cv::namedWindow("Original", cv::WINDOW_NORMAL);
        cv::namedWindow("Output", cv::WINDOW_NORMAL);
        cv::resizeWindow("Original",640,480);
        cv::resizeWindow("Output",640,480);
        cv::imshow ("Original", in);
        cv::imshow ("Output", out);
    }

    if (!colorPalette.empty()) {
        cv::namedWindow("k-Means Color Palette", cv::WINDOW_NORMAL);
        cv::resizeWindow("k-Means Color Palette", 640, 250);
        cv::imshow ("k-Means Color Palette", colorPalette);
    }

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
