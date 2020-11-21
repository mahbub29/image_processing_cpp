#include <iostream>
#include "im1.hpp"
#include "im2.hpp"
#include "get.hpp"
#include "NLM.hpp"
#include "im_kernels.hpp"
#include <opencv2/opencv.hpp>




int main (int argc, char *argv[])
{
    std::string filePath = argv[1];
    // std::string a = argv[2]; int imType = std::stoi(a);
    // a = argv[3]; int r = std::stoi(a);

    get get_;
    imageProcess imp_ (filePath);
    histogramProcess histo_ (filePath);
    imageKernel kernel_ (filePath);
    NLM nlm_;



    char grayOrColor;
    bool noOptionSelected = true;



    std::cout << "Enter G for Grayscale or C for Color and press ENTER: ";
    while (noOptionSelected) {
        std::cin >> grayOrColor;
        if (grayOrColor == 'g' || grayOrColor == 'G' || grayOrColor == 'c' || grayOrColor == 'C') {
            noOptionSelected = false;
        } else {
            std::cout << "ERROR: That is not an option. Select G or C.\n";
        }
    }

    std::cout << "Enter the number of the process you would like to run and press ENTER.\n"
                 "1. Apply Median Filter\n"
                 "2. Apply Adaptive Filter\n"
                 "3. Run k-Means Segmentation\n"
                 "4. Get Equalized Image\n"
                 "5. Peform NLM denoising\n";
    if (grayOrColor == 'g' || grayOrColor == 'G') {
        std::cout << "6. Apply a kernel (options will be provided).\n";
    }

    int option, windowSize;
    noOptionSelected = true;
    cv::Mat out, colorPalette;

    while (noOptionSelected) {
        std::cin >> option;
        if (grayOrColor == 'g' || grayOrColor == 'G') {
            switch (option) {
                case 1: {
                    while (noOptionSelected) {
                        std::cout << "Enter window size (must be odd INT): ";
                        std::cin >> windowSize;
                        if (windowSize%2 != 1) {
                            std::cout << "ERROR: windowSize must be odd INT\n";
                        } else {
                            noOptionSelected = false;
                        }
                    }
                    out = imp_.medianFilterGray (windowSize);
                    break;
                }
                case 2: {
                    while (noOptionSelected) {
                        std::cout << "Enter window size (must be odd INT): ";
                        std::cin >> windowSize;
                        if (windowSize%2 != 1) {
                            std::cout << "ERROR: windowSize must be odd INT\n";
                        } else {
                            noOptionSelected = false;
                        }
                    }
                    out = imp_.adaptiveFilterGray (windowSize);
                    break;
                }
                case 3: {
                    std::vector<cv::Mat> imOut_n_paletteOut = imp_.kmeansSegmentation (imp_.GRAYSCALE, false);
                    out = imOut_n_paletteOut[0];
                    colorPalette = imOut_n_paletteOut[1];
                    noOptionSelected = false;
                    break;
                }
                case 4: {
                    std::cout << "Equalizing image...\n";
                    out = histo_.getEqualizedImage ();
                    std::cout << "Press any key to Quit.\n";
                    noOptionSelected = false;
                    break;
                }
                case 5: {
                    std::cout << "Please enter SPACE separated values for\n"
                                 "   h  sigma  patchRadius  windowRadius\n"
                                 "in that order\n"
                                 "Values: ";
                    std::vector<double> val(4);
                    for (int i=0; i<4; i++) { std::cin >> val[i]; }
                    cv::Mat gI = cv::imread (filePath, cv::IMREAD_GRAYSCALE);
                    double h=val[0]; double sigma=val[1]; int pR=val[2]; int wR=val[3];
                    out = nlm_.nonLocalMeans (gI,h,sigma,pR,wR);
                    noOptionSelected = false;
                    break;
                }
                case 6: {
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
                                 "10.Custom Input Kernel\n";
                    std::cout << "Option: ";
                    int K;
                    std::cin >> K;
                    out = kernel_.SelectAnOption(K);
                    noOptionSelected = false;
                    break;
                }
                default: {
                    std::cout << "ERROR: "<< option << " is not an option.\n";
                }
            }
        } else {
            switch (option) {
                case 1: {
                    while (noOptionSelected) {
                        std::cout << "Enter window size (must be odd INT): ";
                        std::cin >> windowSize;
                        if (windowSize%2 != 1) {
                            std::cout << "ERROR: windowSize must be odd INT\n";
                        } else {
                            noOptionSelected = false;
                        }
                    }
                    out = imp_.medianFilterRGB (windowSize);
                    break;
                }
                case 2: {
                    while (noOptionSelected) {
                        std::cout << "Enter window size (must be odd INT): ";
                        std::cin >> windowSize;
                        if (windowSize%2 != 1) {
                            std::cout << "ERROR: windowSize must be odd INT\n";
                        } else {
                            noOptionSelected = false;
                        }
                    }
                    out = imp_.adaptiveFilterColor (windowSize);
                    break;
                }
                case 3: {
                    std::vector<cv::Mat> imOut_n_paletteOut = imp_.kmeansSegmentation (imp_.COLOR, false);
                    out = imOut_n_paletteOut[0];
                    colorPalette = imOut_n_paletteOut[1];
                    noOptionSelected = false;
                    break;
                }
                case 4: {
                    std::cout << "Equalizing image...\n";
                    out = histo_.getEqlColor ();
                    noOptionSelected = false;
                    std::cout << "Press any key to Quit.\n";
                    break;
                }
                case 5: {
                    std::cout << "Please enter SPACE separated values for\n"
                                 "   h  sigma  patchRadius  windowRadius\n"
                                 "in that order\n"
                                 "Values: ";
                    std::vector<double> val(4);
                    for (int i=0; i<4; i++) { std::cin >> val[i]; }
                    cv::Mat cI = cv::imread (filePath, cv::IMREAD_COLOR);
                    double h=val[0]; double sigma=val[1]; int pR=val[2]; int wR=val[3];
                    out = nlm_.nonLocalMeans (cI,h,sigma,pR,wR);
                    noOptionSelected = false;
                    break;
                }
                default: {
                    std::cout << "ERROR: "<< option << " is not an option.\n";
                }
            }
        }
    }

    cv::Mat in;
    if (grayOrColor == 'g' || grayOrColor == 'G') {
        in = cv::imread (filePath, cv::IMREAD_GRAYSCALE);
    } else {
        in = cv::imread (filePath, cv::IMREAD_COLOR);
    }

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
