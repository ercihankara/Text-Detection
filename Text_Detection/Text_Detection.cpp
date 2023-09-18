#include "stdafx.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "string"
#include "fstream"

using namespace cv;
using namespace std;

// perform morphological gradient filter
// this step highlights (sharpen) the edges and boundaries of objects in the image to help text detection
void morphology(const Mat& in_img, Mat& out_img) {
	Mat morph_kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	morphologyEx(in_img, out_img, MORPH_GRADIENT, morph_kernel);
}

// perform binarization on the image using threshold function
// this step converts the image into a binary image so that possible text regions are represented as white pixels, black background
void binarization(const Mat& in_img, Mat& out_img) {
	threshold(in_img, out_img, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU);
}

// perform connection of horizontally oriented regions of the binary image
// this step groups together the possible text regions that are close to each other in a horizontal manner
void connect_regions(const Mat& in_img, Mat& out_img) {
	Mat morph_kernel = getStructuringElement(MORPH_RECT, Size(9, 1));
	morphologyEx(in_img, out_img, MORPH_CLOSE, morph_kernel);
}

// perform detection of the contours and drawing of the boundary boxes
// this step iterates through the detected contours and filters them based on certain criteria
void detect_draw_text(const Mat& in_img, Mat& imageRGB, double threshold, double constraint) {
	// find contours based on the binary connected image
	// contours are the boundaries of the possible text regions in the image

	// blank image is a binary image of the same size as the input image where text regions are filled with white pixels and non-text regions are black
	Mat blank = Mat::zeros(in_img.size(), CV_8UC1);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(in_img, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	// filter the detected contours based on defined criteria
	for (int index = 0; index >= 0; index = hierarchy[index][0])
	{
		Rect rect = boundingRect(contours[index]);
		// define the region of interest (RoI) within the blank image, clear it
		Mat RoI_blank(blank, rect);
		RoI_blank = Scalar(0, 0, 0);

		// draw the detected contour on the image to be filled
		drawContours(blank, contours, index, Scalar(255, 255, 255), CV_FILLED);

		// calculate the ratio of white pixels in the filled region of the contour
		double ratio = (double)countNonZero(RoI_blank) / (rect.width*rect.height);

		// check the ratio threshold and size constraints of the detected contour
		if (ratio > threshold && (rect.height > constraint && rect.width > constraint))
		{
			// draw a red boundary box on the determined region
			rectangle(imageRGB, rect, Scalar(0, 0, 255), 2);
		}
	}
}

int main(int argc, _TCHAR* argv[]) {

	// ENTER THE ADDRESS OF THE IMAGE TO BE CONSIDERED
	// get the image file (jpg or png)
	Mat imageL = imread("C:/Users/ercih/Desktop/example.png");
	Mat imageRGB;
	Mat imageS;

	// check for the errors
	if (imageL.empty()) {
		cerr << "Error: Could not open or find the image." << endl;
		return -1;
	}

	// perform preprocessing on the image, downsampling and gray-scaling
	// this step eases the text detection as it works with single-channel images
	pyrDown(imageL, imageRGB);
	cvtColor(imageRGB, imageS, CV_BGR2GRAY);

	// apply morphological gradient filter on the image
	Mat gradient_img;
	morphology(imageS, gradient_img);

	// apply binarization on the image
	Mat bin_img;
	binarization(gradient_img, bin_img);

	// apply horizontal connection on the image
	Mat connected_img;
	connect_regions(bin_img, connected_img);

	// apply contour detection and box drawings, define parameters
	// these values can be altered for fine-tuning purposes
	double threshold = 0.30;
	double constraint = 6;
	detect_draw_text(connected_img, imageRGB, threshold, constraint);

	// ENTER THE SAVE ADDRESS FOR THE FINAL IMAGE
	// save the resultant image
	imwrite("C:/Users/ercih/Desktop/" + string("out_img.jpg"), imageRGB);

	// name the window
	String windowName = "image";
	// display the result
	cv::imshow(windowName, imageRGB);
	cv::waitKey(0);
	destroyWindow(windowName);

	return 0;
}
