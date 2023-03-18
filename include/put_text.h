#pragma once
#ifndef PUT_TEXT
#define PUT_TEXT

#include <windows.h>
#include <string>
#include <opencv2/opencv.hpp>

//using namespace cv;

void GetStringSize(HDC hDC, const char* str, int* w, int* h);
void putTextHusky(cv::Mat& dst, const char* str, cv::Point org, cv::Scalar color, int fontSize,
	const char* fn = "Arial", bool italic = false, bool underline = false);

#endif // !PUT_TEXT
