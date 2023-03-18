#pragma once
#ifndef PLATE_DET
#define PLATE_DET

#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;
using namespace Ort;

class PlateDetector {
public:
	PlateDetector(string model_path);
	vector<vector<Point2f>> detect(Mat& img);
	void draw_pred(Mat& img, vector<vector<Point2f>> results);
private:
	float binary_threshold;
	float polygon_threshold;
	float unclipRatio;
	int maxCandidates;
	const int short_size = 736;
	const float meanValues[3] = { 0.485, 0.456, 0.406 };
	const float normValues[3] = { 0.229, 0.224, 0.225 };
	float contourScore(const Mat& binary, const vector<Point>& contour);
	void unclip(const vector<Point2f>& inPoly, vector<Point2f>& outPoly);
	//vector<vector<Point2f>>  order_points_clockwise(vector<vector<Point2f>> results);
	Mat preprocess(Mat img); // �������ã�ָ�봫ֵ�ģ�ע�����ķ���ֵ
	vector<float> input_image_;
	void normalize_(Mat img);

	Session* ort_session;
	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "DBNet");
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
};



#endif // !PLATE_DET
