#pragma once
#ifndef PICODET
#define PICODET
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;
using namespace Ort;

typedef struct BoxInfo {
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	int label;
}BoxInfo;

class PicoDet {
public:
	PicoDet(string model_path, string classes_file, float nms_threshold, float score_threshold);
	void detect(Mat& cv_image);
private:
	float score_threshold = 0.5;
	float nms_threshold = 0.5;
	vector<string> class_names;
	int num_class;

	Mat resize_image(Mat img, int* height, int* width, int* top, int* left);
	vector<float> input_image_;
	void normalize_(Mat img);
	void softmax_(const float* x, float* y, int length);
	void generate_proposal(vector<BoxInfo>& generate_boxes,
		const int stride_, const float* out_score, const float* out_box);
	void nms(vector<BoxInfo>& input_boxes);

	const bool keep_rate = true; // 保持图片的比例
	int inp_width;
	int inp_height;
	int num_outs;
	int reg_max;
	vector<int> stride;
	const float mean[3] = { 103.53,116.28,123.675 };
	const float stds[3] = { 57.375,57.12,58.395 };

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "picodet");
	Ort::Session* ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims;
	vector<vector<int64_t>> output_node_dims;
};

#endif // !PICODET
