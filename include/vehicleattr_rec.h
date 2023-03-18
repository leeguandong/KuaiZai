#pragma once
#ifndef VEHICLEATTR_REC
#define VEHICLEATTR_REC
#include <iostream>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;
using namespace Ort;

class VehicleAttr {
public:
	VehicleAttr(string model_path);
	void detect(Mat cv_image, string& color_res_str, string& type_res_str);
private:
	const float color_threshold = 0.5;
	const float type_threshold = 0.5;
	const char* color_list[10] = { "yellow", "orange", "green", "gray", "red", "blue", "white", "golden", "brown", "black" };
	const char* type_list[9] = { "sedan", "suv", "van", "hatchback", "mpv", "pickup", "bus", "truck", "estate" };
	const float mean[3] = { 0.485, 0.456, 0.406 };
	const float std[3] = { 0.229, 0.224, 0.225 };

	Mat preprocess(Mat img);
	void normalize_(Mat img);
	int inp_width;
	int inp_height;
	int num_out;
	vector<float> input_image_;

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "VehicleAttr"); // 
	Ort::Session* ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names; // 定义输入输出节点名
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims;
	vector<vector<int64_t>> output_node_dims;
};

#endif // ! VEHICLEATTR_REC




