#pragma once
#ifndef VECHICLE
#define VECHICLE
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;
using namespace Ort;

typedef struct BoxInfo {
	int xmin;
	int ymin;
	int xmax;
	int ymax;
	float score;
	string name;
};

class PP_YOLOE {
public:
	PP_YOLOE(string model_path, float conf_threshold);
	vector<BoxInfo> detect(Mat cv_image);
private:
	float conf_threshold;
	const int num_class = 1;

	Mat preprocess(Mat img);
	void normalize_(Mat img);
	int inp_height;
	int inp_width;
	vector<float> input_image_;
	vector<float> scale_factor = { 1,1 };

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "pp-yoloe");
	Ort::Session* ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims;
	vector<vector<int64_t>> output_node_dims;
};




#endif // !VECHICLE






