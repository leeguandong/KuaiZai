#include "vehicle.h"

PP_YOLOE::PP_YOLOE(string model_path, float conf_threshold) {
	wstring widestr = wstring(model_path.begin(), model_path.end());
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

	ort_session = new Session(env, widestr.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++) {
		input_names.push_back(ort_session->GetInputName(i, allocator));
		//Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		//auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		//auto input_dims = input_tensor_info.GetShape();
		auto input_dims = ort_session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++) {
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		//Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		//auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		//auto output_dims = output_tensor_info.GetShape();
		auto output_dims = ort_session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		output_node_dims.push_back(output_dims);
	}
	this->inp_height = input_node_dims[0][2];
	this->inp_width = input_node_dims[0][3];
	this->conf_threshold = conf_threshold;
}

Mat PP_YOLOE::preprocess(Mat img) {
	Mat dst;
	cvtColor(img, dst, COLOR_BGR2RGB);
	resize(dst, dst, Size(this->inp_width, this->inp_height), INTER_LINEAR);
	return dst;
}

void PP_YOLOE::normalize_(Mat img) {
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());
	for (int c = 0; c < 3; c++) {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				float pix = img.ptr<uchar>(i)[j * 3 + c];
				this->input_image_[c * row * col + i * col + j] = pix;
			}
		}
	}
}


vector<BoxInfo> PP_YOLOE::detect(Mat img) {
	Mat dst = this->preprocess(img);
	this->normalize_(dst);
	array<int64_t, 4> input_shape_{ 1,3,this->inp_height,this->inp_width };
	array<int64_t, 2> scale_shape_{ 1,2 };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	vector<Value> ort_inputs;
	ort_inputs.push_back(Value::CreateTensor<float>(allocator_info,
		input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size()));
	ort_inputs.push_back(Value::CreateTensor<float>(allocator_info,
		scale_factor.data(), scale_factor.size(), scale_shape_.data(), scale_shape_.size()));

	// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr },
		input_names.data(), ort_inputs.data(), 2, output_names.data(), output_names.size());
	const float* outs = ort_outputs[0].GetTensorMutableData<float>();
	const int* box_num = ort_outputs[1].GetTensorMutableData<int>();

	const float ratioh = float(img.rows) / this->inp_height;
	const float ratiow = float(img.cols) / this->inp_width;
	vector<BoxInfo> boxes;
	for (int i = 0; i < box_num[0]; i++) {
		if (outs[0] > -1 && outs[1] > this->conf_threshold) {
			boxes.push_back({ int(outs[2] * ratiow),int(outs[3] * ratioh),
				int(outs[4] * ratiow),int(outs[5] * ratioh),outs[1],"vehicle" });
		}
		outs += 6;
	}
	return boxes;
}




