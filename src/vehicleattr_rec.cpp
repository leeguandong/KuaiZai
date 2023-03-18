#include "vehicleattr_rec.h"

VehicleAttr::VehicleAttr(string model_path) {
	wstring widestr = wstring(model_path.begin(), model_path.end());

	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC); // ���û������Ż�
	ort_session = new Session(env, widestr.c_str(), sessionOptions); // ����onnxruntime��api
	size_t numInputNodes = ort_session->GetInputCount();  // ���ģ������������һ��ָ��Ӧ��������Ŀ��һ������ֻ��ͼ��Ļ���input_nodeΪ1
	size_t numOutputNodes = ort_session->GetOutputCount(); // ����Ƕ�������磬��Ӧ���������Ŀ
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++) {
		input_names.push_back(ort_session->GetInputName(i, allocator)); // ����ڵ�����

		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i); // ��ȡ����ά������
		//auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		//auto input_dims = input_tensor_info.GetShape();
		auto input_dims = input_type_info.GetTensorTypeAndShapeInfo().GetShape();

		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++) {
		output_names.push_back(ort_session->GetOutputName(i, allocator));

		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i); // ��ȡ���ά������
		//auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		//auto output_dims = output_tensor_info.GetShape();
		auto output_dims = output_type_info.GetTensorTypeAndShapeInfo().GetShape();

		output_node_dims.push_back(output_dims);
	}
	this->inp_height = input_node_dims[0][2];
	this->inp_width = input_node_dims[0][3];
	num_out = output_node_dims[0][1];
}

Mat VehicleAttr::preprocess(Mat img) {
	Mat dst;
	cvtColor(img, dst, COLOR_BGR2RGB);
	resize(dst, dst, Size(this->inp_width, this->inp_height), INTER_LINEAR);
	return dst;
}

void VehicleAttr::normalize_(Mat img)
{
	//img.convertTo(img, CV_32F);
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = img.ptr<uchar>(i)[j * 3 + c];
				this->input_image_[c * row * col + i * col + j] = (pix / 255.0 - this->mean[c]) / this->std[c];
			}
		}
	}
}


void VehicleAttr::detect(Mat cv_image, string& color_res_str, string& type_res_str) {
	Mat dst = this->preprocess(cv_image);
	this->normalize_(dst);
	// �������ڲ�ͬ��������ж������ڵ�Ͷ������ڵ㣬��Ҫ��std::vector������ort��tensor
	array<int64_t, 4> input_shape_{ 1,3,this->inp_height,this->inp_width };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU); // ��������tensor
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info,
		input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	// ����
	// ����ڵ�����input_tensor������ڵ�ڵ�����������ڵ㣬����ڵ�����
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr },
		&input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());

	const float* pdata = ort_outputs[0].GetTensorMutableData<float>(); // ȡ������ڵ�Ľ��,10�ֳ��ͺ�9����ɫ
	int color_idx;
	float max_prob = -1;
	for (int i = 0; i < 10; i++) {
		if (pdata[i] > max_prob) {
			max_prob = pdata[i];
			color_idx = i;
		}
	}
	int type_idx;
	max_prob = -1;
	for (int i = 10; i < num_out; i++) {
		if (pdata[i] > max_prob) {
			max_prob = pdata[i];
			type_idx = i - 10;
		}
	}

	if (pdata[color_idx] >= this->color_threshold) {
		color_res_str += this->color_list[color_idx];
	}
	else {
		color_res_str += "Unknown";
	}

	if (pdata[type_idx + 10] > this->type_threshold) {
		type_res_str += this->type_list[type_idx];
	}
	else {
		type_res_str += "Unknown";
	}
}






