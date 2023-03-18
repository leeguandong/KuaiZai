#include "picodet.h"

PicoDet::PicoDet(string model_path, string classes_file, float nms_threshold, float score_threshold) {
	ifstream ifs(classes_file.c_str());
	string line;
	while (getline(ifs, line)) this->class_names.push_back(line);
	this->num_class = class_names.size();
	this->nms_threshold = nms_threshold;
	this->score_threshold = score_threshold;

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
	this->num_outs = int(numOutputNodes * 0.5);
	this->reg_max = output_node_dims[this->num_outs][output_node_dims[this->num_outs].size() - 1] / 4 - 1;
	for (int i = 0; i < this->num_outs; i++) {
		stride.push_back(int(8 * pow(2, i)));
	}
}

void PicoDet::generate_proposal(vector<BoxInfo>& generate_boxes,
	const int stride_, const float* out_score, const float* out_box) {
	const int num_grid_y = (int)ceil((float)this->inp_height / stride_); // 52
	const int num_grid_x = (int)ceil((float)this->inp_width / stride_); // 52
	const int reg_lmax = reg_max + 1; // 8

	// 解码
	for (int i = 0; i < num_grid_y; i++) {
		for (int j = 0; j < num_grid_x; j++) {
			const int idx = i * num_grid_x + j;
			const float* scores = out_score + idx * num_class;
			int max_ind = 0;
			float max_score = 0;
			for (int k = 0; k < num_class; k++) { // num_class:80
				if (scores[k] > max_score) {
					max_score = scores[k];
					max_ind = k;
				}
			}
			if (max_score >= score_threshold) {
				const float* pbox = out_box + idx * reg_lmax * 4;
				float dis_pred[4];
				float* y = new float[reg_lmax];
				for (int k = 0; k < 4; k++) {
					softmax_(pbox + k * reg_lmax, y, reg_lmax);
					float dis = 0.f;
					for (int l = 0; l < reg_lmax; l++) {
						dis += l * y[l];
					}
					dis_pred[k] = dis * stride_;
				}
				delete[] y;
				float pb_cx = (j + 0.5f) * stride_ - 0.5;
				float pb_cy = (i + 0.5f) * stride_ - 0.5;
				float x0 = pb_cx - dis_pred[0];
				float y0 = pb_cy - dis_pred[1];
				float x1 = pb_cx + dis_pred[2];
				float y1 = pb_cy + dis_pred[3];
				generate_boxes.push_back(BoxInfo{ x0,y0,x1,y1,max_score,max_ind });
			}
		}
	}
}

void PicoDet::nms(vector<BoxInfo>& input_boxes) {
	sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) {return a.score > b.score; });
	vector<float> vArea(input_boxes.size());
	for (int i = 0; i<int(input_boxes.size()); ++i) {
		vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
			* (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
	}

	vector<bool> isSuppressed(input_boxes.size(), false);
	for (int i = 0; i<int(input_boxes.size()); ++i) {
		if (isSuppressed[i]) { continue; }
		for (int j = i + 1; j<int(input_boxes.size()); ++j) {
			if (isSuppressed[j]) { continue; }
			float xx1 = (max)(input_boxes[i].x1, input_boxes[j].x1);
			float yy1 = (max)(input_boxes[i].y1, input_boxes[j].y1);
			float xx2 = (min)(input_boxes[i].x2, input_boxes[j].x2);
			float yy2 = (min)(input_boxes[i].y2, input_boxes[j].y2);

			float w = (max)(float(0), xx2 - xx1 + 1);
			float h = (max)(float(0), yy2 - yy1 + 1);
			float inter = w * h;
			float ovr = inter / (vArea[i] + vArea[j] - inter);

			if (ovr >= this->nms_threshold) {
				isSuppressed[j] = true;
			}
		}
	}
	int idx_t = 0;
	input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(),
		[&idx_t, &isSuppressed](const BoxInfo& f) {return isSuppressed[idx_t++]; }),
		input_boxes.end());
}


void PicoDet::softmax_(const float* x, float* y, int length) {
	float sum = 0;
	int i = 0;
	for (i = 0; i < length; i++) {
		y[i] = exp(x[i]);
		sum += y[i];
	}
	for (i = 0; i < length; i++) {
		y[i] /= sum;
	}
}


void PicoDet::detect(Mat& img) {
	int height = 0, width = 0, top = 0, left = 0;
	Mat cv_image = img.clone();
	Mat dst = this->resize_image(cv_image, &height, &width, &top, &left);
	this->normalize_(dst);
	array<int64_t, 4> input_shape_{ 1,3,this->inp_height,this->inp_width }; // array的初始化

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info,
		input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr },
		&input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());
	// generate proposals
	vector<BoxInfo> generate_boxes;
	for (int i = 0; i < this->num_outs; i++) {
		const float* cls_score = ort_outputs[i].GetTensorMutableData<float>();
		const float* bbox_pred = ort_outputs[i + this->num_outs].GetTensorMutableData<float>();
		generate_proposal(generate_boxes, stride[i], cls_score, bbox_pred);
	}

	// nms去除多余低置信度重叠框
	nms(generate_boxes);
	float ratioh = (float)cv_image.rows / height;
	float ratiow = (float)cv_image.cols / width;
	for (size_t i = 0; i < generate_boxes.size(); ++i) {
		int xmin = (int)max((generate_boxes[i].x1 - left) * ratiow, 0.f);
		int ymin = (int)max((generate_boxes[i].y1 - top) * ratioh, 0.f);
		int xmax = (int)min((generate_boxes[i].x2 - left) * ratiow, (float)cv_image.cols);
		int ymax = (int)min((generate_boxes[i].y2 - top) * ratioh, (float)cv_image.rows);
		rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 0, 255), 2);
		string label = format("%.2f", generate_boxes[i].score);
		label = this->class_names[generate_boxes[i].label] + ":" + label;
		putText(img, label, Point(xmin, ymin - 5), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
	}
}


void PicoDet::normalize_(Mat img) {
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());
	for (int c = 0; c < 3; c++) {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				float pix = img.ptr<uchar>(i)[j * 3 + c];
				this->input_image_[c * row * col + i * col + j] =
					(pix / 255.0 - mean[c] / 255.0) / (stds[c] / 255.0);
				//this->input_image_[c * row * col + i * col + j] = (pix - mean[c]) / stds[c];
			}
		}
	}
}

Mat PicoDet::resize_image(Mat img, int* height, int* width, int* top, int* left) {
	int img_h = img.rows, img_w = img.cols;
	*height = this->inp_height;
	*width = this->inp_width;

	Mat dst;
	if (this->keep_rate && img_h != img_w) { // 保持比例和图片宽和高不一致
		float hw_scale = (float)img_h / img_w; // 高宽比
		if (hw_scale > 1) {
			*height = this->inp_height;
			*width = int(this->inp_width / hw_scale);
			resize(img, dst, Size(*width, *height), INTER_AREA);
			*left = int((this->inp_width - *width) * 0.5);
			copyMakeBorder(dst, dst, 0, 0, *left, this->inp_width - *width - *left, BORDER_CONSTANT, 0);
		}
		else {
			*height = (int)this->inp_height * hw_scale;
			*width = this->inp_width;
			resize(img, dst, Size(*width, *height), INTER_AREA);
			*top = (int)(this->inp_height - *height) * 0.5;
			copyMakeBorder(dst, dst, *top, this->inp_height - *height - *top, 0, 0, BORDER_CONSTANT, 0);
		}
	}
	else {
		resize(img, dst, Size(*width, *height), INTER_AREA);
	}
	return dst;
}























