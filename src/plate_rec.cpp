#include "plate_rec.h"

TextRecognizer::TextRecognizer(string model_path, string classes_file) {
	/*simcode["Õã"] = 'ZJ-';
	simcode["ÔÁ"] = 'GD-';
	simcode["¾©"] = 'BJ-';
	simcode["½ò"] = 'TJ-';
	simcode["¼½"] = 'HE-';
	simcode["½ú"] = 'SX-';
	simcode["ÃÉ"] = 'NM-';
	simcode["ÁÉ"] = 'LN-';
	simcode["ºÚ"] = 'HLJ-';
	simcode["»¦"] = 'SH-';
	simcode["¼ª"] = 'JL-';
	simcode["ËÕ"] = 'JS-';
	simcode["Íî"] = 'AH-';
	simcode["¸Ó"] = 'JX-';
	simcode["Â³"] = 'SD-';
	simcode["Ô¥"] = 'HA-';
	simcode["¶õ"] = 'HB-';
	simcode["Ïæ"] = 'HN-';
	simcode["¹ð"] = 'GX-';
	simcode["Çí"] = 'HI-';
	simcode["Óå"] = 'CQ-';
	simcode["´¨"] = 'SC-';
	simcode["¹ó"] = 'GZ-';
	simcode["ÔÆ"] = 'YN-';
	simcode["²Ø"] = 'XZ-';
	simcode["ÉÂ"] = 'SN-';
	simcode["¸Ê"] = 'GS-';
	simcode["Çà"] = 'QH-';
	simcode["Äþ"] = 'NX-';
	simcode["Ãö"] = 'FJ-';
	simcode["¡¤"] = ' ';*/
	//simcode.insert(pair<string, string>('Õã', 'ZJ-'));

	/*simcode = {
		{'Õã', 'ZJ-'},
		{ 'ÔÁ' , 'GD-' },
		{ '¾©' , 'BJ-' },
		{ '½ò' , 'TJ-' },
		{ '¼½' , 'HE-' },
		{ '½ú' , 'SX-' },
		{ 'ÃÉ' , 'NM-' },
		{ 'ÁÉ' , 'LN-' },
		{ 'ºÚ' , 'HLJ-' },
		{ '»¦' , 'SH-' },
		{ '¼ª' , 'JL-' },
		{ 'ËÕ' , 'JS-' },
		{ 'Íî' , 'AH-' },
		{ '¸Ó' , 'JX-' },
		{ 'Â³' , 'SD-' },
		{ 'Ô¥' , 'HA-' },
		{ '¶õ', 'HB-' },
		{ 'Ïæ' , 'HN-' },
		{ '¹ð' , 'GX-' },
		{ 'Çí' , 'HI-' },
		{ 'Óå', 'CQ-' },
		{ '´¨' , 'SC-' },
		{ '¹ó' , 'GZ-' },
		{ 'ÔÆ' , 'YN-' },
		{ '²Ø' , 'XZ-' },
		{ 'ÉÂ' , 'SN-' },
		{ '¸Ê' , 'GS-' },
		{ 'Çà' , 'QH-' },
		{ 'Äþ' , 'NX-' },
		{ 'Ãö' , 'FJ-' },
		{ '¡¤' , ' ' },
	};*/

	wstring widestr = wstring(model_path.begin(), model_path.end());
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, widestr.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++) {
		input_names.push_back(ort_session->GetInputName(i, allocator));
		auto input_dims = ort_session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++) {
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		auto output_dims = ort_session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		output_node_dims.push_back(output_dims);
	}

	max_wh_ratio = (float)this->inp_width / (float)this->inp_height;
	imgW = int(this->inp_height * max_wh_ratio); //±£Ö¤¿í¸ß±È£¬¸ßÊÇ48,

	ifstream ifs(classes_file.c_str());
	string line;
	while (getline(ifs, line)) {
		this->alphabet.push_back(line);
	}
	names_len = this->alphabet.size();
}


Mat TextRecognizer::get_rotate_crop_image_v1(const Mat& frame, Point2f vertices[4]) {
	int left = 10000;
	int right = 0;
	int top = 10000;
	int bottom = 0;
	for (int i = 0; i < 4; i++) {
		if (vertices[i].x < left) {
			left = int(vertices[i].x);
		}
		if (vertices[i].y < top) {
			top = int(vertices[i].y);
		}
		if (vertices[i].x > right) {
			right = int(vertices[i].x);
		}
		if (vertices[i].y > bottom) {
			bottom = int(vertices[i].y);
		}
	}

	Rect rect;
	rect.x = left;
	rect.y = top;
	rect.width = right - left;
	rect.height = bottom - top;
	if (rect.width == 0)
		rect.width = right - left + 1;
	if (rect.height == 0)
		rect.height = bottom - top + 1;
	Mat crop_plate = frame(rect);

	const Size outputSize = Size(rect.width, rect.height);
	Point2f targetVertices[4] = {
		Point(0,outputSize.height - 1),
		Point(0,0),Point(outputSize.width - 1,0),
		Point(outputSize.width - 1,outputSize.height - 1),
	};
	for (int i = 0; i < 4; i++) {
		vertices[i].x -= left;
		vertices[i].y -= top;
	}
	Mat rotationMatrix = getPerspectiveTransform(vertices, targetVertices);
	Mat result;
	warpPerspective(crop_plate, result, rotationMatrix, outputSize);
	return result;
}


Mat TextRecognizer::preprocess(Mat img) {
	Mat dst;
	cvtColor(img, dst, COLOR_BGR2RGB);
	int h = img.rows;
	int w = img.cols;
	const float ratio = w / float(h);
	int resized_w = int(ceil((float)this->inp_height * ratio));
	if (ceil(this->inp_height * ratio) > imgW) {
		resized_w = imgW;
	}
	resize(dst, dst, Size(resized_w, this->inp_height), INTER_LINEAR);
	return dst;
}


void TextRecognizer::normalize_(Mat img) {
	//    img.convertTo(img, CV_32F);
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(this->inp_height * imgW * img.channels());

	for (int c = 0; c < 3; c++) {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < imgW; j++) {
				if (j < col) {
					float pix = img.ptr<uchar>(i)[j * 3 + c];
					this->input_image_[c * row * imgW + i * imgW + j] = (pix / 255.0 - 0.5) / 0.5;
				}
				else {
					this->input_image_[c * row * imgW + i * imgW + j] = 0;
				}
			}
		}
	}
}

string TextRecognizer::detect(Mat img) {
	Mat dst = this->preprocess(img);
	this->normalize_(dst);
	array<int64_t, 4> input_shape_{ 1,3,this->inp_height,this->imgW };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info,
		input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr },
		&input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());
	const float* pdata = ort_outputs[0].GetTensorMutableData<float>();

	//int i = 0, j = 0;
	int h = ort_outputs.at(0).GetTensorTypeAndShapeInfo().GetShape().at(2);
	int w = ort_outputs.at(0).GetTensorTypeAndShapeInfo().GetShape().at(1);
	int* preb_label = new int[w];//¿ª±ÙÄÚ´æÖÐµÄwÊÇ±äÁ¿
	for (int i = 0; i < w; i++) {
		int one_label_idx = 0;
		float max_data = -10000;
		for (int j = 0; j < h; j++) {
			float data_ = pdata[i * h + j];
			if (data_ > max_data) {
				max_data = data_;
				one_label_idx = j;
			}
		}
		preb_label[i] = one_label_idx;
	}

	vector<int> no_repeat_blank_label;
	for (size_t index = 1; index < w; ++index) {
		if (preb_label[index] != 0 && preb_label[index - 1] != preb_label[index]) {
			no_repeat_blank_label.push_back(preb_label[index] - 1);
		}
	}

	delete[] preb_label;
	int len_s = no_repeat_blank_label.size();
	string plate_text;
	for (int i = 0; i < len_s; i++) {
		plate_text += alphabet[no_repeat_blank_label[i]];
	}

	return plate_text;
}

