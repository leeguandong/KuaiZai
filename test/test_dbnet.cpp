#include <iostream>
#include "dbnet.h"


int main() {
	const char* image_path = "data/1.jpg";
	cv::Mat image = cv::imread(image_path, 1);

	float thresh = 0.2;
	float box_thresh = 0.4;
	float upclip_ratio = 1.7;
	//如果采用原始的dbnet预处理resize方式，targe_size代表短边固定尺寸建议设为640；
	//如果采用等比例缩放到最近32像素倍数，targe_size代表resize后最长边不超过的尺寸建议设为960
	int targe_size = 960;

	const char* model_param = "weights/ocr/dbnet_shufflenet_v2_x0_5_128_finetune_best.param";
	const char* model_bin = "weights/ocr/dbnet_shufflenet_v2_x0_5_128_finetune_best.bin";
	Text_Detect dbnet(model_param, model_bin, thresh, box_thresh, upclip_ratio);//构造模型实体
	std::vector<DetTextBox> textboxes = dbnet.detect(image, targe_size);
}


