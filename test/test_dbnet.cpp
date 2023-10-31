#include <iostream>
#include "dbnet.h"


int main() {
	const char* image_path = "data/1.jpg";
	cv::Mat image = cv::imread(image_path, 1);

	float thresh = 0.2;
	float box_thresh = 0.4;
	float upclip_ratio = 1.7;
	//�������ԭʼ��dbnetԤ����resize��ʽ��targe_size����̱߹̶��ߴ罨����Ϊ640��
	//������õȱ������ŵ����32���ر�����targe_size����resize����߲������ĳߴ罨����Ϊ960
	int targe_size = 960;

	const char* model_param = "weights/ocr/dbnet_shufflenet_v2_x0_5_128_finetune_best.param";
	const char* model_bin = "weights/ocr/dbnet_shufflenet_v2_x0_5_128_finetune_best.bin";
	Text_Detect dbnet(model_param, model_bin, thresh, box_thresh, upclip_ratio);//����ģ��ʵ��
	std::vector<DetTextBox> textboxes = dbnet.detect(image, targe_size);
}


