#pragma once
#ifndef DBNET
#define DBNET
#include <vector>
#include "opencv2/opencv.hpp"
#include "net.h"


// ÿһ���ı���ļ��ṹ�壬������������Ŷ�
struct DetTextBox {
	int x1, y1, x2, y2, x3, y3, x4, y4;
	float score;
};

class Text_Detect {
public:
	Text_Detect(const char* model_param, // ����ṹ
		const char* model_bin, //����Ȩ��
		const float thresh = 0.2,
		const float box_thresh = 0.45,
		const float unclip_ratio = 1.7);
	~Text_Detect(); // ��������
	std::vector<DetTextBox> detect(const cv::Mat& image, const int& inshortsize);
	// ��ȡopencv����contour����С��Ӿ��ε��ĸ���minibox���þ�����̱߳���minedgesize���þ��ε��ܳ�perimeter���area
	void get_mini_boxes(std::vector<cv::Point>& contour, cv::Point2f minibox[4], float& minedgesize, float& perimeter, float& area);
	// ͨ������ͼ������contour,���ظ��ı������ڵ�����ֵ��ƽ�����Ŷ�
	float box_score_fast(cv::Mat& probability_map, std::vector<cv::Point>& contour); //��Ϊncnn�ƶϳ����������Ŷȸ������Ĳ��죬���õ���ÿ���ı�������Ŷ����в�ͬ
	// ���ź���������clipper�⣬ͨ����С��Ӿ��ε��ĵ㡢��Ӧ���ε��ܳ������ų�����������ź���ı����contour
	void unclip(cv::Point2f minibox[4], float& perimeter, float& area, float& unclip_ratio, std::vector<cv::Point>& expand_contour);

private:
	ncnn::Net Model;

	const float mean_vals_dbnet[3] = { 0.485 * 255, 0.456 * 255, 0.406 * 255 };
	const float norm_vals_dbnet[3] = { 1.0 / 0.229 / 255.0, 1.0 / 0.224 / 255.0, 1.0 / 0.225 / 255.0 };
	int num_thread = 4;
	std::string model_param_;
	std::string model_bin_;
	float unclip_ratio_ = 1.5;
	float box_thresh_ = 0.5;
	float thresh_ = 0.3;
	int min_size_ = 3;
};


#endif // 
