#pragma once
#ifndef DBNET
#define DBNET
#include <vector>
#include "opencv2/opencv.hpp"
#include "net.h"


// 每一个文本框的检测结构体，包括坐标和置信度
struct DetTextBox {
	int x1, y1, x2, y2, x3, y3, x4, y4;
	float score;
};

class Text_Detect {
public:
	Text_Detect(const char* model_param, // 网络结构
		const char* model_bin, //网络权重
		const float thresh = 0.2,
		const float box_thresh = 0.45,
		const float unclip_ratio = 1.7);
	~Text_Detect(); // 析构函数
	std::vector<DetTextBox> detect(const cv::Mat& image, const int& inshortsize);
	// 获取opencv轮廓contour的最小外接矩形的四个点minibox、该矩形最短边长度minedgesize、该矩形的周长perimeter面积area
	void get_mini_boxes(std::vector<cv::Point>& contour, cv::Point2f minibox[4], float& minedgesize, float& perimeter, float& area);
	// 通过概率图和轮廓contour,返回该文本区域内的像素值的平均置信度
	float box_score_fast(cv::Mat& probability_map, std::vector<cv::Point>& contour); //因为ncnn推断出的像素置信度浮点数的差异，最后得到的每个文本框的置信度略有不同
	// 扩张函数，调用clipper库，通过最小外接矩形的四点、相应矩形的周长和扩张常数，获得扩张后的文本框的contour
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
