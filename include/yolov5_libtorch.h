#pragma once
#ifndef YOLOV5_LIBTORCH
#define YOLOV5_LIBTORCH
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <ctime>

/*
ImageResizeData 图片处理过后保存图片的数据结构
*/
class ImageResizeData {
public:
	bool isW(); //当原始图片宽高比大于处理过后图片宽高比时返回true
	bool isH(); // 当原始图片高宽比大于处理过后图片高宽比时返回true
	int height; // 处理过后图片高
	int width; // 处理过后图片宽
	int w; // 原始图片的宽
	int h; // 原始图片的高
	int border; // 从原始图片到处理图片所添加的黑边大小
	cv::Mat img; // 处理过后的图片
};


class Yolov5Libtorch {
public:
	/**
	  * 构造函数
	  * @param ptFile yoloV5 pt文件路径
	  * @param isCuda 是否使用 cuda 默认不起用
	  * @param height yoloV5 训练时图片的高
	  * @param width yoloV5 训练时图片的宽
	  * @param confThres 非极大值抑制中的 scoreThresh
	  * @param iouThres 非极大值抑制中的 iouThresh
	  */
	Yolov5Libtorch(std::string ptFile, bool isCuda = false, bool isHalf = false, int height = 640,
		int width = 640, float confThres = 0.25, float iouThres = 0.45);


	/**
	* 预测函数
	* @param data 预测的数据格式 (batch, rgb, height, width)
	*/
	std::vector<torch::Tensor> prediction(torch::Tensor data);
	std::vector<torch::Tensor> prediction(cv::Mat img);

	/**
	 * 改变图片大小的函数
	 * @param img 原始图片
	 * @param height 要处理成的图片的高
	 * @param width 要处理成的图片的宽
	 * @return 封装好的处理过后图片数据结构
	 */
	static ImageResizeData resize(cv::Mat img, int height, int width);
	ImageResizeData resize(cv::Mat img);

	/**
	 * 根据输出结果在给定图片中画出框
	 * @param imgs 原始图片集合
	 * @param rectangles 通过预测函数处理好的结果
	 * @param labels 类别标签
	 * @param thickness 线宽
	 * @return 画好框的图片
	 */
	cv::Mat drawRectangle(cv::Mat img, torch::Tensor rectangle, std::map<int, std::string> labels, int thickness = 2);
	cv::Mat drawRectangle(cv::Mat img, torch::Tensor rectangle, std::map<int, cv::Scalar> colors, std::map<int, std::string> labels, int thickness = 2);

	void showShape(torch::Tensor);

private:
	bool isCuda; // 是否启动cuda
	bool isHalf; // 是否使用半精度
	float confThres; // nms中的分数阈值
	float iouThres; // nms中iou阈值
	float height;  // 模型所需要的图片的高
	float width;  // 模型所需要的图片的宽
	std::map<int, cv::Scalar> mainColors; //画框颜色 map
	torch::jit::script::Module model; //模型

	cv::Mat img2RGB(cv::Mat img); // 图片通道转换为rgb
	torch::Tensor img2Tensor(cv::Mat img); // 图片变为tensor
	torch::Tensor xywh2xyxy(torch::Tensor); // (center_x center_y w h) to (left,top,right,bottom)
	torch::Tensor nms(torch::Tensor bboxes, torch::Tensor scores, float thresh);
	std::vector<torch::Tensor> sizeOriginal(std::vector<torch::Tensor> result, std::vector<ImageResizeData> imgRDs); // 预测出来的框根据原始图片还原算法
	std::vector<torch::Tensor> non_max_suppression(torch::Tensor preds, float confThres = 0.25, float iouThres = 0.45); // nms
};


#endif YOLOV5_LIBTORCH

