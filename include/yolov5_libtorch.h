#pragma once
#ifndef YOLOV5_LIBTORCH
#define YOLOV5_LIBTORCH
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <ctime>

/*
ImageResizeData ͼƬ������󱣴�ͼƬ�����ݽṹ
*/
class ImageResizeData {
public:
	bool isW(); //��ԭʼͼƬ��߱ȴ��ڴ������ͼƬ��߱�ʱ����true
	bool isH(); // ��ԭʼͼƬ�߿�ȴ��ڴ������ͼƬ�߿��ʱ����true
	int height; // �������ͼƬ��
	int width; // �������ͼƬ��
	int w; // ԭʼͼƬ�Ŀ�
	int h; // ԭʼͼƬ�ĸ�
	int border; // ��ԭʼͼƬ������ͼƬ����ӵĺڱߴ�С
	cv::Mat img; // ��������ͼƬ
};


class Yolov5Libtorch {
public:
	/**
	  * ���캯��
	  * @param ptFile yoloV5 pt�ļ�·��
	  * @param isCuda �Ƿ�ʹ�� cuda Ĭ�ϲ�����
	  * @param height yoloV5 ѵ��ʱͼƬ�ĸ�
	  * @param width yoloV5 ѵ��ʱͼƬ�Ŀ�
	  * @param confThres �Ǽ���ֵ�����е� scoreThresh
	  * @param iouThres �Ǽ���ֵ�����е� iouThresh
	  */
	Yolov5Libtorch(std::string ptFile, bool isCuda = false, bool isHalf = false, int height = 640,
		int width = 640, float confThres = 0.25, float iouThres = 0.45);


	/**
	* Ԥ�⺯��
	* @param data Ԥ������ݸ�ʽ (batch, rgb, height, width)
	*/
	std::vector<torch::Tensor> prediction(torch::Tensor data);
	std::vector<torch::Tensor> prediction(cv::Mat img);

	/**
	 * �ı�ͼƬ��С�ĺ���
	 * @param img ԭʼͼƬ
	 * @param height Ҫ����ɵ�ͼƬ�ĸ�
	 * @param width Ҫ����ɵ�ͼƬ�Ŀ�
	 * @return ��װ�õĴ������ͼƬ���ݽṹ
	 */
	static ImageResizeData resize(cv::Mat img, int height, int width);
	ImageResizeData resize(cv::Mat img);

	/**
	 * �����������ڸ���ͼƬ�л�����
	 * @param imgs ԭʼͼƬ����
	 * @param rectangles ͨ��Ԥ�⺯������õĽ��
	 * @param labels ����ǩ
	 * @param thickness �߿�
	 * @return ���ÿ��ͼƬ
	 */
	cv::Mat drawRectangle(cv::Mat img, torch::Tensor rectangle, std::map<int, std::string> labels, int thickness = 2);
	cv::Mat drawRectangle(cv::Mat img, torch::Tensor rectangle, std::map<int, cv::Scalar> colors, std::map<int, std::string> labels, int thickness = 2);

	void showShape(torch::Tensor);

private:
	bool isCuda; // �Ƿ�����cuda
	bool isHalf; // �Ƿ�ʹ�ð뾫��
	float confThres; // nms�еķ�����ֵ
	float iouThres; // nms��iou��ֵ
	float height;  // ģ������Ҫ��ͼƬ�ĸ�
	float width;  // ģ������Ҫ��ͼƬ�Ŀ�
	std::map<int, cv::Scalar> mainColors; //������ɫ map
	torch::jit::script::Module model; //ģ��

	cv::Mat img2RGB(cv::Mat img); // ͼƬͨ��ת��Ϊrgb
	torch::Tensor img2Tensor(cv::Mat img); // ͼƬ��Ϊtensor
	torch::Tensor xywh2xyxy(torch::Tensor); // (center_x center_y w h) to (left,top,right,bottom)
	torch::Tensor nms(torch::Tensor bboxes, torch::Tensor scores, float thresh);
	std::vector<torch::Tensor> sizeOriginal(std::vector<torch::Tensor> result, std::vector<ImageResizeData> imgRDs); // Ԥ������Ŀ����ԭʼͼƬ��ԭ�㷨
	std::vector<torch::Tensor> non_max_suppression(torch::Tensor preds, float confThres = 0.25, float iouThres = 0.45); // nms
};


#endif YOLOV5_LIBTORCH

