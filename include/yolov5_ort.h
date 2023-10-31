#pragma once
#ifndef YOLOV5_ORT
#define YOLOV5_ORT

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <utility>
#include <fstream>
#include <codecvt>


class Detection {
public:
	cv::Rect box;
	float conf{}; //定义了一个名为 conf 的变量，类型为 float，并将其初始化为 0
	int classId{};
};


class Yolov5ORT {
public:
	Yolov5ORT(const std::string modelPath, const bool isGPU, const cv::Size inputSize);

	std::vector<Detection> detect(cv::Mat& image, const float confThreshold, const float iouThreshold);

	std::vector<std::string> loadNames(const std::string path);

	std::wstring charToWstring(const char* str);

	void visualizeDetection(cv::Mat& image, std::vector<Detection>& detections, const std::vector<std::string> classNames);

private:
	Ort::Env env{ nullptr }; // Ort::Env 类表示 ONNX Runtime 的运行环境，包括线程池、内存分配器等
	Ort::SessionOptions sessionOptions{ nullptr }; //Ort::SessionOptions 类是一个包含用于配置如何运行 ONNX model 的选项的类
	Ort::Session session{ nullptr }; //Ort::Session 类表示一个运行 ONNX Model 的会话对象

	void letterbox(const cv::Mat& image, cv::Mat& outImage, const cv::Size newShape, const cv::Scalar color, bool auto_, bool scaleUp, int stride);
	void preprocessing(cv::Mat& image, float*& blob, std::vector<int64_t>& inputTensorShape);
	std::vector<Detection> postprocessing(const cv::Size resizedImageShape,
		const cv::Size originalImageShape, std::vector<Ort::Value>& outputTensor, const float confThreshold, const float iouThreshold);
	void scaleCoords(const cv::Size imageShape, cv::Rect& coords, const cv::Size imageoriginalShape);

	static void getBestClassInfo(std::vector<float>::iterator it, const int& numClasses, float& bestConf, int& bestClassId);
	size_t vectorProduct(const std::vector<int64_t> vector);

	std::vector<const char*> inputNames;
	std::vector<const char*> outputNames;
	bool isDynamicInputShape{};
	cv::Size2f inputImageShape;
};



#endif // !YOLOV5_ORT
