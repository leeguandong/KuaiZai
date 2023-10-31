//#pragma once
//#ifndef YOLOV5_ORT_FLYCV
//#define YOLOV5_ORT_FLYCV
//
////#include <opencv2/opencv.hpp>
//#include <assert.h>
//#include <flycv.h>
//#include <onnxruntime_cxx_api.h>
//#include <utility>
//#include <fstream>
//#include <codecvt>
//#include <iostream>
//
//
//
//class Detection {
//public:
//	fcv::Rect box;
//	float conf{}; //������һ����Ϊ conf �ı���������Ϊ float���������ʼ��Ϊ 0
//	int classId{};
//};
//
//
//class Yolov5ORT {
//public:
//	Yolov5ORT(const std::string modelPath, const bool isGPU, const fcv::Size inputSize);
//
//	std::vector<Detection> detect(fcv::Mat& image, const float confThreshold, const float iouThreshold);
//
//	std::vector<std::string> loadNames(const std::string path);
//
//	std::wstring charToWstring(const char* str);
//
//	void visualizeDetection(fcv::Mat& image, std::vector<Detection>& detections, const std::vector<std::string> classNames);
//
//private:
//	Ort::Env env{ nullptr }; // Ort::Env ���ʾ ONNX Runtime �����л����������̳߳ء��ڴ��������
//	Ort::SessionOptions sessionOptions{ nullptr }; //Ort::SessionOptions ����һ��������������������� ONNX model ��ѡ�����
//	Ort::Session session{ nullptr }; //Ort::Session ���ʾһ������ ONNX Model �ĻỰ����
//
//	void letterbox(const fcv::Mat& image, fcv::Mat& outImage, const fcv::Size newShape, const fcv::Scalar color, bool auto_, bool scaleUp, int stride);
//	void preprocessing(fcv::Mat& image, float*& blob, std::vector<int64_t>& inputTensorShape);
//	std::vector<Detection> postprocessing(const fcv::Size resizedImageShape,
//		const fcv::Size originalImageShape, std::vector<Ort::Value>& outputTensor, const float confThreshold, const float iouThreshold);
//	void scaleCoords(const fcv::Size imageShape, fcv::Rect& coords, const fcv::Size imageoriginalShape);
//
//	static void getBestClassInfo(std::vector<float>::iterator it, const int& numClasses, float& bestConf, int& bestClassId);
//	size_t vectorProduct(const std::vector<int64_t> vector);
//
//	std::vector<const char*> inputNames;
//	std::vector<const char*> outputNames;
//	bool isDynamicInputShape{};
//	//fcv::Size2f inputImageShape;
//	fcv::Size inputImageShape;
//};
//
//
//
//#endif // !YOLOV5_ORT_FLYCV
