//#include <iostream>
//#include <opencv2/opencv.hpp>
//#include <time.h>
//
//#include "cmdline.h"
//#include "yolov5_ort.h"
//
//
////int main(int argc, char* argv[]) {
//int main() {
//
//	// ./KuaiZai.exe --model_path "weights/yolov5_person/weights/best.onnx" --video "data/2��8��(1).mp4" --class_naes "weights/yolov5_person/person.txt" ��������д������debug
//	//cmdline::parser cmd;
//	//cmd.add<std::string>("model_path", 'm', "path to onnx model", false, "weights/yolov5_person/weights/best.onnx");
//	//cmd.add<std::string>("image", 'i', "Image source to be detected", true, "");
//	//cmd.add<std::string>("video", 'v', "Video source to be detected", false, "data/2��8��(1).mp4");
//	//cmd.add<std::string>("class_names", 'c', "path to class names file", false, "weights/yolov5_person/person.txt");
//	//cmd.add("gpu", '\0', "Inference on cuda device");
//	//cmd.parse_check(argc, argv);
//
//	//bool isGPU = cmd.exist("gpu");
//	//const std::string imagePath = cmd.get<std::string>("video");
//	//const std::string modelPath = cmd.get<std::string>("model_path");
//	//const std::string classNamesPath = cmd.get<std::string>("class_names");
//
//	bool isGPU = false;
//	const std::string imagePath = "data/2��8��(1).mp4";
//	const std::string modelPath = "weights/yolov5_person/weights/best.onnx";
//	const std::string classNamesPath = "weights/yolov5_person/person.txt";
//	const std::string output_dir = "./results/2��8��(1)_ort/";
//
//	Yolov5ORT yolo(modelPath, isGPU, cv::Size(640, 640)); // ģ�ͳ�ʼ��
//	const float confThreshold = 0.3f;
//	const float iouThreshold = 0.4f;
//
//	const std::vector<std::string> classNames = yolo.loadNames(classNamesPath);
//	if (classNames.empty()) {
//		std::cerr << "Error: Empty class names file." << std::endl;
//		return -1;
//	}
//
//	int frameCount = 0;
//	time_t startTime, curTime;
//	clock_t startClock, curClock;
//	double duration, fps;
//	startTime = time(NULL); // ��¼��ʼʱ��
//	startClock = clock(); // ��¼��ʼʱ��
//
//	std::vector<Detection> result;
//	cv::VideoCapture cap;
//	cap.open(imagePath);
//	cv::Mat frame;
//	int frame_id = 0;
//	while (cap.isOpened()) {
//		try {
//			cap.read(frame);
//			//clock_t start = clock();
//			result = yolo.detect(frame, confThreshold, iouThreshold);
//			//clock_t ends = clock();
//			//std::cout << "Runing Time:" << (double)(ends - start) / CLOCKS_PER_SEC << std::endl;
//
//			++frameCount; // ֡��������1
//			curTime = time(NULL); // ��¼��ǰʱ��
//			curClock = clock(); // ��¼��ǰʱ��
//			duration = difftime(curTime, startTime) + (double)(curClock - startClock) / CLOCKS_PER_SEC;
//			if (duration >= 1.0) { // ÿ�����һ��fps
//				fps = frameCount / duration;
//				printf("FPS: %.2f\n", fps);
//				frameCount = 0; // ����֡������
//				startTime = time(NULL); // ���¿�ʼʱ��
//				startClock = clock(); // ���¿�ʼʱ����
//			}
//		}
//		catch (const std::exception& e) {
//			std::cerr << e.what() << std::endl;
//			return -1;
//		}
//		yolo.visualizeDetection(frame, result, classNames);
//		frame_id += 1;
//		std::string filename = output_dir + "image_" + std::to_string(frame_id) + ".png";
//		cv::imwrite(filename, frame);
//	}
//
//	return 0;
//}
//
