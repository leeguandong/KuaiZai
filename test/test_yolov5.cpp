#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>
#include "yolov5_libtorch.h"

//int main() {
//
//	// libtorch模式来调用yolov5
//	Yolov5Libtorch yolo(torch::cuda::is_available() ? "weights/yolov5_person/weights/best.torchscript" : "weights/yolov5_person/weights/best.torchscript", torch::cuda::is_available()); // 第二个参数为是否启动cuda,第一个参数表示如果cuda可用就用第一个路径，否则用第二个路径
//	//Yolov5Libtorch yolo(torch::cuda::is_available() ? "F:/Dataset/qiantu/yolov5-libtorch-main/test/yolov5s.cuda.pt" : "F:/Dataset/qiantu/yolov5-libtorch-main/test/yolov5s.cpu.pt", torch::cuda::is_available()); // 
//	yolo.prediction(torch::rand({ 1,3,640,640 })); // 模型预热
//
//	std::ifstream f("weights/yolov5_person/person.txt");
//	//std::ifstream f("F:/Dataset/qiantu/yolov5-libtorch-main/test/coco.txt");
//	std::string name = "";
//	int i = 0;
//	std::map<int, std::string> labels;
//	while (std::getline(f, name)) {
//		labels.insert(std::pair<int, std::string>(i, name));
//		i++;
//	}
//	std::string video_path = "data/2月8日(1).mp4";
//	std::string output_dir = "./results/2月8日(1)/";
//	cv::VideoCapture cap;
//	cap.open(video_path.c_str());
//	//cv::VideoCapture cap = cv::VideoCapture(0);
//
//	// 设置宽高，对原始图进行统一处理，是yolov5训练的宽高保持一致
//	// 传入方式是构造yolov5对象时传入width默认值是640，height是640
//	//cap.set(cv::CAP_PROP_FRAME_WIDTH,1000);
//	//cap.set(cv::CAP_PROP_FRAME_HEIGHT, 800);
//	cv::Mat frame;
//	int frame_id = 0;
//	while (cap.isOpened()) {
//		//读取一帧
//		cap.read(frame);
//		if (frame.empty()) {
//			std::cout << "Read frame failed!" << std::endl;
//			break;
//		}
//		clock_t start = clock();
//		std::vector<torch::Tensor> pred = yolo.prediction(frame);
//		clock_t ends = clock();
//		std::cout << "Runing Time:" << (double)(ends - start) / CLOCKS_PER_SEC << std::endl;
//		frame = yolo.drawRectangle(frame, pred[0], labels);
//		// show图片
//	/*	cv::imshow("", frame);
//		if (cv::waitKey(1) == 27) break;*/
//		frame_id += 1;
//		std::string filename = output_dir + "image_" + std::to_string(frame_id) + ".png";
//		cv::imwrite(filename, frame);
//	}
//
//
//	return 0;
//}

