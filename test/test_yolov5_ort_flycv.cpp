//#include <iostream>
//#include <flycv.h>
//#include <time.h>
//
//#include "cmdline.h"
//#include "yolov5_ort_flycv.h"
//
//
////int main(int argc, char* argv[]) {
//int main() {
//
//	// ./KuaiZai.exe --model_path "weights/yolov5_person/weights/best.onnx" --video "data/2月8日(1).mp4" --class_naes "weights/yolov5_person/person.txt" 但是这种写法很难debug
//	//cmdline::parser cmd;
//	//cmd.add<std::string>("model_path", 'm', "path to onnx model", false, "weights/yolov5_person/weights/best.onnx");
//	//cmd.add<std::string>("image",'i',"Image source to be detected",true,"");	
//	//cmd.add<std::string>("class_names", 'c', "path to class names file", false, "weights/yolov5_person/person.txt");
//	//cmd.add("gpu", '\0', "Inference on cuda device");
//	////cmd.parse_check(argc, argv);
//
//	//bool isGPU = cmd.exist("gpu");
//	//const std::string imagePath = cmd.get<std::string>("video");
//	//const std::string modelPath = cmd.get<std::string>("model_path");
//	//const std::string classNamesPath = cmd.get<std::string>("class_names");
//
//	bool isGPU = false;
//	const std::string imagePath = "data/2月8日(1)_Moment.jpg";
//	const std::string modelPath = "weights/yolov5_person/weights/best.onnx";
//	const std::string classNamesPath = "weights/yolov5_person/person.txt";
//	const std::string output_dir = "./results/";
//
//	Yolov5ORT yolo(modelPath, isGPU, fcv::Size(640, 640)); // 模型初始化
//	const float confThreshold = 0.3f;
//	const float iouThreshold = 0.4f;
//
//	const std::vector<std::string> classNames = yolo.loadNames(classNamesPath);
//	if (classNames.empty()) {
//		std::cerr << "Error: Empty class names file." << std::endl;
//		return -1;
//	}
//
//	std::vector<Detection> result;
//	fcv::Mat image = fcv::imread(imagePath);
//
//	try {
//		clock_t start = clock();
//		result = yolo.detect(image, confThreshold, iouThreshold);
//		clock_t ends = clock();
//		std::cout << "Runing Time:" << (double)(ends - start) / CLOCKS_PER_SEC << std::endl;
//	}
//	catch (const std::exception& e) {
//		std::cerr << e.what() << std::endl;
//		return -1;
//	}
//
//	yolo.visualizeDetection(image, result, classNames);
//
//	std::string filename = output_dir + "image_" + std::to_string(1) + ".png";
//	fcv::imwrite(filename, image);
//
//	return 0;
//}
//
//


// 批量注释 CTRL+K+C，批量取消注释 CTRL+K+U
