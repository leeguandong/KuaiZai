#include <opencv2/core/core.hpp>
#include <algorithm>
#include <chrono>
#include <math.h>
#include "vehicle.h"
#include "plate_det.h"
#include "plate_rec.h"
#include "utils.h"

//#define OS_PATH_SEP "/"

//int main() {
//	PlateDetector detect_plate_model("weights/pp_vehicle/ch_PP-OCRv3_det_infer.onnx");
//	TextRecognizer rec_plate_model("weights/pp_vehicle/ch_PP-OCRv3_rec_infer.onnx",
//		"weights/pp_vehicle/rec_word_dict.txt");
//
//	string image_path = "";
//	string image_dir = "data/202301301759_image";
//	string output_dir = "results";
//	int batch_size = 1;
//	//string video_out_name = "output.mp4";
//
//	if (!PathExists(output_dir)) {
//		MkDir(output_dir);
//	}
//
//	std::vector<std::string> all_img_paths;
//	std::vector<cv::String> cv_all_img_paths;
//	//std::vector<std::string> nameArr;
//	if (!image_path.empty() || !image_dir.empty()) {
//		if (!image_path.empty()) {
//			all_img_paths.push_back(image_path);
//		}
//		else {
//			cv::glob(image_dir, cv_all_img_paths);
//			// ��ͼƬ��������
//			size_t count = cv_all_img_paths.size();
//			for (int i = 0; i < count; ++i) {
//				// 1.��ȡ����·�����ļ���
//				string::size_type iPos = cv_all_img_paths[i].find_last_of(OS_PATH_SEP) + 1;
//				string filename = cv_all_img_paths[i].substr(iPos, cv_all_img_paths[i].length() - iPos);
//
//				// 2.��ȡ������׺���ļ���
//				string name = filename.substr(0, filename.rfind("."));
//
//				all_img_paths.emplace_back(name);
//				//all_img_paths.push_back(img_path);
//			}
//			sort(all_img_paths.begin(), all_img_paths.end(),
//				[](string a, string b) {return stoi(a) < stoi(b); });
//		}
//	}
//
//	int steps = ceil(float(all_img_paths.size()) / batch_size);
//	printf("total images = %d, batch_size = %d, total steps = %d\n",
//		all_img_paths.size(),
//		batch_size,
//		steps);
//
//	vector<double> det_times;
//	vector<string> plate_text_;
//	cv::Mat frame;
//	int frame_id = 1;
//	int skip_frame_num = 1; // �����֡����
//
//	auto total_start = std::chrono::steady_clock::now();
//
//	for (int idx = 0; idx < steps; idx++) {
//		//vector<Mat> batch_imgs;
//		//int left_image_cnt = all_img_paths.size() - idx * batch_size;
//		//if (left_image_cnt > batch_size) {
//		//	left_image_cnt = batch_size;
//		//}
//		//for (int bs = 0; bs < left_image_cnt; bs++) {
//		//	string image_file_path = all_img_paths.at(idx * batch_size + bs);
//		//	frame = cv::imread(image_file_path, 1);
//		//	//batch_imgs.insert(batch_imgs.end(), frame);
//		//}
//		string image_file_path = image_dir + "/" + all_img_paths.at(idx) + ".jpg";
//		frame = imread(image_file_path, 1);
//
//		if (frame_id % skip_frame_num != 0) {
//			frame_id += 1;
//			continue;
//		}
//
//		if (frame.empty()) {
//			break;
//		}
//		//vector<cv::Mat>  imgs;
//		//imgs.push_back(frame);
//		printf("detect frame: %d\n", frame_id);
//		Mat img = frame;
//		auto detect_plate_start = std::chrono::steady_clock::now();
//		vector<vector<Point2f>> results = detect_plate_model.detect(img);
//		auto detect_plate_end = std::chrono::steady_clock::now();
//
//		// ����Ƿ��г���
//		detect_plate_model.draw_pred(img, results);
//		namedWindow("detect_plate", WINDOW_NORMAL);
//		imshow("detect_plate", img);
//		waitKey(0);
//		destroyAllWindows();
//
//		auto rec_plate_start = std::chrono::steady_clock::now();
//		for (size_t i = 0; i < results.size(); i++) {
//			Point2f vertices[4];
//			bool detect_ = false;
//			for (int j = 0; j < 4; ++j) {
//				vertices[j].x = results[i][j].x;
//				vertices[j].y = results[i][j].y;
//				detect_ = isnan(results[i][j].x);
//				detect_ = isnan(results[i][j].y);
//			}
//			if (detect_)
//				continue;
//
//			Mat plate_img = rec_plate_model.get_rotate_crop_image_v1(img, vertices);
//			//imshow("plate_img", plate_img);
//			//waitKey(0);
//			//destroyAllWindows();
//
//			// ������ͼƬ��һЩ��Ԥ����
//			vector<Point> plate_box;
//			plate_box.push_back(Point2f(0, 0));
//			plate_box.push_back(Point2f(plate_img.rows, 0));
//			plate_box.push_back(Point2f(plate_img.rows, plate_img.cols));
//			plate_box.push_back(Point2f(0, plate_img.cols));
//			double area = contourArea(plate_box);
//			if (area < 500)
//				continue;
//			string plate_text = rec_plate_model.detect(plate_img);
//			//wstring plate_text_wstr = wstring(plate_text.begin(), plate_text.end()); // ת�ɿ��ַ���
//			// ����������һЩɸѡ
//			if (plate_text.length() > 4 && plate_text.length() < 10) {
//				string plate_text_ch;
//				string plate_text_en;
//				//map<char, string>::iterator it;
//
//				// �ȰѺ���ȡ������Ȼ���ڰѺ����滻��
//				for (int i = 0; i < plate_text.length(); i++) {
//					if (plate_text[i] < 255 && plate_text[i]>0) { //�����ASCII�ַ���ΧΪ0-255,����,����һ���ֽ�
//						plate_text_en.append(plate_text.substr(i, 1));
//					}
//					else { // <0,>255���Ǻ���,���������ֽ�
//						plate_text_ch.append(plate_text.substr(i, 2));
//						++i;
//					}
//				}
//
//				if (plate_text_ch == "��")
//					plate_text_en = plate_text_en.insert(0, "JS-");
//				else if (plate_text_ch == "��")
//					plate_text_en = plate_text_en.insert(0, "ZJ-");
//				else if (plate_text_ch == "��")
//					plate_text_en = plate_text_en.insert(0, "GD-");
//				else if (plate_text_ch == "��")
//					plate_text_en = plate_text_en.insert(0, "BJ-");
//				else if (plate_text_ch == "��")
//					plate_text_en = plate_text_en.insert(0, "TJ-");
//				else if (plate_text_ch == "��")
//					plate_text_en = plate_text_en.insert(0, "HE-");
//				else if (plate_text_ch == "��")
//					plate_text_en = plate_text_en.insert(0, "SX-");
//				else if (plate_text_ch == "��")
//					plate_text_en = plate_text_en.insert(0, "NM-");
//				else if (plate_text_ch == "��")
//					plate_text_en = plate_text_en.insert(0, "LN-");
//				else if (plate_text_ch == "��")
//					plate_text_en = plate_text_en.insert(0, "HLJ-");
//				else if (plate_text_ch == "��")
//					plate_text_en = plate_text_en.insert(0, "SH-");
//				else if (plate_text_ch == "��")
//					plate_text_en = plate_text_en.insert(0, "JL-");
//				else if (plate_text_ch == "��")
//					plate_text_en = plate_text_en.insert(0, "JS-");
//				else if (plate_text_ch == "��")
//					plate_text_en = plate_text_en.insert(0, "AH-");
//				else if (plate_text_ch == "��")
//					plate_text_en = plate_text_en.insert(0, "JX-");
//				else if (plate_text_ch == "³")
//					plate_text_en = plate_text_en.insert(0, "SD-");
//				else if (plate_text_ch == "ԥ")
//					plate_text_en = plate_text_en.insert(0, "HA-");
//				else if (plate_text_ch == "��")
//					plate_text_en = plate_text_en.insert(0, "HB-");
//				else if (plate_text_ch == "��")
//					plate_text_en = plate_text_en.insert(0, "HN-");
//				else if (plate_text_ch == "��")
//					plate_text_en = plate_text_en.insert(0, "GX-");
//				else if (plate_text_ch == "��")
//					plate_text_en = plate_text_en.insert(0, "HI-");
//				else if (plate_text_ch == "��")
//					plate_text_en = plate_text_en.insert(0, "CQ-");
//				else if (plate_text_ch == "��")
//					plate_text_en = plate_text_en.insert(0, "SC-");
//				else if (plate_text_ch == "��")
//					plate_text_en = plate_text_en.insert(0, "GZ-");
//				else if (plate_text_ch == "��")
//					plate_text_en = plate_text_en.insert(0, "YN-");
//				else if (plate_text_ch == "��")
//					plate_text_en = plate_text_en.insert(0, "XZ-");
//				else if (plate_text_ch == "��")
//					plate_text_en = plate_text_en.insert(0, "SN-");
//				else if (plate_text_ch == "��")
//					plate_text_en = plate_text_en.insert(0, "GS-");
//				else if (plate_text_ch == "��")
//					plate_text_en = plate_text_en.insert(0, "QH-");
//				else if (plate_text_ch == "��")
//					plate_text_en = plate_text_en.insert(0, "NX-");
//				else if (plate_text_ch == "��")
//					plate_text_en = plate_text_en.insert(0, "FJ-");
//
//				/*it = rec_plate_model.simcode.find(plate_text_ch);
//				if (it != rec_plate_model.simcode.end()) {
//					plate_text_en = plate_text_en.insert(0, it->second);
//				}*/
//
//				/*	replace(plate_text.begin(), plate_text.end(),
//								it->first, it->second);*/
//
//				putText(img, plate_text_en, Point((int)results[i][3].x, (int)results[i][3].y - 10),
//					FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
//				//namedWindow("detect_plate", WINDOW_NORMAL);
//				//imshow("detect_plate", img);
//				//waitKey(0);
//				//destroyAllWindows();
//
//				plate_text_.push_back(plate_text);
//				printf("plate: %s\n", plate_text.c_str());
//			}
//		}
//		auto rec_plate_end = std::chrono::steady_clock::now();
//
//		//video_out.write(img);
//		std::vector<int> compression_params;
//		compression_params.push_back(IMWRITE_JPEG_QUALITY);
//		compression_params.push_back(95);
//		//std::string image_file_path = all_img_paths.at(idx * batch_size + i);
//		string output_path(output_dir);
//		if (output_dir.rfind(OS_PATH_SEP) != output_dir.size() - 1) {
//			output_path += OS_PATH_SEP;
//		}
//		output_path +=
//			image_file_path.substr(image_file_path.find_last_of('/') + 1);
//		cv::imwrite(output_path, img, compression_params);
//		//printf("Visualized output saved as %s\n", output_path.c_str());
//		frame_id += 1;
//
//		/*std::chrono::duration<float> detect_diff = detect_plate_end - detect_plate_start;
//		std::chrono::duration<float> rec_diff = rec_plate_end - rec_plate_start;
//		det_times.push_back(double(detect_diff.count() * 1000));
//		det_times.push_back(double(rec_diff.count() * 1000));*/
//
//		// ������5֡��������ͬ����5֡ǰ3֡���Ʋ�ͬ������Ϊ���ʱ��
//
//		bool enter_parking = false;
//		bool leave_parking = false;
//		if (plate_text_.size() > 7) {
//			int i = int(plate_text_.size() - 1);
//			//for (int i = 7; i < plate_text_.size(); i++) {
//			if ((plate_text_[i] == plate_text_[i - 1]) &&
//				plate_text_[i - 1] == plate_text_[i - 2] &&
//				plate_text_[i - 2] == plate_text_[i - 3] &&
//				plate_text_[i - 3] == plate_text_[i - 4] &&
//				plate_text_[i - 4] != plate_text_[i - 5] &&
//				plate_text_[i - 5] != plate_text_[i - 6] &&
//				(not enter_parking)) {
//				auto now = std::chrono::system_clock::now();
//				//ͨ����ͬ���Ȼ�ȡ���ĺ�����
//				//uint64_t dis_millseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count()
//				//	- std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count() * 1000;
//				time_t tt = std::chrono::system_clock::to_time_t(now);
//				auto time_tm = localtime(&tt);
//				char strTime[25] = { 0 };
//				sprintf(strTime, "%d-%02d-%02d %02d:%02d:%02d", time_tm->tm_year + 1900,
//					time_tm->tm_mon + 1, time_tm->tm_mday, time_tm->tm_hour,
//					time_tm->tm_min, time_tm->tm_sec);
//				std::cout << "enter parking: " << strTime << std::endl;
//				enter_parking = true;
//			}
//			if (plate_text_[i].length() < 4 &&
//				plate_text_[i - 1].length() < 4 &&
//				plate_text_[i - 1] != plate_text_[i - 2] &&
//				plate_text_[i - 2] == plate_text_[i - 3] &&
//				plate_text_[i - 3] == plate_text_[i - 4] &&
//				plate_text_[i - 4] == plate_text_[i - 5] &&
//				not leave_parking) {
//				auto now = std::chrono::system_clock::now();
//				//ͨ����ͬ���Ȼ�ȡ���ĺ�����
//				//uint64_t dis_millseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count()
//				//	- std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count() * 1000;
//				time_t tt = std::chrono::system_clock::to_time_t(now);
//				auto time_tm = localtime(&tt);
//				char strTime[25] = { 0 };
//				sprintf(strTime, "%d-%02d-%02d %02d:%02d:%02d", time_tm->tm_year + 1900,
//					time_tm->tm_mon + 1, time_tm->tm_mday, time_tm->tm_hour,
//					time_tm->tm_min, time_tm->tm_sec);
//				std::cout << "leave parking: " << strTime << std::endl;
//				leave_parking = true;
//			}
//			//}
//		}
//	}
//	//while (capture.read(frame)) {
//	//	
//	//}
//
//	//capture.release();
//	//video_out.release();
//
//	auto total_end = std::chrono::steady_clock::now();
//	std::chrono::duration<float> total_diff = total_end - total_start;
//	//total_time = double(total.count() * 1000);
//	double fps = frame_id / double(total_diff.count());
//	printf("total times: %f, total frames: %d, fps: %f\n",
//		double(total_diff.count()), frame_id, fps);
//return 0;
//}
//
