#include "vehicle.h"
#include "vehicleattr_rec.h"
#include "plate_det.h"
#include "plate_rec.h"
//#include "utils.h"

/*


int main() {
	PP_YOLOE detect_vehicle_model("weights/pp_vehicle/mot_ppyoloe_s_36e_ppvehicle.onnx", 0.6);
	VehicleAttr rec_vehicle_attr_model("weights/pp_vehicle/vehicle_attribute_model.onnx");
	PlateDetector detect_plate_model("weights/pp_vehicle/ch_PP-OCRv3_det_infer.onnx");
	TextRecognizer rec_plate_model("weights/pp_vehicle/ch_PP-OCRv3_rec_infer.onnx",
		"weights/pp_vehicle/rec_word_dict.txt");

	string imgpath = "data/street_00001.jpg";
	Mat img = imread(imgpath);
	vector<BoxInfo> boxes = detect_vehicle_model.detect(img);
	for (size_t n = 0; n < boxes.size(); ++n) {
		Rect rect;
		rect.x = boxes[n].xmin;
		rect.y = boxes[n].ymin;
		rect.width = boxes[n].xmax - boxes[n].xmin;
		rect.height = boxes[n].ymax - boxes[n].ymin;
		Mat crop_img = img(rect);
		string color_res_str = "Color: ";
		string type_res_str = "Type: ";
		rec_vehicle_attr_model.detect(crop_img, color_res_str, type_res_str);
		vector<vector<Point2f>> results = detect_plate_model.detect(crop_img);

		detect_plate_model.draw_pred(crop_img, results);
		namedWindow("detect_plate", WINDOW_NORMAL);
		imshow("detect_plate", crop_img);
		waitKey(0);
		destroyAllWindows();

		for (size_t i = 0; i < results.size(); i++) {
			Point2f vertices[4];
			for (int j = 0; j < 4; ++j) {
				vertices[j].x = results[i][j].x;
				vertices[j].y = results[i][j].y;
			}

			Mat plate_img = rec_plate_model.get_rotate_crop_image(crop_img, vertices);
			imshow("plate_img", plate_img);
			waitKey(0);
			destroyAllWindows();

			string plate_text = rec_plate_model.detect(plate_img);
			cout << plate_text << endl;  // 输出也是乱码，txt需要转成ANSI，但是会丢失字符
			//putTextHusky(img, plate_text.c_str(), cv::Point(boxes[n].xmin, boxes[n].ymin - 20), cv::Scalar(0, 255, 0),30, "Arial");
		}
		string label = type_res_str + "," + color_res_str;
		rectangle(img, Point(boxes[n].xmin, boxes[n].ymin), Point(boxes[n].xmax, boxes[n].ymax), Scalar(0, 0, 255), 2);
		putText(img, label, Point(boxes[n].xmin, boxes[n].ymin - 10), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
	}

	imwrite("data/result_plate.jpg", img);
	static const string kWinName = "Deep learning object detection in ONNXRuntime";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, img);
	waitKey(0);
	destroyAllWindows();
}

*/