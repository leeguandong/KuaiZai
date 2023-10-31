#include "dbnet.h"
#include "clipper.h"

#define PRINT_FLAG 0
#define VISUAL_FLAG 0

Text_Detect::Text_Detect(const char* model_param, const char* model_bin, const float thresh,
	const float box_thresh, const float unclip_ratio) {
	this->thresh_ = thresh;
	this->box_thresh_ = box_thresh;
	this->unclip_ratio_ = unclip_ratio;
	this->model_param_ = model_param;
	this->model_bin_ = model_bin;
#if PRINT_FLAG
	printf("��ʼ����ģ��! \n");
#endif
	Model.load_param(model_param);
	Model.load_model(model_bin);
#if PRINT_FLAG
	printf("����ģ�ͳɹ���\n");
	std::cout << "���õĲ�����thresh: " << thresh << " box_thresh: " << box_thresh << " unclip_ratio: " << unclip_ratio << std::endl;
#endif
}

Text_Detect::~Text_Detect() {}


//2.Ԥ����ʽ���� PSENet��Դtensorflow��Ԥ����resize_image��ͼ����resize�������32���ر���������ģ����Ԥ��һЩ����Ƚϴ���߽�С��ͼƬЧ���ϲ�
void ResizeImgType2(const cv::Mat& img, cv::Mat& resize_img, const int& max_size_len, int& resize_h, int& resize_w) {
	int w = img.cols;
	int h = img.rows;

	float ratio = 1.f;
	int max_wh = w > h ? w : h;
	if (max_wh > max_size_len) { //���ͼƬ���߸���max_size_len,��Ҫ���¼���ratio
		if (h > w) {
			ratio = float(max_size_len) / float(h);
		}
		else {
			ratio = float(max_size_len) / float(w);
		}
	}
	resize_h = int(float(h) * ratio); // ratio����ߺ�960�ı�ֵ�������Լ����960��
	resize_w = int(float(w) * ratio);
	if (resize_h % 32 == 0)
		resize_h = resize_h;
	else
		resize_h = (resize_h / 32 + 1) * 32; // һ��Ҫת��32�ı���
	if (resize_w % 32 == 0)
		resize_w = resize_w;
	else
		resize_w = (resize_w / 32 + 1) * 32;
	cv::resize(img, resize_img, cv::Size(int(resize_w), int(resize_h)));
}



// ��ȡͼ��padding_resize��ָ��ͼ��ߴ�ķ������
cv::Mat get_affine_transform(const cv::Point2f& center, const float img_maxsize, const float target_size,
	const int inv = 0) {
	cv::Point2f srcTriangle[3]; //����任��Ҫ���㣬�˴�ѡ����һ��������
	cv::Point2f dstTriangle[3];

	srcTriangle[0] = center;
	srcTriangle[1] = center + cv::Point2f(0, img_maxsize / 2.0);
	if (center.x >= center.y) { // ����ԭͼ�Ŀ�ȴ��ڸ߶�
		srcTriangle[2] = cv::Point2f(0, center.y - center.x);
	}
	else {
		srcTriangle[2] = cv::Point2f(center.x - center.y, 0);
	}

	dstTriangle[0] = cv::Point2f(target_size / 2.0, target_size / 2.0);
	dstTriangle[1] = dstTriangle[0] + cv::Point2f(0, target_size / 2.0);
	dstTriangle[2] = cv::Point2f(0, 0);

	// �������任����
	cv::Mat affineMat(2, 3, CV_32FC1); //2��3�У��������������Ϊ 32 λ��������ÿ��Ԫ��ռ�� 1 ��ͨ����
	if (inv == 0) {
		affineMat = cv::getAffineTransform(srcTriangle, dstTriangle);
	}
	else {
		affineMat = cv::getAffineTransform(dstTriangle, srcTriangle);
	}
	return affineMat;
}

/*
bool cvpointcompare(cv::Point a, cv::Point b) {
	return a.x < b.x;
}

void Text_Detect::get_mini_boxes(std::vector<cv::Point>& contour, cv::Point2f minibox[4], float& minedgesize, float& perimeter, float& area) {

	// ʹ�� cv::minAreaRect(contour) ��ȡ��������С��Ӿ��� textrect��
	cv::RotatedRect textrect = cv::minAreaRect(contour);
	// ʹ�� cv::boxPoints(textrect, boxPoints2f) ����С��Ӿ��ε��ĸ����㱣�浽 boxPoints2f �С�
	cv::Mat boxPoints2f;
	cv::boxPoints(textrect, boxPoints2f);
	// ��boxPoints2f������ָ��ת��Ϊfloat*����ָ�룬�����丳ֵ������p1��
	// �� OpenCV �У�boxPoints2f ��һ�� cv::Mat ���󣬱�ʾ�����ĸ�����������ľ���boxPoints2f.data ���ؾ������ݵ�ָ�루����һ�����صĵ�ַ����Ĭ��������� uchar* ���͵�ָ�롣
	// �����������Ͳ�ͬ��������Ҫ�� uchar* ָ��ת��Ϊ float* ָ�룬�Ա�������������ͨ������ת�������(float*) ��ʵ�֡����գ����� p1 ��ָ�� boxPoints2f ���ݵ��ڴ��ַ�����ұ�����Ϊ float ���͵�ָ�롣
	// ͨ��������������ǿ���ʹ�� p1 �����ʺʹ��� boxPoints2f ����ĸ��������ݡ��ڴ���ĺ������֣����ǿ��Կ��� p1 ��������ȡ���εĶ������ꡣ
	float* p1 = (float*)boxPoints2f.data;
	// �� p1 �еĶ�������ת��Ϊ cv::Point2f ���ͣ������浽 tmpvev �С�
	std::vector<cv::Point2f> tmpvev;
	for (int i = 0; i < 4; ++i, p1 += 2) {
		tmpvev.push_back(cv::Point2f(p1[0], p1[1]));
	}
	// �� tmpvev �еĶ��㰴�� cvpointcompare ��������������ȷ�����εĶ��㰴��˳ʱ�����ʱ���˳�����С�
	std::sort(tmpvev.begin(), tmpvev.end(), cvpointcompare);
}
*/

void Text_Detect::get_mini_boxes(std::vector<cv::Point>& contour, cv::Point2f minibox[4], float& minedgesize, float& perimeter, float& area) {
	// ʹ�� cv::minAreaRect(contour) ��ȡ��������С��Ӿ��� textrect��
	cv::RotatedRect textrect = cv::minAreaRect(contour);
	// ʹ�� textrect.points() ����С��Ӿ��ε��ĸ���������洢�� boxPoints2f �����С�
	cv::Point2f boxPoints2f[4];
	textrect.points(boxPoints2f);

	// �ȽϺ���ͨ���Ƚϵ�� x ������ȷ�����˳�򣬼����մ����ҵ�˳������
	// [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; } ��һ�� lambda ���ʽ�����ڶ���һ���Զ���ıȽϺ�����
	// ����� lambda ���ʽ�У�������������� cv::Point2f ���͵����� a �� b���ȽϺ���ͨ���Ƚ���������� x ������ȷ�����ǵ�˳��
	// lambda ���ʽʹ�÷�����[] ��ָ�������б�����������У������б�Ϊ�գ���ʾû�в����κ��ⲿ������
	// lambda ���ʽ�ķ���ֵ��һ������ֵ����ʾ������ıȽϽ���������һ���� a �� x ����С�ڵڶ����� b �� x ���꣬�򷵻� true�����򷵻� false��
	// ��� lambda ���ʽ�ڴ����б����� std::sort() �����ĵ�����������������ıȽϺ����������� std::sort() �������յ�� x ������������������
	std::sort(std::begin(boxPoints2f), std::end(boxPoints2f), [](const cv::Point2f& a, const cv::Point2f& b) {
		return a.x < b.x; });

	// ����������˳��ѡ����С��Ӿ��ε��ĸ����㣬�������Ǵ洢��������� minibox �����С�
	int index1 = (boxPoints2f[1].y > boxPoints2f[0].y) ? 0 : 1;
	int index4 = (index1 == 0) ? 1 : 0;
	int index2 = (boxPoints2f[3].y > boxPoints2f[2].y) ? 2 : 3;
	int index3 = (index2 == 2) ? 3 : 2;
	minibox[0] = boxPoints2f[index1];
	minibox[1] = boxPoints2f[index2];
	minibox[2] = boxPoints2f[index3];
	minibox[3] = boxPoints2f[index4];

	// ������С��Ӿ��ε���̱߳ߴ硢�ܳ�����������ֱ�洢��������� minedgesize��perimeter �� area �С�
	minedgesize = (std::min)(textrect.size.width, textrect.size.height);
	perimeter = 2.f * (textrect.size.width + textrect.size.height);
	area = textrect.size.width * textrect.size.height;
}


float Text_Detect::box_score_fast(cv::Mat& probability_map, std::vector<cv::Point>& contour) {
	std::vector<cv::Point> box = contour;
	int width = probability_map.cols;
	int height = probability_map.rows;
	int xmax = -1, xmin = 1000000, ymax = -1, ymin = 1000000;
	// ʹ��ѭ������������ÿ���㣬���� xmax��xmin��ymax��ymin ��ֵ��ʹ��ֱ𱣳������� x �� y �����������С���ꡣ
	for (int i = 0; i < box.size(); ++i) {
		if (xmax < box[i].x)
			xmax = box[i].x;
		if (xmin > box[i].x)
			xmin = box[i].x;
		if (ymax < box[i].y)
			ymax = box[i].y;
		if (ymin > box[i].y)
			ymin = box[i].y;
	}
	// �� xmax��xmin��ymax��ymin ��ֵ�޶��ں����ͼƬ�ߴ緶Χ�ڡ�
	xmax = (std::min)((std::max)(xmax, 0), width - 1);
	xmin = (std::max)((std::min)(xmin, width - 1), 0);
	ymax = (std::min)((std::max)(ymax, 0), height - 1);
	ymin = (std::max)((std::min)(ymin, height - 1), 0);

	// ʹ��ѭ������������ÿ���㣬��ÿ����������ȥ xmin �� ymin���൱�ڽ���С��Ӿ��ε����Ͻ��ƶ���ԭ�㡣
	for (int i = 0; i < box.size(); ++i)
	{
		box[i].x = box[i].x - xmin;
		box[i].y = box[i].y - ymin;
		//std::cout<<box[i]<<std::endl;
	}

	// ������һ�� maskmat ��������һ�� CV_8UC1 ���͵ĵ�ͨ��ͼ�����СΪ (ymax - ymin + 1) �У�(xmax - xmin + 1) �С���ʼʱ�������ر�����Ϊ��ɫ��ֵΪ0����
	cv::Mat maskmat(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1, cv::Scalar(0, 0, 0));
	// ʹ�� box �Ķ������������ maskmat �Ͻ��ж������䣬��������ᱻ����Ϊ��ɫ��ֵΪ1����ͨ�� cv::Scalar(1, 1, 1) ָ�������ɫ��
	cv::fillPoly(maskmat, std::vector<std::vector<cv::Point>>{ box }, cv::Scalar(1, 1, 1), 1);
	// �������� probability_map ����ͼ���ָ�������ڣ�����Ϊ maskmat �����ص�ƽ��ֵ��
	// ����ʹ�� cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1) �����˸���Ȥ����ROI����Ȼ��ͨ�� clone() �����˸���Ȥ����ĸ��������ʹ�� cv::mean() ���������������ص�ƽ��ֵ�����ؽ���� [0] ������ʾ��ȡ����ƽ��ֵ��
	return cv::mean(probability_map(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)), maskmat)[0];
}


void Text_Detect::unclip(cv::Point2f minibox[4], float& perimeter, float& area, float& unclip_ratio, std::vector<cv::Point>& expand_contour) {
	// ��������ʾ��Ҫ���߽�������չ�ľ���
	float distance = area * unclip_ratio / perimeter;
	ClipperLib::Path poly;
	ClipperLib::ClipperOffset offset;

	// ����һ��ClipperLib::Path����poly��������С��Ӿ��ε��ĸ���������ת��Ϊ���ͣ�����ӵ�poly�С�
	for (int i = 0; i < 4; ++i) {
		poly.push_back(ClipperLib::IntPoint(minibox[i].x, minibox[i].y));
	}
	// ��һ��ClipperLib::ClipperOffset����offset����ʹ��AddPath������poly��ӵ�offset�С�
	offset.AddPath(poly, ClipperLib::jtRound, ClipperLib::etClosedPolygon);

	// ����һ��ClipperLib::Paths����polys������poly��ӵ�polys�С�		
	ClipperLib::Paths polys;
	polys.push_back(poly);
	// ����Execute������polys����unclip����������������к�Ľ���洢��polys�У���չ����Ϊdistance��
	offset.Execute(polys, distance);

	// ����polys�е�ÿ��·���������еĵ�����ת��Ϊcv::Point��ʽ������ӵ�expand_contour�С�
	// ���ش�������չ����expand_contour
	for (int i = 0; i < polys.size(); i++)
	{
		for (int j = 0; j < polys[polys.size() - 1].size(); j++)
		{
			expand_contour.emplace_back(polys[i][j].X, polys[i][j].Y);    //��polysת��Ϊcontour�ĸ�ʽ
		}
	}
}


cv::Point2f transform_preds(const cv::Mat& warpMat, const cv::Point2f& pt) {
	// warpMat ����任����pt���任������
	// ����һ����СΪ(1, 3)����������Ϊ64λ�����͵�pt_mid����
	cv::Mat pt_mid(1, 3, CV_64FC1);
	// ������ĵ������x��y�����ֱ�ֵ��pt_mid����ĵ�һ�����ݣ��������һ��Ԫ������Ϊ1.0����ʾ��������е�1.
	pt_mid.at<double>(0, 0) = pt.x;
	pt_mid.at<double>(0, 1) = pt.y;
	pt_mid.at<double>(0, 2) = 1.0;
	// ����һ��new_pt����ͨ����pt_mid���������任����warpMat��˵õ���
	cv::Mat new_pt = pt_mid * warpMat;
	// ��ȡnew_pt�����еĵ�һ��Ԫ�غ͵ڶ���Ԫ����Ϊx��y���꣬����һ��cv::Point2f���͵Ķ��󣬱�ʾ�任��ĵ����ꡣ


}


std::vector<DetTextBox> Text_Detect::detect(const cv::Mat& image, const int& target_size) {
	double time_preprocess = (double)cv::getTickCount();

	std::vector<DetTextBox> textboxes;
	int width, height, resize_w, resize_h;
	width = resize_w = image.cols;
	height = resize_h = image.rows;
	cv::Point2f center(width / 2.0, height / 2.0);
	int img_maxsize = width > height ? width : height; // ���width>height��ȡwidth
	int img_minsize = width < height ? width : height; // ���width<height��ȡwidth
	int square_size = 640;

	cv::Mat dst;
	bool use_padding_resize = false;
	if (1.0 * height / width >= 5 || 1.0 * width / height >= 5 && img_minsize <= 32) {
		// padding_resize Ԥ������Ҫ�ǶԼ��˿�Ȼ�����sizeС��32��padding����任��padding֮��ĳߴ���640*640
		use_padding_resize = true;
		std::cout << "���õ���padding_resize" << std::endl;
		cv::Mat affineMat = get_affine_transform(center, img_maxsize, square_size, 0); // padding_size�ĳߴ���640*640�ķ�ͼ
		cv::warpAffine(image, dst, affineMat, cv::Size(square_size, square_size));//���ݷ���任���󣬽�ԭͼӳ�䵽640x640�ķ���ͼƬ��
		cv::imwrite("dst.jpg", dst);
	}
	else {//�ȱ�������
		use_padding_resize = false;
		std::cout << "���õ��ǵȱ���resize" << std::endl;
		ResizeImgType2(image, dst, target_size, resize_h, resize_w);//���ַ�ʽtarget_sizeĬ����960,dst�����ǵ�ַ
		std::cout << "resize_h, resize_w is " << resize_h << "," << resize_w << std::endl;
	}
	double time_end_preprocess = ((double)cv::getTickCount() - time_preprocess) / cv::getTickFrequency();
	std::cout << "�ı�Ԥ������ʱ�䣺" << time_end_preprocess << "��" << std::endl;

	double time_start_detect = (double)cv::getTickCount();
	// �� dst ת��Ϊ ncnn::Mat ���͵� dbnet_input��dst ��һ�� OpenCV �� cv::Mat �������а�����ͼ�����ݡ�
	// from_pixels �������ڴ�ԭʼͼ�����ݴ��� ncnn::Mat ����ncnn::Mat::PIXEL_BGR2RGB ��ʾͼ���ͨ�������� BGR�������� RGB��
	ncnn::Mat dbnet_input = ncnn::Mat::from_pixels(dst.data, ncnn::Mat::PIXEL_BGR2RGB, dst.cols, dst.rows);
#if PRINT_FLAG
	printf("dbnet_input: c=%d,resize_w=%d,resize_h=%d\n", dbnet_input.c, dbnet_input.w, dbnet_input.h);
#endif
	// ʹ�� substract_mean_normalize ������ dbnet_input ���о�ֵ�����͹�һ������mean_vals_dbnet �� norm_vals_dbnet ��Ԥ����ľ�ֵ�͹�һ�����������ڶ�ͼ�����ݽ���Ԥ����
	dbnet_input.substract_mean_normalize(mean_vals_dbnet, norm_vals_dbnet);
	// ���� Extractor ���� dbnet_ex�����ڴ�ģ������ȡ������Model ��Ԥ�����ģ�Ͷ��󣬱�ʾ�Ѽ��ص�ģ�͡�
	ncnn::Extractor dbnet_ex = Model.create_extractor();
	// ���� dbnet_ex ���߳�����Ϊ num_thread�����Ż����м������ܡ�
	dbnet_ex.set_num_threads(num_thread);
	// ʹ�� input ������ dbnet_input ����Ϊ dbnet_ex �����롣input ��������ָ�����������ƺͶ�Ӧ���������ݡ�
	dbnet_ex.input("input", dbnet_input);
	// ���� dbnet_out ��Ϊ��������������
	ncnn::Mat dbnet_out;
	// ʹ�� extract ������ dbnet_ex ����ȡ��������������������� dbnet_out �С�"out" ��ָ�������������ơ�
	dbnet_ex.extract("out", dbnet_out);
#if PRINT_FLAG
	printf("dbnet_out: c=%d,resize_w=%d,resize_h=%d\n", dbnet_out.c, dbnet_out.w, dbnet_out.h);
#endif
	double time_end_detect = ((double)cv::getTickCount() - time_start_detect) / cv::getTickFrequency();
	std::cout << "�ı�����ƶϻ���ʱ�䣺" << time_end_detect << "��" << std::endl;

	double time_start_postprocess = (double)cv::getTickCount();
	// ������һ����Ϊ probability_map �� cv::Mat ���������洢����ͼ�����ݡ�
	// (const float*)dbnet_out.channel(0) �ǽ�����ͼ�е�0��ͨ��������ת��Ϊ const float* ���͵�ָ�롣ͨ�� (void*) ǿ������ת��Ϊ void* ���͵�ָ�룬��Ϊ cv::Mat ���캯����Ҫ����һ�� void* ���͵�ָ����Ϊ���ݡ�
	cv::Mat probability_map(dbnet_out.h, dbnet_out.w, CV_32FC1, (void*)(const float*)dbnet_out.channel(0));
	// ����һ���µ���Ϊ binary_map �� Mat ���󣬲�ʹ����ֵ thresh_ �� probability_map ��ֵ������������ֵ��������Ϊ��ɫ��255����С�ڵ�����ֵ��������Ϊ��ɫ��0���������õ��� norfmapmat ��һ��CV_8UC1���͵Ķ�ֵͼ��8λ�޷������ͣ���ͨ������
	cv::Mat binary_map;
	binary_map = probability_map > thresh_; // 
#if VISUAL_FLAG
	cv::imwrite("thersh.jpg", binary_map);
#endif

	textboxes.clear();
	std::vector<std::vector<cv::Point>> contours; // ��ά����
	// ʹ��OpenCV�� findContours ������ binary_map ���ҵ����е����������洢�� contours �С�
	// ���� RETR_LIST ��ʾ��ȡ���е����������� CHAIN_APPROX_SIMPLE ��ʾѹ��ˮƽ����ֱ�ͶԽ��߷�������ص㣬�������˵㡣
	// contours ��һ�����������д洢�˼�⵽��������Ϣ������һ�� std::vector<std::vector<cv::Point>> ���͵ı������㼶�ṹ���£�
	// ��� std::vector���洢���е��������ڲ� std::vector<cv::Point>��ÿ���ڲ� std::vector �洢һ�������ĵ㼯�ϡ�
	// ÿ����������һϵ�е� cv::Point ������ɣ���ʾ�����ϵĵ�����ꡣ����ͨ��ʹ���������������ڲ���������ȡ����������������ϵĵ㡣
	// ���磬contours[i] ��ʾ�� i ��������contours[i][j] ��ʾ�� i �������ϵĵ� j ���㡣
	// ÿ�� cv::Point ��������������ԣ�x����� x ���ꡣy����� y ���ꡣ
	// ͨ������ contours �����е�Ԫ�أ����Ի�ȡÿ�����������е��������Ϣ��
	cv::findContours(binary_map, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
#if PRINT_FLAG    
	std::cout << "�����ĸ�����" << contours.size() << std::endl;
#endif

	// findcontours֮����ĵ�Ҫ������ϴ
	for (int i = 0; i < contours.size(); ++i) {
		// ���ú��� get_mini_boxes()�����뵱ǰ��������������������õ���С��Ӿ��ε��ĸ��������ꡢ��С�߳����ܳ��������
		cv::Point2f minibox[4];
		float minedgesize, perimeter, area;
		get_mini_boxes(contours[i], minibox, minedgesize, perimeter, area);

		// �����С�߳�С�� min_size_����������ǰ�����Ĵ���������һ��ѭ����
		if (minedgesize < min_size_)
			continue;

		// �������ͼ�͵�ǰ����������õ���ǰ�����ķ�����
		float score = box_score_fast(probability_map, contours[i]);

		// �������С�� box_thresh_����������ǰ�����Ĵ���������һ��ѭ����
		if (score < box_thresh_)
			continue;
#if PRINT_FLAG     
		std::cout << "��" << i << "�����ķ�����" << score << std::endl;
		std::cout << "��" << i << "�����õ���minibox:" << std::endl;
		for (int i = 0; i < 4; i++)
			std::cout << minibox[i] << std::endl;
#endif

		std::vector<cv::Point> expand_contour;
		unclip(minibox, perimeter, area, unclip_ratio_, expand_contour);
#if PRINT_FLAG        
		std::cout << "��" << i << "�����õ���expand_contour:" << std::endl;
		for (int i = 0; i < expand_contour.size(); i++)
			std::cout << expand_contour[i] << std::endl;
#endif

		// unclip֮�������¼�����С��Ӿ���
		get_mini_boxes(expand_contour, minibox, minedgesize, perimeter, area);
#if PRINT_FLAG                
		std::cout << "��" << i << "�����ڶ��εõ���minibox:" << std::endl;
		for (int i = 0; i < 4; i++)
			std::cout << minibox[i] << std::endl;
#endif    

		// �����С�߳�С���趨����ֵ(min_size_ + 2)���������������������к�������
		if (minedgesize < min_size_ + 2)
			continue;

		// Ҫӳ���ԭͼ���꣬ͨ���ڵڶ�������Ҫcrnnʶ��
		if (use_padding_resize) {
			// ��ͨ������get_affine_transform������ȡ����任����warpMat���þ������ڽ���С��Ӿ��ν��б任�����У�center��ʾ���ĵ����꣬img_maxsize��ʾͼ������ߴ磬square_size��ʾĿ��ߴ硣
			cv::Mat wrapMat = get_affine_transform(center, img_maxsize, square_size, 1).t();
			for (int j = 0; j < 4; ++j) {
				cv::Point2f tmp_pt(minibox[j].x, minibox[j].y);
				//tmp_pt =

			}
		}
		else {

		}



	}






	return textboxes;
}

