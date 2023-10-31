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
	printf("开始加载模型! \n");
#endif
	Model.load_param(model_param);
	Model.load_model(model_bin);
#if PRINT_FLAG
	printf("载入模型成功！\n");
	std::cout << "设置的参数是thresh: " << thresh << " box_thresh: " << box_thresh << " unclip_ratio: " << unclip_ratio << std::endl;
#endif
}

Text_Detect::~Text_Detect() {}


//2.预处理方式二： PSENet开源tensorflow版预处理resize_image是图像宽高resize到最近的32像素倍数，但是模型在预测一些长宽比较大或者较小的图片效果较差
void ResizeImgType2(const cv::Mat& img, cv::Mat& resize_img, const int& max_size_len, int& resize_h, int& resize_w) {
	int w = img.cols;
	int h = img.rows;

	float ratio = 1.f;
	int max_wh = w > h ? w : h;
	if (max_wh > max_size_len) { //如果图片最大边高于max_size_len,需要重新计算ratio
		if (h > w) {
			ratio = float(max_size_len) / float(h);
		}
		else {
			ratio = float(max_size_len) / float(w);
		}
	}
	resize_h = int(float(h) * ratio); // ratio是最长边和960的比值，将最长边约束到960内
	resize_w = int(float(w) * ratio);
	if (resize_h % 32 == 0)
		resize_h = resize_h;
	else
		resize_h = (resize_h / 32 + 1) * 32; // 一定要转成32的倍数
	if (resize_w % 32 == 0)
		resize_w = resize_w;
	else
		resize_w = (resize_w / 32 + 1) * 32;
	cv::resize(img, resize_img, cv::Size(int(resize_w), int(resize_h)));
}



// 获取图像padding_resize到指定图像尺寸的放射矩阵
cv::Mat get_affine_transform(const cv::Point2f& center, const float img_maxsize, const float target_size,
	const int inv = 0) {
	cv::Point2f srcTriangle[3]; //仿射变换需要三点，此处选择了一个三角形
	cv::Point2f dstTriangle[3];

	srcTriangle[0] = center;
	srcTriangle[1] = center + cv::Point2f(0, img_maxsize / 2.0);
	if (center.x >= center.y) { // 代表原图的宽度大于高度
		srcTriangle[2] = cv::Point2f(0, center.y - center.x);
	}
	else {
		srcTriangle[2] = cv::Point2f(center.x - center.y, 0);
	}

	dstTriangle[0] = cv::Point2f(target_size / 2.0, target_size / 2.0);
	dstTriangle[1] = dstTriangle[0] + cv::Point2f(0, target_size / 2.0);
	dstTriangle[2] = cv::Point2f(0, 0);

	// 计算仿射变换矩阵
	cv::Mat affineMat(2, 3, CV_32FC1); //2行3列，矩阵的数据类型为 32 位浮点数，每个元素占用 1 个通道。
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

	// 使用 cv::minAreaRect(contour) 获取轮廓的最小外接矩形 textrect。
	cv::RotatedRect textrect = cv::minAreaRect(contour);
	// 使用 cv::boxPoints(textrect, boxPoints2f) 将最小外接矩形的四个顶点保存到 boxPoints2f 中。
	cv::Mat boxPoints2f;
	cv::boxPoints(textrect, boxPoints2f);
	// 将boxPoints2f的数据指针转换为float*类型指针，并将其赋值给变量p1，
	// 在 OpenCV 中，boxPoints2f 是一个 cv::Mat 对象，表示包含四个浮点数坐标的矩阵。boxPoints2f.data 返回矩阵数据的指针（即第一个像素的地址），默认情况下是 uchar* 类型的指针。
	// 由于数据类型不同，我们需要将 uchar* 指针转换为 float* 指针，以便后续处理。这可以通过类型转换运算符(float*) 来实现。最终，变量 p1 将指向 boxPoints2f 数据的内存地址，并且被解释为 float 类型的指针。
	// 通过这个操作，我们可以使用 p1 来访问和处理 boxPoints2f 矩阵的浮点数数据。在代码的后续部分，我们可以看到 p1 被用于提取矩形的顶点坐标。
	float* p1 = (float*)boxPoints2f.data;
	// 将 p1 中的顶点坐标转换为 cv::Point2f 类型，并保存到 tmpvev 中。
	std::vector<cv::Point2f> tmpvev;
	for (int i = 0; i < 4; ++i, p1 += 2) {
		tmpvev.push_back(cv::Point2f(p1[0], p1[1]));
	}
	// 对 tmpvev 中的顶点按照 cvpointcompare 函数进行排序，以确保矩形的顶点按照顺时针或逆时针的顺序排列。
	std::sort(tmpvev.begin(), tmpvev.end(), cvpointcompare);
}
*/

void Text_Detect::get_mini_boxes(std::vector<cv::Point>& contour, cv::Point2f minibox[4], float& minedgesize, float& perimeter, float& area) {
	// 使用 cv::minAreaRect(contour) 获取轮廓的最小外接矩形 textrect。
	cv::RotatedRect textrect = cv::minAreaRect(contour);
	// 使用 textrect.points() 将最小外接矩形的四个顶点坐标存储在 boxPoints2f 数组中。
	cv::Point2f boxPoints2f[4];
	textrect.points(boxPoints2f);

	// 比较函数通过比较点的 x 坐标来确定点的顺序，即按照从左到右的顺序排序。
	// [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; } 是一个 lambda 表达式，用于定义一个自定义的比较函数。
	// 在这个 lambda 表达式中，输入参数是两个 cv::Point2f 类型的引用 a 和 b。比较函数通过比较这两个点的 x 坐标来确定它们的顺序。
	// lambda 表达式使用方括号[] 来指定捕获列表。在这个例子中，捕获列表为空，表示没有捕获任何外部变量。
	// lambda 表达式的返回值是一个布尔值，表示两个点的比较结果。如果第一个点 a 的 x 坐标小于第二个点 b 的 x 坐标，则返回 true，否则返回 false。
	// 这个 lambda 表达式在代码中被用作 std::sort() 函数的第三个参数，即排序的比较函数。它告诉 std::sort() 函数按照点的 x 坐标来进行升序排序。
	std::sort(std::begin(boxPoints2f), std::end(boxPoints2f), [](const cv::Point2f& a, const cv::Point2f& b) {
		return a.x < b.x; });

	// 根据排序后的顺序，选择最小外接矩形的四个顶点，并将它们存储在输出参数 minibox 数组中。
	int index1 = (boxPoints2f[1].y > boxPoints2f[0].y) ? 0 : 1;
	int index4 = (index1 == 0) ? 1 : 0;
	int index2 = (boxPoints2f[3].y > boxPoints2f[2].y) ? 2 : 3;
	int index3 = (index2 == 2) ? 3 : 2;
	minibox[0] = boxPoints2f[index1];
	minibox[1] = boxPoints2f[index2];
	minibox[2] = boxPoints2f[index3];
	minibox[3] = boxPoints2f[index4];

	// 计算最小外接矩形的最短边尺寸、周长和面积，并分别存储在输出参数 minedgesize、perimeter 和 area 中。
	minedgesize = (std::min)(textrect.size.width, textrect.size.height);
	perimeter = 2.f * (textrect.size.width + textrect.size.height);
	area = textrect.size.width * textrect.size.height;
}


float Text_Detect::box_score_fast(cv::Mat& probability_map, std::vector<cv::Point>& contour) {
	std::vector<cv::Point> box = contour;
	int width = probability_map.cols;
	int height = probability_map.rows;
	int xmax = -1, xmin = 1000000, ymax = -1, ymin = 1000000;
	// 使用循环遍历轮廓的每个点，更新 xmax、xmin、ymax、ymin 的值，使其分别保持轮廓在 x 和 y 方向的最大和最小坐标。
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
	// 将 xmax、xmin、ymax、ymin 的值限定在合理的图片尺寸范围内。
	xmax = (std::min)((std::max)(xmax, 0), width - 1);
	xmin = (std::max)((std::min)(xmin, width - 1), 0);
	ymax = (std::min)((std::max)(ymax, 0), height - 1);
	ymin = (std::max)((std::min)(ymin, height - 1), 0);

	// 使用循环遍历轮廓的每个点，将每个点的坐标减去 xmin 和 ymin，相当于将最小外接矩形的左上角移动到原点。
	for (int i = 0; i < box.size(); ++i)
	{
		box[i].x = box[i].x - xmin;
		box[i].y = box[i].y - ymin;
		//std::cout<<box[i]<<std::endl;
	}

	// 创建了一个 maskmat 矩阵，它是一个 CV_8UC1 类型的单通道图像，其大小为 (ymax - ymin + 1) 行，(xmax - xmin + 1) 列。初始时所有像素被设置为黑色（值为0）。
	cv::Mat maskmat(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1, cv::Scalar(0, 0, 0));
	// 使用 box 的多边形轮廓，在 maskmat 上进行多边形填充，填充的区域会被设置为白色（值为1），通过 cv::Scalar(1, 1, 1) 指定填充颜色。
	cv::fillPoly(maskmat, std::vector<std::vector<cv::Point>>{ box }, cv::Scalar(1, 1, 1), 1);
	// 计算了在 probability_map 概率图像的指定区域内，掩码为 maskmat 的像素的平均值。
	// 首先使用 cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1) 创建了感兴趣区域（ROI），然后通过 clone() 复制了感兴趣区域的副本，最后使用 cv::mean() 计算了掩码内像素的平均值。返回结果的 [0] 索引表示提取出的平均值。
	return cv::mean(probability_map(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)), maskmat)[0];
}


void Text_Detect::unclip(cv::Point2f minibox[4], float& perimeter, float& area, float& unclip_ratio, std::vector<cv::Point>& expand_contour) {
	// 这个距离表示需要将边界向外扩展的距离
	float distance = area * unclip_ratio / perimeter;
	ClipperLib::Path poly;
	ClipperLib::ClipperOffset offset;

	// 创建一个ClipperLib::Path对象poly，并将最小外接矩形的四个顶点坐标转换为整型，并添加到poly中。
	for (int i = 0; i < 4; ++i) {
		poly.push_back(ClipperLib::IntPoint(minibox[i].x, minibox[i].y));
	}
	// 建一个ClipperLib::ClipperOffset对象offset，并使用AddPath函数将poly添加到offset中。
	offset.AddPath(poly, ClipperLib::jtRound, ClipperLib::etClosedPolygon);

	// 创建一个ClipperLib::Paths对象polys，并将poly添加到polys中。		
	ClipperLib::Paths polys;
	polys.push_back(poly);
	// 调用Execute函数对polys进行unclip操作，并将解除剪切后的结果存储在polys中，扩展距离为distance。
	offset.Execute(polys, distance);

	// 遍历polys中的每个路径，将其中的点坐标转换为cv::Point格式，并添加到expand_contour中。
	// 返回处理后的扩展轮廓expand_contour
	for (int i = 0; i < polys.size(); i++)
	{
		for (int j = 0; j < polys[polys.size() - 1].size(); j++)
		{
			expand_contour.emplace_back(polys[i][j].X, polys[i][j].Y);    //将polys转化为contour的格式
		}
	}
}


cv::Point2f transform_preds(const cv::Mat& warpMat, const cv::Point2f& pt) {
	// warpMat 仿射变换矩阵，pt待变换的坐标
	// 创建一个大小为(1, 3)、数据类型为64位浮点型的pt_mid矩阵。
	cv::Mat pt_mid(1, 3, CV_64FC1);
	// 将输入的点坐标的x、y分量分别赋值给pt_mid矩阵的第一行数据，并将最后一个元素设置为1.0，表示齐次坐标中的1.
	pt_mid.at<double>(0, 0) = pt.x;
	pt_mid.at<double>(0, 1) = pt.y;
	pt_mid.at<double>(0, 2) = 1.0;
	// 创建一个new_pt矩阵，通过将pt_mid矩阵与仿射变换矩阵warpMat相乘得到。
	cv::Mat new_pt = pt_mid * warpMat;
	// 提取new_pt矩阵中的第一个元素和第二个元素作为x和y坐标，创建一个cv::Point2f类型的对象，表示变换后的点坐标。


}


std::vector<DetTextBox> Text_Detect::detect(const cv::Mat& image, const int& target_size) {
	double time_preprocess = (double)cv::getTickCount();

	std::vector<DetTextBox> textboxes;
	int width, height, resize_w, resize_h;
	width = resize_w = image.cols;
	height = resize_h = image.rows;
	cv::Point2f center(width / 2.0, height / 2.0);
	int img_maxsize = width > height ? width : height; // 如果width>height，取width
	int img_minsize = width < height ? width : height; // 如果width<height，取width
	int square_size = 640;

	cv::Mat dst;
	bool use_padding_resize = false;
	if (1.0 * height / width >= 5 || 1.0 * width / height >= 5 && img_minsize <= 32) {
		// padding_resize 预处理，主要是对极端宽比或者是size小于32，padding仿射变换，padding之后的尺寸是640*640
		use_padding_resize = true;
		std::cout << "采用的是padding_resize" << std::endl;
		cv::Mat affineMat = get_affine_transform(center, img_maxsize, square_size, 0); // padding_size的尺寸是640*640的放图
		cv::warpAffine(image, dst, affineMat, cv::Size(square_size, square_size));//根据仿射变换矩阵，将原图映射到640x640的方形图片上
		cv::imwrite("dst.jpg", dst);
	}
	else {//等比例放缩
		use_padding_resize = false;
		std::cout << "采用的是等比例resize" << std::endl;
		ResizeImgType2(image, dst, target_size, resize_h, resize_w);//该种方式target_size默认是960,dst传的是地址
		std::cout << "resize_h, resize_w is " << resize_h << "," << resize_w << std::endl;
	}
	double time_end_preprocess = ((double)cv::getTickCount() - time_preprocess) / cv::getTickFrequency();
	std::cout << "文本预处理花费时间：" << time_end_preprocess << "秒" << std::endl;

	double time_start_detect = (double)cv::getTickCount();
	// 将 dst 转换为 ncnn::Mat 类型的 dbnet_input。dst 是一个 OpenCV 的 cv::Mat 对象，其中包含了图像数据。
	// from_pixels 函数用于从原始图像数据创建 ncnn::Mat 对象。ncnn::Mat::PIXEL_BGR2RGB 表示图像的通道排列是 BGR，而不是 RGB。
	ncnn::Mat dbnet_input = ncnn::Mat::from_pixels(dst.data, ncnn::Mat::PIXEL_BGR2RGB, dst.cols, dst.rows);
#if PRINT_FLAG
	printf("dbnet_input: c=%d,resize_w=%d,resize_h=%d\n", dbnet_input.c, dbnet_input.w, dbnet_input.h);
#endif
	// 使用 substract_mean_normalize 函数对 dbnet_input 进行均值减法和归一化处理。mean_vals_dbnet 和 norm_vals_dbnet 是预定义的均值和归一化参数，用于对图像数据进行预处理。
	dbnet_input.substract_mean_normalize(mean_vals_dbnet, norm_vals_dbnet);
	// 创建 Extractor 对象 dbnet_ex，用于从模型中提取特征。Model 是预定义的模型对象，表示已加载的模型。
	ncnn::Extractor dbnet_ex = Model.create_extractor();
	// 设置 dbnet_ex 的线程数量为 num_thread，以优化并行计算性能。
	dbnet_ex.set_num_threads(num_thread);
	// 使用 input 函数将 dbnet_input 设置为 dbnet_ex 的输入。input 函数用于指定输入层的名称和对应的输入数据。
	dbnet_ex.input("input", dbnet_input);
	// 创建 dbnet_out 作为输出结果的容器。
	ncnn::Mat dbnet_out;
	// 使用 extract 函数从 dbnet_ex 中提取输出结果，并将结果保存在 dbnet_out 中。"out" 是指定的输出层的名称。
	dbnet_ex.extract("out", dbnet_out);
#if PRINT_FLAG
	printf("dbnet_out: c=%d,resize_w=%d,resize_h=%d\n", dbnet_out.c, dbnet_out.w, dbnet_out.h);
#endif
	double time_end_detect = ((double)cv::getTickCount() - time_start_detect) / cv::getTickFrequency();
	std::cout << "文本检测推断花费时间：" << time_end_detect << "秒" << std::endl;

	double time_start_postprocess = (double)cv::getTickCount();
	// 创建了一个名为 probability_map 的 cv::Mat 对象，用来存储概率图的数据。
	// (const float*)dbnet_out.channel(0) 是将概率图中第0个通道的数据转换为 const float* 类型的指针。通过 (void*) 强制类型转换为 void* 类型的指针，因为 cv::Mat 构造函数需要接受一个 void* 类型的指针作为数据。
	cv::Mat probability_map(dbnet_out.h, dbnet_out.w, CV_32FC1, (void*)(const float*)dbnet_out.channel(0));
	// 创建一个新的名为 binary_map 的 Mat 对象，并使用阈值 thresh_ 将 probability_map 二值化。将大于阈值的像素设为白色（255），小于等于阈值的像素设为黑色（0）。这样得到的 norfmapmat 是一个CV_8UC1类型的二值图像（8位无符号整型，单通道）。
	cv::Mat binary_map;
	binary_map = probability_map > thresh_; // 
#if VISUAL_FLAG
	cv::imwrite("thersh.jpg", binary_map);
#endif

	textboxes.clear();
	std::vector<std::vector<cv::Point>> contours; // 二维向量
	// 使用OpenCV的 findContours 函数从 binary_map 中找到所有的轮廓，并存储在 contours 中。
	// 参数 RETR_LIST 表示提取所有的轮廓，参数 CHAIN_APPROX_SIMPLE 表示压缩水平、垂直和对角线方向的像素点，仅保留端点。
	// contours 是一个容器，其中存储了检测到的轮廓信息。它是一个 std::vector<std::vector<cv::Point>> 类型的变量，层级结构如下：
	// 外层 std::vector：存储所有的轮廓。内层 std::vector<cv::Point>：每个内层 std::vector 存储一个轮廓的点集合。
	// 每个轮廓都由一系列的 cv::Point 对象组成，表示轮廓上的点的坐标。可以通过使用索引访问外层和内层容器来获取具体的轮廓和轮廓上的点。
	// 例如，contours[i] 表示第 i 个轮廓，contours[i][j] 表示第 i 个轮廓上的第 j 个点。
	// 每个 cv::Point 对象包含两个属性：x：点的 x 坐标。y：点的 y 坐标。
	// 通过遍历 contours 容器中的元素，可以获取每个轮廓的所有点的坐标信息。
	cv::findContours(binary_map, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
#if PRINT_FLAG    
	std::cout << "轮廓的个数是" << contours.size() << std::endl;
#endif

	// findcontours之后的四点要进行清洗
	for (int i = 0; i < contours.size(); ++i) {
		// 调用函数 get_mini_boxes()，传入当前轮廓和其他参数，计算得到最小外接矩形的四个顶点坐标、最小边长、周长和面积。
		cv::Point2f minibox[4];
		float minedgesize, perimeter, area;
		get_mini_boxes(contours[i], minibox, minedgesize, perimeter, area);

		// 如果最小边长小于 min_size_，则跳过当前轮廓的处理，继续下一轮循环。
		if (minedgesize < min_size_)
			continue;

		// 传入概率图和当前轮廓，计算得到当前轮廓的分数。
		float score = box_score_fast(probability_map, contours[i]);

		// 如果分数小于 box_thresh_，则跳过当前轮廓的处理，继续下一轮循环。
		if (score < box_thresh_)
			continue;
#if PRINT_FLAG     
		std::cout << "第" << i << "轮廓的分数是" << score << std::endl;
		std::cout << "第" << i << "轮廓得到的minibox:" << std::endl;
		for (int i = 0; i < 4; i++)
			std::cout << minibox[i] << std::endl;
#endif

		std::vector<cv::Point> expand_contour;
		unclip(minibox, perimeter, area, unclip_ratio_, expand_contour);
#if PRINT_FLAG        
		std::cout << "第" << i << "轮廓得到的expand_contour:" << std::endl;
		for (int i = 0; i < expand_contour.size(); i++)
			std::cout << expand_contour[i] << std::endl;
#endif

		// unclip之后再重新计算最小外接矩形
		get_mini_boxes(expand_contour, minibox, minedgesize, perimeter, area);
#if PRINT_FLAG                
		std::cout << "第" << i << "轮廓第二次得到的minibox:" << std::endl;
		for (int i = 0; i < 4; i++)
			std::cout << minibox[i] << std::endl;
#endif    

		// 如果最小边长小于设定的阈值(min_size_ + 2)，则跳过该轮廓，不进行后续处理。
		if (minedgesize < min_size_ + 2)
			continue;

		// 要映射回原图坐标，通常在第二步还需要crnn识别
		if (use_padding_resize) {
			// ，通过调用get_affine_transform函数获取仿射变换矩阵warpMat，该矩阵用于将最小外接矩形进行变换。其中，center表示中心点坐标，img_maxsize表示图像的最大尺寸，square_size表示目标尺寸。
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

