#include "yolov5_libtorch.h"

Yolov5Libtorch::Yolov5Libtorch(std::string ptFile, bool isCuda, bool isHalf, int height, int width, float confThres, float iouThres) {
	model = torch::jit::load(ptFile);
	if (isCuda) {
		model.to(torch::kCUDA);
	}
	if (isHalf) {
		model.to(torch::kHalf);
	}
	this->height = height;
	this->width = width;
	this->isCuda = isCuda;
	this->iouThres = iouThres;
	this->confThres = confThres;
	this->isHalf = isHalf;
	model.eval();
	unsigned seed = time(0);
	std::srand(seed); // 根据当前时间种子初始化随机数发生器。
}


std::vector<torch::Tensor> Yolov5Libtorch::non_max_suppression(torch::Tensor prediction, float confThres, float iouThres) {
	/*
	select() 是用来获取张量某个维度上的一个数据切片的函数。select(dim, index) 函数会返回一个新的张量，该张量是原始张量 input 在指定维度 dim 上索引为 index 的切片。
	*/
	torch::Tensor xc = prediction.select(2, 4) > confThres;
	std::cout << "score confthres tensor shape:";
	showShape(xc);

	// 定义两个常量，分别为最大宽高和最大 NMS 候选框数量。
	int maxWh = 4096;
	int maxNms = 30000;
	// 初始化一个空的 std::vector<torch::Tensor> 类型的输出向量 output，大小为 batch_size。并将每个子张量都初始化为一个大小为 (0, 6) 的全零张量。
	std::vector<torch::Tensor> output;
	for (int i = 0; i < prediction.size(0); i++) {
		output.push_back(torch::zeros({ 0,6 }));
	}
	for (int i = 0; i < prediction.size(0); i++) {
		// 对于每一张输入图片的预测结果，取出满足目标置信度阈值要求的预测框，并保存到张量变量 x 中。
		torch::Tensor x = prediction[i];
		x = x.index_select(0, torch::nonzero(xc[i]).select(1, 0));
		if (x.size(0) == 0)continue;//如果 x 中没有任何符合要求的预测框，则直接跳过下面的代码继续处理下一张输入图片。

		x.slice(1, 5, x.size(1)).mul_(x.slice(1, 4, 5)); //就是选取x张量在第1个维度上，从索引5开始到末尾的那一部分，并在同一个位置上，用第二个张量 x.slice(1, 4, 5) 的对应位置元素做乘法，然后更新原来的 x 张量。计算每个预测框的面积（通过将其宽和高相乘）并将其乘以置信度，得到预测框的“得分”。
		torch::Tensor box = xywh2xyxy(x.slice(1, 0, 4));
		/*
		首先，第一行代码使用了 PyTorch 中的 max() 函数来寻找每个预测框中概率最高的类别。具体地，它选出了预测张量 x 沿着第一个维度按照顺序从第 5 个位置开始到最后一个位置的所有数据（假设 x 的维度为 [B, num_anchors, (num_classes+5), H, W]，其中 B 为 batch size）。这些数据表示了每个预测框中所有类别的概率（注意这里也包含了背景类别）。然后，在这些数据中沿着第二个维度选出最大值，并返回一个大小为 [B, num_anchors, 1, H, W] 的张量（这个张量包含了每个预测框中概率最高的类别的索引）和一个大小为 [B, num_anchors, 1, H, W] 的张量（这个张量包含了每个预测框中概率最高的类别的概率值）。
		*/
		std::tuple<torch::Tensor, torch::Tensor> max_tuple = torch::max(x.slice(1, 5, x.size(1)), 1, true);
		/*
	 将三个张量 box、std::get<0>(max_tuple) 和 std::get<1>(max_tuple) 沿着第二个维度进行拼接，box 张量包含了每个预测框的位置信息，std::get<0>(max_tuple) 张量是概率最大的类别的索引（1 维），std::get<1>(max_tuple) 张量是概率最大的类别的概率值（1 维），两个张量共同组成了预测框的分类信息（共 2 维）。
		*/
		x = torch::cat({ box,std::get<0>(max_tuple),std::get<1>(max_tuple) }, 1);
		// 过滤掉概率低于阈值的预测框，得到一个新的张量。
		x = x.index_select(0, torch::nonzero(std::get<0>(max_tuple) > confThres).select(1, 0));

		std::cout << "prediction中:";
		showShape(x);

		// 没有框就continue，如果框超过了最大值则只保留钱maxnms个
		int n = x.size(0);
		if (n == 0) {
			continue;
		}
		else if (n > maxNms) {
			x = x.index_select(0, x.select(1, 4).argsort(0, true).slice(0, 0, maxNms));
		}
		/*
		这两行代码是将预测框从相对位置（中心-宽度-高度）转换为图像绝对坐标。
		首先，x.slice(1, 5, 6) 表示从 x 中选取维度为 1（即第二个维度，也就是与预测框有关的维度），下标从 5 开始、结束于 6 的所有元素，这些元素表示预测框的高度信息。然后将这些高度值乘以一个最大的宽度和高度比例因子 maxWh，得到在当前特征图下的实际高度。
		接着，x.slice(1, 0, 4) 表示从 x 中选取维度为 1，下标从 0 开始、结束于 4 的所有元素，这些元素表示预测框的位置信息（即左上角和右下角的坐标）。然后将这些位置信息加上刚刚计算得到的中心点偏移量,得到该预测框在原图中的绝对坐标。
		*/
		torch::Tensor c = x.slice(1, 5, 6) * maxWh;
		torch::Tensor boxes = x.slice(1, 0, 4) + c, scores = x.select(1, 4);
		torch::Tensor ix = nms(boxes, scores, iouThres).to(x.device());
		output[i] = x.index_select(0, ix).cpu(); // 使用得到的索引集合从预测张量中选取被保留的预测框，并将其存入输出张量中。
	}
	return output;
}

cv::Mat Yolov5Libtorch::img2RGB(cv::Mat img) {
	int imgC = img.channels();
	if (imgC == 1) {
		cv::cvtColor(img, img, cv::COLOR_GRAY2RGB);
	}
	else {
		cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
	}
	return img;
}

torch::Tensor Yolov5Libtorch::img2Tensor(cv::Mat img) {
	/*
	首先使用 torch::from_blob() 函数将输入数据转换成张量。其中，img.data 是输入图像的指针，(height, width, 3) 是输入图像的尺寸，torch::kByte 表示转换后的张量数据类型为 uint8 （即 unsigned char）。
	接着使用 permute() 函数交换张量维度，将通道维度（C）放到第一维，将宽度维度（W）放到第二维，将高度维度（H）放到第三维。这是因为在 PyTorch 中，通道维度是第一维，相当于 NumPy 中的最后一维。
	调用 toType() 函数将张量的数据类型转换为 torch::kFloat，即单精度浮点型。这是因为在深度学习中，大多数模型的输入和输出都是浮点型。
	使用 div() 函数将张量的数值范围从 (0,255) 归一化为 (0,1)。这是为了将输入数据的数值范围映射到模型训练时使用的区间内，以达到更好的精度和效果。
	最后使用 unsqueeze() 函数在第一维插入一个新的维度，将张量的尺寸从 (3, H, W) 扩展为 (1, 3, H, W)。这是因为在 PyTorch 中，模型的输入一般都是一个 batch 的数据，需要在第一维插入一个 batch 维度。
	*/
	torch::Tensor data = torch::from_blob(img.data, { (int)height,(int)width,3 }, torch::kByte);
	data = data.permute({ 2,0,1 });
	data = data.toType(torch::kFloat);
	data = data.div(255);
	data = data.unsqueeze(0);
	return data;
}


torch::Tensor Yolov5Libtorch::xywh2xyxy(torch::Tensor x) {
	torch::Tensor y = x.clone();
	y.select(1, 0) = x.select(1, 0) - x.select(1, 2) / 2;
	y.select(1, 1) = x.select(1, 1) - x.select(1, 3) / 2;
	y.select(1, 2) = x.select(1, 0) + x.select(1, 2) / 2;
	y.select(1, 3) = x.select(1, 1) + x.select(1, 3) / 2;
	return y;
}


torch::Tensor Yolov5Libtorch::nms(torch::Tensor bboxes, torch::Tensor scores, float thresh) {
	std::cout << "nms bbox shape:";
	showShape(bboxes); // 13x4
	std::cout << "nms scores shape:";
	showShape(scores); // 13

	auto x1 = bboxes.select(1, 0); // 取出检测框的左上角x坐标
	auto y1 = bboxes.select(1, 1); // 取出检测框的左上角y坐标
	auto x2 = bboxes.select(1, 2); // 取出检测框的右下角x坐标
	auto y2 = bboxes.select(1, 3);
	auto areas = (x2 - x1) * (y2 - y1); // 计算面积
	auto tuple_sorted = scores.sort(0, true); // 将 scores 张量按照从大到小的顺序进行排序，并返回排序后的结果和对应的索引。具体而言，参数 0 表示按第 0 维度排序（即按行排序），true 表示降序排列。
	auto order = std::get<1>(tuple_sorted); // 从 tuple_sorted 中取出排好序的索引值，即 scores 张量中每个元素对应的原始位置。第一个元素是排序后的结果张量，第二个元素是对应的索引张量。因此 std::get<1>(tuple_sorted) 返回的就是排序后的索引张量。

	std::vector<int> keep; //保存索引向量
	while (order.numel() > 0) {
		// 如果 order 张量中只剩下 1 个元素，那么直接将该元素的索引加入 keep 向量
		if (order.numel() == 1) {
			auto i = order.item();
			keep.push_back(i.toInt());
			break;
		}
		else {
			// 如果 order 张量中还剩下多个元素，则将其中排名最高的元素的索引加入 keep 向量，即 order[0] 所对应的索引值,这里调用了 PyTorch 的 item() 函数，将张量中唯一的标量值取出来，并将类型转换为整型。然后使用 push_back() 方法将其添加到 keep 向量末尾。
			auto i = order[0].item();
			keep.push_back(i.toInt());
		}

		// order_mask 张量是一个由 order 张量去掉第一个元素后形成的新张量，它保存了除最高分检测框以外的所有检测框的索引值。
		auto order_mask = order.narrow(0, 1, order.size(-1) - 1);
		// 用 index() 方法按照 order_mask 中的索引值从 x1、y1、x2、y2 四个张量中取出对应的值。其中，clamp() 方法用来将取出的值限制在指定的范围内。这里需要注意的是，clamp() 方法的第一个参数可以是一个张量，表示将对应的元素分别限制在相应的范围内。
		auto xx1 = x1.index({ order_mask }).clamp(x1[keep.back()].item().toFloat(), 1e10);
		auto yy1 = y1.index({ order_mask }).clamp(y1[keep.back()].item().toFloat(), 1e10);
		auto xx2 = x2.index({ order_mask }).clamp(0, x2[keep.back()].item().toFloat());
		auto yy2 = y2.index({ order_mask }).clamp(0, y2[keep.back()].item().toFloat());

		// 首先，计算相交面积 inter，即两个检测框的重叠部分的面积。具体而言，(xx2 - xx1) 和 (yy2 - yy1) 表示两个检测框在水平和垂直方向上的重叠长度，二者相乘即为相交面积。这里同样使用了 clamp() 方法将相交面积限制在 0 和 1e10 之间。接着，计算 IOU 值。其中，area1 表示第一个检测框的面积，即 areas[keep.back()]；area2 表示其它检测框的面积之和，即 areas.index({ order.narrow(0,1,order.size(-1) - 1) })。这里还用到了 narrow() 方法，表示将张量沿着指定维度进行裁剪，保留指定范围内的元素。
		auto inter = (xx2 - xx1).clamp(0, 1e10) * (yy2 - yy1).clamp(0, 1e10);
		auto iou = inter / (areas[keep.back()] + areas.index({ order.narrow(0,1,order.size(-1) - 1) }) - inter);

		// 这段代码判断 IOU 值是否小于阈值，并把符合条件的索引保存到向量 idx 中。如果 idx 向量为空，说明已经没有符合条件的检测框，跳出循环。
		auto idx = (iou <= thresh).nonzero().squeeze();
		if (idx.numel() == 0) {
			break;
		}
		// 如果 idx 向量非空，则将 order 张量中对应的元素删除，并将新的张量赋值给 order 变量。这里需要注意的是，由于 idx 向量中保存的是除最高分检测框以外的索引，因此在进行索引操作时需要先将这些索引加 1，才能得到正确的结果。
		order = order.index({ idx + 1 });
	}
	std::cout << "nms keep shape:";
	showShape(torch::tensor(keep)); // 
	return torch::tensor(keep);
}

std::vector<torch::Tensor> Yolov5Libtorch::sizeOriginal(std::vector<torch::Tensor> result, std::vector<ImageResizeData> imgRDs) {
	//std::cout << "sizeOriginal result shape";
	//showShape(result);
	std::vector<torch::Tensor> resultOrg;
	// 遍历检测结果向量 result 中的每个元素
	for (int i = 0; i < result.size(); i++) {
		torch::Tensor data = result[i];
		ImageResizeData imgRD = imgRDs[i];
		// 对于每个检测框
		for (int j = 0; j < data.size(0); j++) {
			torch::Tensor tensor = data.select(0, j); // 从张量中取出第 j 个检测框
			// 判断图片是否被缩放到了宽度上限。因为缩放后的图片宽高比可能与原图不同，所以需要根据图片是否被缩放来分别调整检测框的坐标。
			if (imgRD.isW()) {
				tensor[1] -= imgRD.border; // 宽度border补全
				tensor[3] -= imgRD.border;

				tensor[0] *= (float)imgRD.w / (float)imgRD.width;
				tensor[2] *= (float)imgRD.w / (float)imgRD.width;
				tensor[1] *= (float)imgRD.h / (float)(imgRD.height - 2 * imgRD.border);
				tensor[3] *= (float)imgRD.h / (float)(imgRD.height - 2 * imgRD.border);
			}
			else {
				tensor[0] -= imgRD.border;
				tensor[2] -= imgRD.border;
				tensor[1] *= (float)imgRD.h / (float)imgRD.height;
				tensor[3] *= (float)imgRD.h / (float)imgRD.height;
				tensor[0] *= (float)imgRD.w / (float)(imgRD.width - 2 * imgRD.border);
				tensor[2] *= (float)imgRD.w / (float)(imgRD.width - 2 * imgRD.border);
			}
			// 加了黑边之后预测结果可能在黑边上，就会造成结果为负数
			for (int k = 0; k < 4; k++)
			{
				if (tensor[k].item().toFloat() < 0)
				{
					tensor[k] = 0;
				}
			}
		}
		resultOrg.push_back(data);
	}
	return resultOrg;
}


std::vector<torch::Tensor> Yolov5Libtorch::prediction(torch::Tensor data) {
	if (!data.is_cuda() && this->isCuda) {
		data = data.cuda();
	}
	if (data.is_cuda() && !this->isCuda) {
		data = data.cpu();
	}
	if (this->isHalf) {
		data = data.to(torch::kHalf);
	}
	/*
	这段代码是对输入数据进行前向传播的操作。其中，model 是一个 PyTorch 模型，data 是一个输入数据的张量。通过调用 model.forward({ data }) 方法，将输入数据传递给模型进行前向传播，并获得一个 at::IValue 类型的返回值（即模型输出）。返回值是一个元组（Tuple）类型，包含了模型的所有输出。由于这里只有一个输出，因此通过 toTuple()->elements()[0] 获取元组中的第一个元素，然后再通过 toTensor() 将其转换为一个张量对象（torch::Tensor）。
	*/
	torch::Tensor pred = model.forward({ data }).toTuple()->elements()[0].toTensor();

	// 打印pred的shape
	std::cout << "pred tensor shape:";
	showShape(pred);

	return non_max_suppression(pred, confThres, iouThres);
}

std::vector<torch::Tensor> Yolov5Libtorch::prediction(cv::Mat img) {
	ImageResizeData imgRD = resize(img);
	cv::Mat reImg = img2RGB(imgRD.img);
	torch::Tensor data = img2Tensor(reImg);
	std::vector<torch::Tensor> result = prediction(data);
	std::vector<ImageResizeData> imgRDs;
	imgRDs.push_back(imgRD);
	//std::vector<torch::Tensor> output;
	return sizeOriginal(result, imgRDs);
}


void Yolov5Libtorch::showShape(torch::Tensor data) {
	//std::cout << "pred tensor shape:";
	auto shape = data.sizes();
	for (size_t i = 0; i < shape.size(); i++) {
		std::cout << shape[i];
		if (i < shape.size() - 1) {
			std::cout << "x";
		}
	}
	std::cout << std::endl;
}

ImageResizeData Yolov5Libtorch::resize(cv::Mat img, int height, int width) {
	ImageResizeData imgResizeData;
	int w = img.cols, h = img.rows;

	imgResizeData.h = h; // 原始图片的高宽
	imgResizeData.w = w;
	imgResizeData.height = height; // 转换过后图片的高宽
	imgResizeData.width = width;
	/*
	这段代码的作用是根据 isW 的值来进行图像的缩放。具体来说，首先根据图像的目标宽高比 w/h 和原始宽高比 width/height 的大小关系来确定是按照宽度还是高度进行缩放，即如果宽度比例更大则按照宽度进行缩放，否则按照高度进行缩放。然后利用 OpenCV 库中的 resize 函数来对图像进行缩放。如果按照宽度进行缩放，则将图像的宽度设置为目标宽度 width，高度则按比例计算得到；如果按照高度进行缩放，则将图像的高度设置为目标高度 height，宽度则按比例计算得到。最终得到的缩放后的图像会覆盖掉原始图像 img。
	*/
	bool isW = (float)w / (float)h > (float)width / (float)height; // 原始宽高比比转换宽高比要大，返回true，说明原始图很宽
	cv::resize(img, img, cv::Size(
		isW ? width : (int)((float)height / (float)h * w),
		isW ? (int)((float)width / (float)w * h) : height));

	w = img.cols, h = img.rows;
	if (isW) {
		imgResizeData.border = (height - h) / 2;
		/*
		这段代码的作用是对图像进行边框填充。具体来说，利用 OpenCV 库中的 copyMakeBorder 函数来在图像的上下方添加边框，使得图像的高度等于目标高度 height，同时左右方向上不进行扩展。这里，参数 (height - h) / 2 和 height - h - (height - h) / 2 分别表示图像上方和下方需要填充的像素个数，它们的计算公式为 (height - h) / 2 和 height - h - (height - h) / 2，其中 height 是目标高度，h 是图像原始高度。最后一个参数 BORDER_CONSTANT 表示采用常量填充边框，这里填充的常量默认为黑色。经过这样的边框填充之后，图像的尺寸变成了 (height, width)，并且图像居中显示。*/
		cv::copyMakeBorder(img, img, (height - h) / 2, height - h - (height - h) / 2, 0, 0, cv::BORDER_CONSTANT);
	}
	else {
		imgResizeData.border = (width - w) / 2;
		cv::copyMakeBorder(img, img, 0, 0, (width - w) / 2, width - w - (width - w) / 2, cv::BORDER_CONSTANT);
	}
	imgResizeData.img = img;
	return imgResizeData;
}

ImageResizeData Yolov5Libtorch::resize(cv::Mat img) {
	return Yolov5Libtorch::resize(img, height, width);
}

bool ImageResizeData::isW() {
	return (float)w / (float)h > (float)width / (float)height;
}

bool ImageResizeData::isH() {
	return (float)h / (float)w > (float)height / (float)width;
}



cv::Mat Yolov5Libtorch::drawRectangle(cv::Mat img, torch::Tensor rectangle, std::map<int, cv::Scalar> colors, std::map<int, std::string> labels, int thickness) {
	std::cout << "rectangle tensor shape:";
	showShape(rectangle);

	std::map<int, cv::Scalar>::iterator it;
	std::map<int, std::string>::iterator labelIt;
	for (int i = 0; i < rectangle.size(0); i++) {
		int clazz = rectangle[i][5].item().toInt();

		// 选颜色
		it = colors.find(clazz);
		cv::Scalar color = NULL;
		if (it == colors.end()) {
			it = mainColors.find(clazz);//没找到，随机赋值
			if (it == mainColors.end()) {
				color = cv::Scalar(std::rand() % 256, std::rand() % 256, std::rand() % 256);
				mainColors.insert(std::pair<int, cv::Scalar>(clazz, color));
			}
			else {
				color = it->second;
			}
		}
		else {
			color = it->second;
		}

		cv::rectangle(img, cv::Point(rectangle[i][0].item().toInt(), rectangle[i][1].item().toInt()), cv::Point(rectangle[i][2].item().toInt(), rectangle[i][3].item().toInt()), color, thickness);
		labelIt = labels.find(clazz);

		std::ostringstream oss;
		if (labelIt != labels.end()) {
			oss << labelIt->second << "";
		}
		oss << rectangle[i][4].item().toFloat();
		std::string label = oss.str();
		cv::putText(img, label, cv::Point(rectangle[i][0].item().toInt(), rectangle[i][1].item().toInt()), cv::FONT_HERSHEY_PLAIN, 1, color, thickness);
	}
	return img;
}


cv::Mat Yolov5Libtorch::drawRectangle(cv::Mat img, torch::Tensor rectangle, std::map<int, std::string> labels, int thickness) {
	std::map<int, cv::Scalar> colors;
	return drawRectangle(img, rectangle, colors, labels, thickness);
}
