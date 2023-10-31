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
	std::srand(seed); // ���ݵ�ǰʱ�����ӳ�ʼ���������������
}


std::vector<torch::Tensor> Yolov5Libtorch::non_max_suppression(torch::Tensor prediction, float confThres, float iouThres) {
	/*
	select() ��������ȡ����ĳ��ά���ϵ�һ��������Ƭ�ĺ�����select(dim, index) �����᷵��һ���µ���������������ԭʼ���� input ��ָ��ά�� dim ������Ϊ index ����Ƭ��
	*/
	torch::Tensor xc = prediction.select(2, 4) > confThres;
	std::cout << "score confthres tensor shape:";
	showShape(xc);

	// ���������������ֱ�Ϊ����ߺ���� NMS ��ѡ��������
	int maxWh = 4096;
	int maxNms = 30000;
	// ��ʼ��һ���յ� std::vector<torch::Tensor> ���͵�������� output����СΪ batch_size������ÿ������������ʼ��Ϊһ����СΪ (0, 6) ��ȫ��������
	std::vector<torch::Tensor> output;
	for (int i = 0; i < prediction.size(0); i++) {
		output.push_back(torch::zeros({ 0,6 }));
	}
	for (int i = 0; i < prediction.size(0); i++) {
		// ����ÿһ������ͼƬ��Ԥ������ȡ������Ŀ�����Ŷ���ֵҪ���Ԥ��򣬲����浽�������� x �С�
		torch::Tensor x = prediction[i];
		x = x.index_select(0, torch::nonzero(xc[i]).select(1, 0));
		if (x.size(0) == 0)continue;//��� x ��û���κη���Ҫ���Ԥ�����ֱ����������Ĵ������������һ������ͼƬ��

		x.slice(1, 5, x.size(1)).mul_(x.slice(1, 4, 5)); //����ѡȡx�����ڵ�1��ά���ϣ�������5��ʼ��ĩβ����һ���֣�����ͬһ��λ���ϣ��õڶ������� x.slice(1, 4, 5) �Ķ�Ӧλ��Ԫ�����˷���Ȼ�����ԭ���� x ����������ÿ��Ԥ���������ͨ�������͸���ˣ�������������Ŷȣ��õ�Ԥ���ġ��÷֡���
		torch::Tensor box = xywh2xyxy(x.slice(1, 0, 4));
		/*
		���ȣ���һ�д���ʹ���� PyTorch �е� max() ������Ѱ��ÿ��Ԥ����и�����ߵ���𡣾���أ���ѡ����Ԥ������ x ���ŵ�һ��ά�Ȱ���˳��ӵ� 5 ��λ�ÿ�ʼ�����һ��λ�õ��������ݣ����� x ��ά��Ϊ [B, num_anchors, (num_classes+5), H, W]������ B Ϊ batch size������Щ���ݱ�ʾ��ÿ��Ԥ������������ĸ��ʣ�ע������Ҳ�����˱�����𣩡�Ȼ������Щ���������ŵڶ���ά��ѡ�����ֵ��������һ����СΪ [B, num_anchors, 1, H, W] ���������������������ÿ��Ԥ����и�����ߵ�������������һ����СΪ [B, num_anchors, 1, H, W] ���������������������ÿ��Ԥ����и�����ߵ����ĸ���ֵ����
		*/
		std::tuple<torch::Tensor, torch::Tensor> max_tuple = torch::max(x.slice(1, 5, x.size(1)), 1, true);
		/*
	 ���������� box��std::get<0>(max_tuple) �� std::get<1>(max_tuple) ���ŵڶ���ά�Ƚ���ƴ�ӣ�box ����������ÿ��Ԥ����λ����Ϣ��std::get<0>(max_tuple) �����Ǹ�����������������1 ά����std::get<1>(max_tuple) �����Ǹ����������ĸ���ֵ��1 ά��������������ͬ�����Ԥ���ķ�����Ϣ���� 2 ά����
		*/
		x = torch::cat({ box,std::get<0>(max_tuple),std::get<1>(max_tuple) }, 1);
		// ���˵����ʵ�����ֵ��Ԥ��򣬵õ�һ���µ�������
		x = x.index_select(0, torch::nonzero(std::get<0>(max_tuple) > confThres).select(1, 0));

		std::cout << "prediction��:";
		showShape(x);

		// û�п��continue������򳬹������ֵ��ֻ����Ǯmaxnms��
		int n = x.size(0);
		if (n == 0) {
			continue;
		}
		else if (n > maxNms) {
			x = x.index_select(0, x.select(1, 4).argsort(0, true).slice(0, 0, maxNms));
		}
		/*
		�����д����ǽ�Ԥ�������λ�ã�����-���-�߶ȣ�ת��Ϊͼ��������ꡣ
		���ȣ�x.slice(1, 5, 6) ��ʾ�� x ��ѡȡά��Ϊ 1�����ڶ���ά�ȣ�Ҳ������Ԥ����йص�ά�ȣ����±�� 5 ��ʼ�������� 6 ������Ԫ�أ���ЩԪ�ر�ʾԤ���ĸ߶���Ϣ��Ȼ����Щ�߶�ֵ����һ�����Ŀ�Ⱥ͸߶ȱ������� maxWh���õ��ڵ�ǰ����ͼ�µ�ʵ�ʸ߶ȡ�
		���ţ�x.slice(1, 0, 4) ��ʾ�� x ��ѡȡά��Ϊ 1���±�� 0 ��ʼ�������� 4 ������Ԫ�أ���ЩԪ�ر�ʾԤ����λ����Ϣ�������ϽǺ����½ǵ����꣩��Ȼ����Щλ����Ϣ���ϸոռ���õ������ĵ�ƫ����,�õ���Ԥ�����ԭͼ�еľ������ꡣ
		*/
		torch::Tensor c = x.slice(1, 5, 6) * maxWh;
		torch::Tensor boxes = x.slice(1, 0, 4) + c, scores = x.select(1, 4);
		torch::Tensor ix = nms(boxes, scores, iouThres).to(x.device());
		output[i] = x.index_select(0, ix).cpu(); // ʹ�õõ����������ϴ�Ԥ��������ѡȡ��������Ԥ��򣬲����������������С�
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
	����ʹ�� torch::from_blob() ��������������ת�������������У�img.data ������ͼ���ָ�룬(height, width, 3) ������ͼ��ĳߴ磬torch::kByte ��ʾת�����������������Ϊ uint8 ���� unsigned char����
	����ʹ�� permute() ������������ά�ȣ���ͨ��ά�ȣ�C���ŵ���һά�������ά�ȣ�W���ŵ��ڶ�ά�����߶�ά�ȣ�H���ŵ�����ά��������Ϊ�� PyTorch �У�ͨ��ά���ǵ�һά���൱�� NumPy �е����һά��
	���� toType() ��������������������ת��Ϊ torch::kFloat���������ȸ����͡�������Ϊ�����ѧϰ�У������ģ�͵������������Ǹ����͡�
	ʹ�� div() ��������������ֵ��Χ�� (0,255) ��һ��Ϊ (0,1)������Ϊ�˽��������ݵ���ֵ��Χӳ�䵽ģ��ѵ��ʱʹ�õ������ڣ��Դﵽ���õľ��Ⱥ�Ч����
	���ʹ�� unsqueeze() �����ڵ�һά����һ���µ�ά�ȣ��������ĳߴ�� (3, H, W) ��չΪ (1, 3, H, W)��������Ϊ�� PyTorch �У�ģ�͵�����һ�㶼��һ�� batch �����ݣ���Ҫ�ڵ�һά����һ�� batch ά�ȡ�
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

	auto x1 = bboxes.select(1, 0); // ȡ����������Ͻ�x����
	auto y1 = bboxes.select(1, 1); // ȡ����������Ͻ�y����
	auto x2 = bboxes.select(1, 2); // ȡ����������½�x����
	auto y2 = bboxes.select(1, 3);
	auto areas = (x2 - x1) * (y2 - y1); // �������
	auto tuple_sorted = scores.sort(0, true); // �� scores �������մӴ�С��˳��������򣬲����������Ľ���Ͷ�Ӧ��������������ԣ����� 0 ��ʾ���� 0 ά�����򣨼��������򣩣�true ��ʾ�������С�
	auto order = std::get<1>(tuple_sorted); // �� tuple_sorted ��ȡ���ź��������ֵ���� scores ������ÿ��Ԫ�ض�Ӧ��ԭʼλ�á���һ��Ԫ���������Ľ���������ڶ���Ԫ���Ƕ�Ӧ��������������� std::get<1>(tuple_sorted) ���صľ�������������������

	std::vector<int> keep; //������������
	while (order.numel() > 0) {
		// ��� order ������ֻʣ�� 1 ��Ԫ�أ���ôֱ�ӽ���Ԫ�ص��������� keep ����
		if (order.numel() == 1) {
			auto i = order.item();
			keep.push_back(i.toInt());
			break;
		}
		else {
			// ��� order �����л�ʣ�¶��Ԫ�أ�������������ߵ�Ԫ�ص��������� keep �������� order[0] ����Ӧ������ֵ,��������� PyTorch �� item() ��������������Ψһ�ı���ֵȡ��������������ת��Ϊ���͡�Ȼ��ʹ�� push_back() ����������ӵ� keep ����ĩβ��
			auto i = order[0].item();
			keep.push_back(i.toInt());
		}

		// order_mask ������һ���� order ����ȥ����һ��Ԫ�غ��γɵ����������������˳���߷ּ�����������м��������ֵ��
		auto order_mask = order.narrow(0, 1, order.size(-1) - 1);
		// �� index() �������� order_mask �е�����ֵ�� x1��y1��x2��y2 �ĸ�������ȡ����Ӧ��ֵ�����У�clamp() ����������ȡ����ֵ������ָ���ķ�Χ�ڡ�������Ҫע����ǣ�clamp() �����ĵ�һ������������һ����������ʾ����Ӧ��Ԫ�طֱ���������Ӧ�ķ�Χ�ڡ�
		auto xx1 = x1.index({ order_mask }).clamp(x1[keep.back()].item().toFloat(), 1e10);
		auto yy1 = y1.index({ order_mask }).clamp(y1[keep.back()].item().toFloat(), 1e10);
		auto xx2 = x2.index({ order_mask }).clamp(0, x2[keep.back()].item().toFloat());
		auto yy2 = y2.index({ order_mask }).clamp(0, y2[keep.back()].item().toFloat());

		// ���ȣ������ཻ��� inter��������������ص����ֵ������������ԣ�(xx2 - xx1) �� (yy2 - yy1) ��ʾ����������ˮƽ�ʹ�ֱ�����ϵ��ص����ȣ�������˼�Ϊ�ཻ���������ͬ��ʹ���� clamp() �������ཻ��������� 0 �� 1e10 ֮�䡣���ţ����� IOU ֵ�����У�area1 ��ʾ��һ�������������� areas[keep.back()]��area2 ��ʾ������������֮�ͣ��� areas.index({ order.narrow(0,1,order.size(-1) - 1) })�����ﻹ�õ��� narrow() ��������ʾ����������ָ��ά�Ƚ��вü�������ָ����Χ�ڵ�Ԫ�ء�
		auto inter = (xx2 - xx1).clamp(0, 1e10) * (yy2 - yy1).clamp(0, 1e10);
		auto iou = inter / (areas[keep.back()] + areas.index({ order.narrow(0,1,order.size(-1) - 1) }) - inter);

		// ��δ����ж� IOU ֵ�Ƿ�С����ֵ�����ѷ����������������浽���� idx �С���� idx ����Ϊ�գ�˵���Ѿ�û�з��������ļ�������ѭ����
		auto idx = (iou <= thresh).nonzero().squeeze();
		if (idx.numel() == 0) {
			break;
		}
		// ��� idx �����ǿգ��� order �����ж�Ӧ��Ԫ��ɾ���������µ�������ֵ�� order ������������Ҫע����ǣ����� idx �����б�����ǳ���߷ּ������������������ڽ�����������ʱ��Ҫ�Ƚ���Щ������ 1�����ܵõ���ȷ�Ľ����
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
	// ������������� result �е�ÿ��Ԫ��
	for (int i = 0; i < result.size(); i++) {
		torch::Tensor data = result[i];
		ImageResizeData imgRD = imgRDs[i];
		// ����ÿ������
		for (int j = 0; j < data.size(0); j++) {
			torch::Tensor tensor = data.select(0, j); // ��������ȡ���� j ������
			// �ж�ͼƬ�Ƿ����ŵ��˿�����ޡ���Ϊ���ź��ͼƬ��߱ȿ�����ԭͼ��ͬ��������Ҫ����ͼƬ�Ƿ��������ֱ������������ꡣ
			if (imgRD.isW()) {
				tensor[1] -= imgRD.border; // ���border��ȫ
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
			// ���˺ڱ�֮��Ԥ���������ںڱ��ϣ��ͻ���ɽ��Ϊ����
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
	��δ����Ƕ��������ݽ���ǰ�򴫲��Ĳ��������У�model ��һ�� PyTorch ģ�ͣ�data ��һ���������ݵ�������ͨ������ model.forward({ data }) ���������������ݴ��ݸ�ģ�ͽ���ǰ�򴫲��������һ�� at::IValue ���͵ķ���ֵ����ģ�������������ֵ��һ��Ԫ�飨Tuple�����ͣ�������ģ�͵������������������ֻ��һ����������ͨ�� toTuple()->elements()[0] ��ȡԪ���еĵ�һ��Ԫ�أ�Ȼ����ͨ�� toTensor() ����ת��Ϊһ����������torch::Tensor����
	*/
	torch::Tensor pred = model.forward({ data }).toTuple()->elements()[0].toTensor();

	// ��ӡpred��shape
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

	imgResizeData.h = h; // ԭʼͼƬ�ĸ߿�
	imgResizeData.w = w;
	imgResizeData.height = height; // ת������ͼƬ�ĸ߿�
	imgResizeData.width = width;
	/*
	��δ���������Ǹ��� isW ��ֵ������ͼ������š�������˵�����ȸ���ͼ���Ŀ���߱� w/h ��ԭʼ��߱� width/height �Ĵ�С��ϵ��ȷ���ǰ��տ�Ȼ��Ǹ߶Ƚ������ţ��������ȱ����������տ�Ƚ������ţ������ո߶Ƚ������š�Ȼ������ OpenCV ���е� resize ��������ͼ��������š�������տ�Ƚ������ţ���ͼ��Ŀ������ΪĿ���� width���߶��򰴱�������õ���������ո߶Ƚ������ţ���ͼ��ĸ߶�����ΪĿ��߶� height������򰴱�������õ������յõ������ź��ͼ��Ḳ�ǵ�ԭʼͼ�� img��
	*/
	bool isW = (float)w / (float)h > (float)width / (float)height; // ԭʼ��߱ȱ�ת����߱�Ҫ�󣬷���true��˵��ԭʼͼ�ܿ�
	cv::resize(img, img, cv::Size(
		isW ? width : (int)((float)height / (float)h * w),
		isW ? (int)((float)width / (float)w * h) : height));

	w = img.cols, h = img.rows;
	if (isW) {
		imgResizeData.border = (height - h) / 2;
		/*
		��δ���������Ƕ�ͼ����б߿���䡣������˵������ OpenCV ���е� copyMakeBorder ��������ͼ������·���ӱ߿�ʹ��ͼ��ĸ߶ȵ���Ŀ��߶� height��ͬʱ���ҷ����ϲ�������չ��������� (height - h) / 2 �� height - h - (height - h) / 2 �ֱ��ʾͼ���Ϸ����·���Ҫ�������ظ��������ǵļ��㹫ʽΪ (height - h) / 2 �� height - h - (height - h) / 2������ height ��Ŀ��߶ȣ�h ��ͼ��ԭʼ�߶ȡ����һ������ BORDER_CONSTANT ��ʾ���ó������߿��������ĳ���Ĭ��Ϊ��ɫ�����������ı߿����֮��ͼ��ĳߴ����� (height, width)������ͼ�������ʾ��*/
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

		// ѡ��ɫ
		it = colors.find(clazz);
		cv::Scalar color = NULL;
		if (it == colors.end()) {
			it = mainColors.find(clazz);//û�ҵ��������ֵ
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
