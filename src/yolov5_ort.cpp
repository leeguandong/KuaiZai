#include "yolov5_ort.h"


Yolov5ORT::Yolov5ORT(const std::string modelPath, const bool isGPU = true, const cv::Size inputSize = cv::Size(640, 640)) {
	// ���д��봴����һ��Ort::Env������������ONNXRuntime�����л�������־���𡣲���ORT_LOGGING_LEVEL_WARNINGָ������־����Ϊ���棬�������������־��"ONNX_DETECTION"�Ǹû��������ƣ����ڼ�¼��־��
	env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
	sessionOptions = Ort::SessionOptions(); // ��ʼ��ģ�ͻỰ��Ĭ������£����캯������Ҫ�κβ��������Զ�����ϵͳ���ý��г�ʼ����

	/*
	��δ�����ʹ�� ONNX Runtime��ORT�����е� API ����ȡ��ǰ���õ�ִ���ṩ�����б��������Ƿ�����Ϊ "CUDAExecutionProvider" ���ṩ����
	���ȣ�Ort::GetAvailableProviders() ��������һ�� std::vector ���͵��ַ������������а������õ� ORT ִ���ṩ��������ơ��������������Ϊ availableProviders��
	��������std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider") ���� "CUDAExecutionProvider" �������õ� ORT ִ���ṩ�����б����Ƿ���� CUDA ִ���ṩ����std::find ��������������������ָ��Ԫ�أ�������ָ���Ԫ�صĵ������������Ԫ�ز����ڣ��򷵻�������ĩβ�����������������������Ϊ cudaAvailable��
	*/
	std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
	auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");

	OrtCUDAProviderOptions cudaOption;

	if (isGPU && (cudaAvailable == availableProviders.end())) {
		std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
		std::cout << "Inference device: CPU" << std::endl;
	}
	else if (isGPU && (cudaAvailable != availableProviders.end())) {
		std::cout << "Inference device: GPU" << std::endl;
		/*
		sessionOptions ��һ�� Ort::SessionOptions ���͵Ķ��󡣵��� AppendExecutionProvider_CUDA() �������� cudaOption ��Ϊ�������ݸ��������Խ� CUDA ִ���ṩ������ӵ��Ựѡ���С�cudaOption ��һ�� OrtCUDAProviderOptions ���͵Ķ��󣬱�ʾ CUDA ��ѡ�
		*/
		sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
	}
	else {
		std::cout << "Inference device: CPU" << std::endl;
	}

#ifdef _WIN32 //�����뻷��Ϊ Windows ʱ����Ҫ�� C++ ��׼�ַ������ͣ�std::string��ת��Ϊ���ַ��ַ������ͣ�std::wstring��
	std::wstring w_modelPath = charToWstring(modelPath.c_str());//modelPath�е�c_str()��������һ��ָ���Կ��ַ�����β���ַ����飨C����ַ�������ָ�룬��ָ���һ���ַ���ָ�롣
	session = Ort::Session(env, w_modelPath.c_str(), sessionOptions);
#else
	session = Ort::Session(env, modelPath.c_str(), sessionOptions);//Ort::Session �ĵڶ�������������ָ�� ONNX ģ���ļ�·���� C ����ַ���ָ�롣
#endif // _WIN32

	/*
	��δ��������ڻ�ȡ ONNX ģ�͵�����������״��Ϣ��
	���ȣ�����ͨ�� Ort::Session::GetInputTypeInfo() ������ȡ��ģ�͵ĵ�һ������ڵ��������Ϣ inputTypeInfo������ڵ������Ϊ0����ʾ��һ������ڵ㡣���ģ���ж������ڵ㣬���ǿ��Ը��ݽڵ��˳�����λ�ȡ���ǵ�������Ϣ��
	������������ͨ�� Ort::TypeInfo::GetTensorTypeAndShapeInfo() ������ȡ���������������ͺ���״��Ϣ��Ȼ������ͨ�� Ort::TensorTypeAndShapeInfo::GetShape() ������ȡ����������״��Ϣ�������䱣���� inputTensorShape �����С�
	inputTensorShape ��һ�� std::vector<int64_t> ���͵ı��������ڱ�����������״��Ϣ��������˵������һ��һά�����飬ÿ��Ԫ�ر�ʾ�����ڶ�Ӧά���ϵĳ��ȡ����磬���һ����������״�� (2, 3, 4)����ô����Ӧ�� inputTensorShape ����һ����������Ԫ�ص����飺[2, 3, 4]��
	��Ҫע����ǣ����ģ���д��ڲ�ȷ����С��dynamic shape������������ô�ڻ�ȡ������״��Ϣʱ���ܻ��׳��쳣�����⣬��ĳЩ����£�ģ�͵�����������״��Ϣ�����ǲ���֪�ģ����������Ҫ���ʵ��������д���
	*/
	Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
	std::vector<int64_t> inputTensorShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
	this->isDynamicInputShape = false;
	if (inputTensorShape[2] == -1 && inputTensorShape[3] == -1) {
		std::cout << "Dynamic input shape" << std::endl;
		this->isDynamicInputShape = true;
	}

	for (auto shape : inputTensorShape)
		std::cout << "Input shape: " << shape << std::endl;

	/*
	��ʹ�� ONNX Runtime ����ģ������ʱ��������Ҫ���������ݣ�����ͼ���ı��ȣ�ת��Ϊģ�������������ʽ�������䴫�ݸ� Ort::Session::Run() �������м��㡣�ڴ˹����У�������Ҫ����һЩ������ڴ���������Щ�������ݣ������䴫�ݸ� ONNX Runtime ��Ort::AllocatorWithDefaultOptions ����Ϊ���Ŀ�Ķ���Ƶġ�
	���������������� GetInputName() �� GetOutputName() ���������У���һ��������ʾ����/����ڵ���������������Ƕ�������0����ʾ��һ������/����ڵ㡣�ڶ���������һ���ڴ�����������ڷ��䷵�����Ƶ��ڴ�ռ䡣
	��ʹ�� std::vector<const char*>����ָ�����ַ���ָ������������������ʱ��������Ҫȷ����Щ�����������������������ڶ�����Ч�ġ�������Щ�����Ǵ�ģ���л�ȡ�ģ��������ͨ�����ܶ����ǽ����ͷŻ��޸ġ�����Ҫ�޸Ļ��ͷ���Щ���ƣ����Կ���ʹ�� std::vector<std::string> ���������ʵ����ݽṹ���������ǡ�
	*/
	Ort::AllocatorWithDefaultOptions allocator;
	inputNames.push_back(session.GetInputName(0, allocator));
	outputNames.push_back(session.GetOutputName(0, allocator));

	std::cout << "Input name: " << inputNames[0] << std::endl;
	std::cout << "Output name: " << outputNames[0] << std::endl;

	this->inputImageShape = cv::Size2f(inputSize);
}

std::vector<Detection> Yolov5ORT::detect(cv::Mat& image, const float confThreshold = 0.4, const float iouThreshold = 0.45) {
	/*
	blob ��һ�� float ���͵�ָ�룬���ڴ洢����Ԥ������ͼ�����ݡ��ڸ÷����У��Ὣ�����ͼ�� image ����һϵ��Ԥ���������Ȼ�󽫴���õ��Ľ���洢�� blob ��
	*/
	float* blob = nullptr;
	std::vector<int64_t> inputTensorShape{ 1,3,-1,-1 };
	this->preprocessing(image, blob, inputTensorShape);

	/*
	�����д�������������һά�ġ���������˳��洢�ĸ�������������һ�� std::vector<float> �������а���������ͼƬ����������ֵ��������˵����һ�д��������������Ԫ�ص��ܸ�������������ά�ȵĴ�С��ˣ��õ���һ���������͵�ֵ inputTensorSize���ڶ��д��룬������ǰ�洴����ָ�� float �������ݵ�ָ����� blob���Լ�������������ߴ磬������һ������ inputTensorValues��
	���ھ������ӣ�����һ�� RGB ��ɫͼ����ߴ�Ϊ hxw������ h=224��w=224���򾭹�ͨ�������͹�һ���Ȳ����󣬸�ͼ��ͱ�ת��Ϊ��һ����״Ϊ {1, 3, 224, 224} ������������ blob ָ����ڴ�ռ��Ѿ��洢�˴�����ͼ�����ݣ���˿���ͨ�� blob ������Щ����ֵ���������е���Ԫ�ظ���Ϊ 1��3��224��224=30198912����ʱ����һ�д���ͻ����� inputTensorSize ��ֵΪ 30198912�����ڶ��д����� blob ָ����ڴ�ռ��е����ݴ洢���� inputTensorValues �����У�����������������еĸ���������ֵ�洢��һ����������˳���һά�����С��������ں�������������У���������ͻᱻ���ݸ� ONNX Runtime �������棬��������Ŀ��������ļ��㡣onnxruntime ��������һά��������Ҳ����֮ǰ�����кܺ�ʱ�Ĳ��������Ʒ�����norm�У��Եù�һ���ܺ�ʱ��

	����һ����Ϊ inputTensorValues �� std::vector<float> ��������Ԫ�ظ���Ϊ���������е�����Ԫ���������� inputTensorSize�����ҳ�ʼֵΪ�� blob ָ����ڴ��ַ��ʼ�������� inputTensorSize ��������ֵ��������˵�����������ͨ������ָ���һ��Ԫ�ص�ָ��(�� blob)��ָ�����һ��Ԫ�ص�ָ�� (�� blob + inputTensorSize) ����ʼ���ġ����൱�ڽ�һ��ָ�뷶Χ�ڵ����ݣ�ת����һ����׼������������

	����� blob ������ inputTensorValues �Ŀ���������ָ��ͬһ���ڴ�ռ��ָ�룬����� inputTensorValues �е�ÿ���޸Ĳ���Ҳ�ᷴӳ�� blob �ж�Ӧ��λ���ϡ�ͬʱ����Ҳ��ζ�ŵ� blob ��ָ����ڴ�ռ䱻�ͷŻ��޸�ʱ��inputTensorValues ����Ч��Ҳ����֮��ʧ��
	*/
	size_t inputTensorSize = vectorProduct(inputTensorShape);
	std::vector<float> inputTensorValues(blob, blob + inputTensorSize);

	std::vector<Ort::Value> inputTensors;
	/*
	���д�����ʹ�� ONNX Runtime ���е� Ort::MemoryInfo �ഴ���ڴ���Ϣ�������ڴ洢�����������ݵ��ڴ���Ϣ��������ԣ�ָ���˸��ڴ���Ϣ����ʹ�� CPU �ڴ棬���䷽ʽΪ OrtArenaAllocator��ʹ��Ĭ�ϵ��ڴ����� OrtMemTypeDefault��
	Ort::MemoryInfo ���ʾ ONNX Runtime �����е��ڴ���Ϣ�������ڹ����ڴ�ء��ڴ�����Դ��ͨ������ CreateCpu ��̬���������ڴ���Ϣ���󣬴�������������ֱ��Ƿ��䷽ʽ���ڴ����͡�
	���У�OrtArenaAllocator ��ʾҪ��һ���ض����ڴ�أ�arena���з����ڴ棬�Ա����η���С���ڴ�Ŀ������Ӷ�����ڴ�������ܣ�OrtMemTypeDefault ���ʾʹ��Ĭ�ϵ��ڴ����ͣ���Ĭ�ϵ��ڴ��ַ�ռ䣨address space����
	*/
	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault); // CreateCuda

	/*
	���ȣ����ǵ��� Ort::Value::CreateTensor<float> ��������һ�������͵��������󣬲����ö�����ڴ���Ϣ��ֵ���ݡ�Ԫ����������״��Ϣ�������״��Ϣ���鳤����Ϊ�������ݸ��ú�����
	���У�memoryInfo ������ʾ�洢�����������ݵ��ڴ���Ϣ���󣬼���һ�д��봴���� Ort::MemoryInfo ���ͱ��� memoryInfo��
	inputTensorValues.data() ��ʾ���������еĸ�����ֵ���ݣ������ô��ݵĸ������������� inputTensorValues �ĵײ�ָ�룬�����������洢��ͼ��������������ȡ����ֵ���ݣ�
	inputTensorSize ��ʾ����������Ԫ�ص��������������������ϵ����ص�������ͼ�����ĳ��Ϳ���˼���ó���inputTensorShape.data() �� inputTensorShape.size() �ֱ��ʾ������������״��Ϣ����һ���������飬ÿ��Ԫ�ر�ʾ��Ӧά�ȵĴ�С���Լ�������ĳ��ȣ������������Ľ��������Ƿֱ��Ӧ��ǰ�������ͨ����ȡͼ�����ĳ������ͨ���������������Ԫ������ inputTensorShape��
	*/
	inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorSize, inputTensorShape.data(), inputTensorShape.size()));

	/*
	������һ�� Ort::RunOptions ���󣬸ö���ָ����������ģ��ʱҪʹ�õ�ѡ��ڱ����У����ǽ�������Ϊ nullptr����ʾʹ��Ĭ��ѡ�
	���� this->session.Run ��������ģ�ͣ����������������ϡ�����������������Լ��������������Ϊ�������ݸ��÷������ڱ����У������������� 1 ��������������� 1 ����������Ǵ�����ֵΪ 1 ��������Ϊ������������������������
	inputNames.data() �� outputNames.data() �ֱ����������������������������������飬����ͨ�� std::vector<std::string> ���͵ı��� inputNames �� outputNames �洢�����������������֡��ڱ����У�����ֻʹ����һ������������һ�����������������Ǵ����˰��������ַ������������洢���ǵ����ơ�
	�����н���� std::vector<Ort::Value> ���͵Ķ��� outputTensors ���أ����д洢��ģ�ͼ����������������
	*/
	std::vector<Ort::Value> outputTensors = this->session.Run(Ort::RunOptions{ nullptr },
		inputNames.data(), inputTensors.data(), 1, outputNames.data(), 1);

	cv::Size resizedShape = cv::Size((int)inputTensorShape[3], (int)inputTensorShape[2]);
	std::vector<Detection> result = this->postprocessing(resizedShape, image.size(), outputTensors, confThreshold, iouThreshold);

	delete[] blob;
	return result;
}

void Yolov5ORT::preprocessing(cv::Mat& image, float*& blob, std::vector<int64_t>& inputTensorShape) {
	/*
	image������ͼ������Ϊ cv::Mat &��
	blob���洢Ԥ������ͼ�����ݵ�һά����ָ�룬����Ϊ float *&��ע�⣬����ʹ���� & �� *����ʾ�ò�����һ��ָ�����ã������޸���ָ����ڴ��ַ�������ں����ⲿҲ�ܹ����ʵ��޸ĺ��ֵ��
	inputTensorShape��������������״������Ϊ std::vector<int64_t> &��
	*/
	cv::Mat resizedImage, floatImage; // �ߴ����������ת��
	cv::cvtColor(image, resizedImage, cv::COLOR_BGR2RGB);

	letterbox(resizedImage, resizedImage, this->inputImageShape, cv::Scalar(114, 114, 114), this->isDynamicInputShape, true, 32);

	// �������������ĸ߶ȺͿ�ȣ�������Ϊ������� resizedImage �ĸ߶ȺͿ��,rows�Ǹ߶ȣ�cols�ǿ�ȡ�
	inputTensorShape[2] = resizedImage.rows;
	inputTensorShape[3] = resizedImage.cols;

	// ʹ�� convertTo ������������� resizedImage ���ݴ� uchar ����ת��Ϊ float ���ͣ����ҽ��й�һ������������ֵ���ŵ� [0, 1] ��Χ��
	resizedImage.convertTo(floatImage, CV_32FC3, 1 / 255.0);
	// ʹ�� new ������ڶ��Ϸ���һ����СΪ floatImage.cols * floatImage.rows * floatImage.channels() ��һά���飬�����䷵�ص�ָ�븳�� blob ���������ڴ洢Ԥ������ͼ������
	blob = new float[floatImage.cols * floatImage.rows * floatImage.channels()];

	cv::Size floatImageSize{ floatImage.cols,floatImage.rows };
	/*
	�� floatImage �е����ݰ���ͨ����ֳ����� cv::Mat ���͵ľ��󣬲������Ǵ洢�� blob �С�����أ����ȶ���һ�� vector ���͵ı��� chw������Ԫ�ظ������� floatImage ��ͨ�������� 3��Ȼ��ʹ�� cv::Mat ���캯��Ϊÿ��Ԫ�ش���һ����СΪ floatImageSize �ĵ�ͨ�� cv::Mat ���󣬲���������ָ����Ϊ blob + i * floatImageSize.width * floatImageSize.height�����ڴ洢 floatImage �� i ��ͨ�������ݡ���󣬵��� cv::split ������ floatImage �е����ݰ���ͨ����ֵ� chw �����У����� chw �е����ݴ洢�� blob �С�
	����һ�ַ�ʽ����ͨ�� cv �е� merge ʵ�֣�����splitҲ�㹻��Ч��
	*/
	std::vector<cv::Mat> chw(floatImage.channels());
	for (int i = 0; i < floatImage.channels(); ++i) {
		chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
	}
	cv::split(floatImage, chw);
}

std::vector<Detection> Yolov5ORT::postprocessing(const cv::Size resizedImageShape, const cv::Size originalImageShape, std::vector<Ort::Value>& outputTensors, const float confThreshold, const float iouThreshold) {
	std::vector<cv::Rect> boxes; // ����
	std::vector<float> confs; // ���Ŷ�
	std::vector<int> classIds; // ����ǩ

	/*
	���д�����������ȡģ���������������ָ��ġ�����ͨ������ 0 �� outputTensors �л�ȡ��һ�����������Ȼ��ʹ�� GetTensorData() ������ȡ��������ָ�롣	����� auto* ��һ��ָ�� float ���͵�ָ��.
	rawOutput ָ��������ڷ�����������д洢��ԭʼ���ݡ��ڱ������У����Ǻ����Ὣ��ת��Ϊ std::vector<float> ���͵Ķ��� output���Ա������ش���������ݡ�
	*/
	auto* rawOutput = outputTensors[0].GetTensorData<float>();
	/*
	���д�����������ȡģ�������������״��Ϣ�ġ�����ͨ������ 0 �� outputTensors �л�ȡ��һ�����������Ȼ��ʹ�� GetTensorTypeAndShapeInfo() ������ȡ�������ͺ���״��Ϣ�����ʹ�� GetShape() ������ȡ�����������״��Ϣ��
	����� std::vector<int64_t> ��һ���洢 int64_t ����Ԫ�صĶ�̬���飬����ֵΪ outputTensors[0].GetTensorTypeAndShapeInfo().GetShape() �ķ���ֵ�����԰�������һ�����飬���� outputShape[i] ��ʾ��������ڵ� i ά�ϵĴ�С��
	������Ŀ����ģ���У������������״�ͺ��嶼��Ԥ����õģ�������������ǿ���ͨ�����������������״��Ϣ�����������Ԫ�ص�λ�ã��Ӷ�����ط���������ݡ�
	*/
	std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
	//���д����ǻ�ȡģ�����������Ԫ�ص�����Ŀ��
	size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
	/*
	���д����ǽ����������ԭʼ����ת��Ϊ std::vector<float> ����Ĵ��롣ʹ�� std::vector ��Ĺ��캯������ rawOutput ָ��ָ��������������ת��Ϊ std::vector ���Ͷ��� output��
	����� std::vector<float> ��һ���洢 float ����Ԫ�صĶ�̬���飬����ֵΪ std::vector<float>(rawOutput, rawOutput + count) �ķ���ֵ��������캯���ĵ�һ��������ָ�����ݵ�ָ�룬�ڶ���������ָ�����ݽ���λ�õ�ָ�롣���Ὣָ����Χ�ڵ����ݿ����� std::vector �����С�
	*/
	std::vector<float> output(rawOutput, rawOutput + count);

	/*
	outputShape ��һ���洢���������״��Ϣ�� std::vector<int64_t> ��������һ��һά���飬��ʾ��������ĸ���ά�ȴ�С��
	������˵�����������״һ��Ϊ [batch_size, num_anchors_per_cell * grid_size * grid_size, num_classes + 5]����������ά�ȡ����У�
	�� 1 ά�� batch_size ��ʾ���������е�����������Ҳ�����δ�С����
	�� 2 ά�� num_anchors_per_cell * grid_size * grid_size ��ʾ��ÿ��������Ԥ����ٸ��߽�򣨼�ê��������������������С��ƽ�������ɵõ��ܵı߽��������
	�� 3 ά�� num_classes + 5 ��ʾÿ���߽���Ԥ����������ǰ 4 ��Ԫ���Ǳ߽���������Ϣ���� 5 ��Ԫ�������Ŷȣ�confidence��������� num_classes ��Ԫ���Ǵ���ͬ���ĸ��ʡ�first 5 elements are box[4] and obj confidence
	*/
	int numClasses = (int)outputShape[2] - 5;
	int elementsInBatch = int(outputShape[1] * outputShape[2]);

	/*
	��Ŀ����ģ�͵���������н�����������ߵı߽�򣬲��������ꡢ���Ŷȼ������Ϣ�洢������ boxes��confs �� classIds �С�
	������˵��������ʹ��һ��ѭ�����������е�ÿ��Ԫ�ء�����Ǹ�һά���飬����ÿ��Ԫ�ض�Ӧ��һ���߽�����ÿ��ѭ����Ҫ���������е�һ��������Ԫ�أ���������������Ϊ outputShape[2]��ѭ�����ȶ�ȡ�� 5 ��Ԫ�أ������Ŷȣ���������趨����ֵ confThreshold����˵���ñ߽����ܰ������󣬿��Խ��к�������
	�ڶ�ȡ�߽����Ϣʱ�����ȶ�ȡǰ 4 ��Ԫ�أ����߽������ĵ�����Ϳ�ߣ�Ȼ���������Ͻǵ����ꡣ���ŵ��ú��� getBestClassInfo�������ж�ȡ���ڸñ߽�����������Ϣ�������������Ŷ���˵õ��ñ߽����������Ŷ� confidence����󽫸ñ߽�����Ϣ�����ꡢ���Ŷȡ���𣩴洢������ boxes��confs �� classIds �С�
	*/
	for (auto it = output.begin(); it != output.begin() + elementsInBatch; it += outputShape[2]) {
		float clsConf = it[4];
		if (clsConf > confThreshold) {
			int centerX = (int)(it[0]);
			int centerY = (int)(it[1]);
			int width = (int)(it[2]);
			int height = (int)(it[3]);
			int left = centerX - width / 2;
			int top = centerY - height / 2;

			float objConf;
			int classId;
			this->getBestClassInfo(it, numClasses, objConf, classId);
			float confidence = clsConf * objConf;

			boxes.emplace_back(left, top, width, height);
			confs.emplace_back(confidence);
			classIds.emplace_back(classId);
		}
	}

	/*
	��δ���ʹ�� OpenCV �� NMSBoxes ������Ԥ��ı߽����зǼ���ֵ���ƣ�NMS�������������ص��ı߽�򣬲��������Ŷ���ߵ���һ����
	������˵���������� 5 ��������boxes ��һ���������洢������Ԥ��߽���������Ϣ��confs ��һ���������洢������Ԥ��߽������Ŷ���Ϣ��confThreshold ����ֵ�����ڹ��˵����Ŷȵ������ı߽��iouThreshold �� IOU ��ֵ�����������ص��ȳ�����ֵ�ı߽����Ϊͬһ�����壻indices ��һ������������洢���� NMS ��������µı߽���������
	�ں����ڲ������Ƚ� boxes �� confs ��ÿһ��Ԫ�ش����һ�� cv::Rect ��һ�������������洢�� rects �����С�Ȼ����� cv::dnn::NMSBoxes �������� NMS �����������µı߽��������洢�� indices �����С�
	*/
	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confs, confThreshold, iouThreshold, indices);

	// ��Ŀ��������껹ԭ��ͼ��
	std::vector<Detection> detections;
	for (int idx : indices) {
		Detection det;
		det.box = cv::Rect(boxes[idx]);
		this->scaleCoords(resizedImageShape, det.box, originalImageShape);

		det.conf = confs[idx];
		det.classId = classIds[idx];
		detections.emplace_back(det);
	}
	return detections;
}

void Yolov5ORT::letterbox(const cv::Mat& image, cv::Mat& outImage, const cv::Size newShape = cv::Size(640, 640), const cv::Scalar color = cv::Scalar(114, 114, 114), bool auto_ = true, bool scaleUp = true, int stride = 32) {
	cv::Size shape = image.size();

	/*
	�������ű��� r����������ͼ�����ŵ�ָ����С��Ҫ���Ե�ϵ�������ȣ�������ͼ��ĸ߿���ָ����С�Ƚϣ���ѡ�����н�С��һ����Ϊ����ϵ�������浽���� r �С���� scaleUp ����Ϊ false���� r �ٴ��� 1.0f ȡ��Сֵ��ȷ�������ԭͼ���зŴ������
	*/
	float r = std::min((float)newShape.height / (float)shape.height,
		(float)newShape.width / (float)shape.width); // ��640������Сֵ��������߽�������

	if (!scaleUp)
		r = std::min(r, 1.0f); // ����

	//float ratio[2]{ r,r };
	/*
	������ͼ���С�����ź��ԭͼ���С�����Ҫ��ӵ��������ұ߿��� dw �� dh������������Զ����루�� auto_ Ϊ true������ dw �� dh ����Ϊ stride �ı�������ȷ���ɱ���������
	*/
	int newUnpad[2] = { (int)std::round((float)shape.width * r),
						(int)std::round((float)shape.height * r) };
	auto dw = (float)(newShape.width - newUnpad[0]);
	auto dh = (float)(newShape.height - newUnpad[1]);

	if (auto_) {
		dw = (float)((int)dw % stride);
		dh = (float)((int)dh % stride);
	}
	dw /= 2.0f;
	dh /= 2.0f;
	if (shape.width != newUnpad[0] && shape.height != newUnpad[1]) {
		cv::resize(image, outImage, cv::Size(newUnpad[0], newUnpad[1]));
	}
	/*
	���ݱ߿��Ⱥ���ɫʹ�� cv::copyMakeBorder ������ͼ����Χ��ӱ߿򣬲���������浽 outImage �С���Ҫע����ǣ��ڼ����������ұ߿���ʱ��ʹ���� std::round ������������������������Ҽ�ȥ��һ����С������0.1f�����Ա��� cv::copyMakeBorder �����������⡣
	*/
	int top = int(std::round(dh - 0.1f));
	int bottom = int(std::round(dh + 0.1f));
	int left = int(std::round(dw - 0.1f));
	int right = int(std::round(dw + 0.1f));
	cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

size_t Yolov5ORT::vectorProduct(const std::vector<int64_t> vector) {
	if (vector.empty())
		return 0;

	size_t product = 1;
	for (const auto& element : vector) {
		product *= element;
	}
	return product;
}

void Yolov5ORT::getBestClassInfo(std::vector<float>::iterator it, const int& numClasses, float& bestConf, int& bestClassId) {
	/*
	��δ���ʵ������Ŀ����ģ����������в���Ԥ��߽���Ӧ�������������ŶȵĹ��ܡ�
	������˵��������������������һ�������� it����ָ�����������һ��Ԥ��߽�����ʼλ�ã�һ������ numClasses����ʾ����������Լ��������ò��� bestConf �� bestClassId�������������������������ŶȺͱ�ŷֱ�ֵ��������������
	�������Ƚ���ʼֵ����ΪĬ��ֵ���������� ID Ϊ 5����Ϊǰ 5 ��Ԫ�طֱ��Ǳ߽��Ͷ������Ŷȣ���������Ŷ�Ϊ 0��Ȼ���ڴӵ� 6 ��Ԫ�ص��� numClasses + 5 ��Ԫ���б���������ÿ�����������ĸ���ֵ���ڵ�ǰ��Ѹ���ֵ bestConf���������Ϊ��ǰ����ֵ������������ ID ��Ϊ������ ID��
	*/
	bestClassId = 5;
	bestConf = 0;

	for (int i = 5; i < numClasses + 5; i++) {
		if (it[i] > bestConf) {
			bestConf = it[i];
			bestClassId = i - 5;
		}
	}
}

void Yolov5ORT::scaleCoords(const cv::Size imageShape, cv::Rect& coords, const cv::Size imageOriginalShape) {
	// ����ģ��Ԥ���nms֮�������ָ�,����Ԥ������һ��
	// ����ͼ����Դͼ���ں���������ϵ����ű�����Ȼ��ȡ���������еĽ�Сֵ��Ϊ gain ��ֵ
	float gain = std::min((float)imageShape.height / (float)imageOriginalShape.height,
		(float)imageShape.width / (float)imageOriginalShape.width);

	// �������ź�ˮƽ����ֱ�����paddingֵ
	int pad[2] = { (int)(((float)imageShape.width - (float)imageOriginalShape.width * gain) / 2.0f),
		(int)(((float)imageShape.height - (float)imageOriginalShape.height * gain) / 2.0f) };

	// ���ݼ���õ�������ϵ�������ֵ�����¼��㲢������Ŀ�������������Ϣ,����ȷ��Ŀ��������ԭʼͼ���ϵ�λ����Ϣ�õ�׼ȷ��ԭ�����Ҳ�������ͼ��ߴ��Ӱ�졣
	coords.x = (int)std::round(((float)(coords.x - pad[0]) / gain));//pad��ԭͼת��Ŀ��ߴ�ʱ�ӵģ����ڴ�Ŀ��ߴ���ԭͼתʱҪ������֮���ڳ�gainȥ��ԭ
	coords.y = (int)std::round(((float)(coords.y - pad[1]) / gain));
	coords.width = (int)std::round(((float)coords.width / gain));
	coords.height = (int)std::round(((float)coords.height / gain));
}

std::wstring Yolov5ORT::charToWstring(const char* str) {
	/*
	*��ʾָ�룬����������У�const char* str ��ʾһ��ָ�� char ���͵ĳ���ָ�롣Ҳ����˵��str �����洢һ��ָ�� char �������ݵ��ڴ��ַ���Ҹõ�ַ�����Ա��޸ģ���Ϊʹ���� const �޶�����
	�ú����Ĺ����ǽ� C ����ַ���ת��Ϊ std::wstring ���͵��ַ�����

	����ʵ�ֹ������£�
	����һ�� typedef������ָ���ַ������뷽ʽ������ʹ���� codecvt_utf8<wchar_t> ���ͣ���ʾ�� UTF-8 ������ַ���ת��Ϊ���ַ����루wchar_t�����ַ�����
	����һ�� wstring_convert ���� converter�����ڽ����ַ�������ת����
	ʹ�� converter.from_bytes(str) ����������� C ����ַ��� str ת��Ϊ std::wstring �ַ����������ظ��ַ�������ת�������У�from_bytes() �����ᰴ�� converter ����ָ���ı��뷽ʽ�������ַ���ת��Ϊ���ַ�������ַ�����
	��Ҫע����ǣ��ú���ʵ�������� C++11 �����ϰ汾�ṩ�� std::wstring_convert ��� codecvt_utf8<wchar_t> ���͡���Щ���Ϳ�����ͷ�ļ� <codecvt> ���ҵ���
	*/
	typedef std::codecvt_utf8<wchar_t> convert_type;
	std::wstring_convert<convert_type, wchar_t> converter;
	return converter.from_bytes(str);
}

std::vector<std::string> Yolov5ORT::loadNames(const std::string path) {
	// load class names
	std::vector<std::string> classNames;
	std::ifstream infile(path);
	if (infile.good()) {
		std::string line;
		while (std::getline(infile, line)) { // ��ȡ�ļ��е�ÿһ��
			if (line.back() == '\r')  //��������Իس��� \r ��β�������Ƴ���������Ϊ Windows ϵͳ�µ��ı��ļ��У���ĩͨ������� \r\n�����س����У������� Linux ��ϵͳ�У���ֻ���� \n��
				line.pop_back();
			classNames.emplace_back(line);
		}
		infile.close();
	}
	else {
		std::cerr << "ERROR:Failed to access class name path:" << path << std::endl;
	}
	return classNames;
}

void Yolov5ORT::visualizeDetection(cv::Mat& image, std::vector<Detection>& detections, const std::vector<std::string> className) {
	for (const Detection& detection : detections) {
		cv::rectangle(image, detection.box, cv::Scalar(229, 160, 21), 2);

		int x = detection.box.x;
		int y = detection.box.y;
		int conf = (int)std::round(detection.conf * 100);
		int classId = detection.classId;
		std::string label = className[classId] + "0." + std::to_string(conf);

		int baseline = 0;
		cv::Size size = cv::getTextSize(label, cv::FONT_ITALIC, 0.8, 2, &baseline);
		cv::rectangle(image, cv::Point(x, y - 25), cv::Point(x + size.width, y), cv::Scalar(229, 160, 21), -1);
		cv::putText(image, label, cv::Point(x, y - 3), cv::FONT_ITALIC, 0.8, cv::Scalar(255, 255, 255), 2);
	}
}

