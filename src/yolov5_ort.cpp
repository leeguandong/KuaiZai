#include "yolov5_ort.h"


Yolov5ORT::Yolov5ORT(const std::string modelPath, const bool isGPU = true, const cv::Size inputSize = cv::Size(640, 640)) {
	// 这行代码创建了一个Ort::Env对象，用于设置ONNXRuntime的运行环境和日志级别。参数ORT_LOGGING_LEVEL_WARNING指定了日志级别为警告，即仅输出警告日志。"ONNX_DETECTION"是该环境的名称，用于记录日志。
	env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
	sessionOptions = Ort::SessionOptions(); // 初始化模型会话，默认情况下，构造函数不需要任何参数，会自动根据系统配置进行初始化。

	/*
	这段代码是使用 ONNX Runtime（ORT）库中的 API 来获取当前可用的执行提供程序列表，并查找是否有名为 "CUDAExecutionProvider" 的提供程序。
	首先，Ort::GetAvailableProviders() 函数返回一个 std::vector 类型的字符串向量，其中包含可用的 ORT 执行提供程序的名称。这个向量被命名为 availableProviders。
	接下来，std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider") 根据 "CUDAExecutionProvider" 搜索可用的 ORT 执行提供程序列表中是否存在 CUDA 执行提供程序。std::find 函数可以在容器中搜索指定元素，并返回指向该元素的迭代器。如果该元素不存在，则返回容器的末尾迭代器。这个迭代器被命名为 cudaAvailable。
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
		sessionOptions 是一个 Ort::SessionOptions 类型的对象。调用 AppendExecutionProvider_CUDA() 方法并将 cudaOption 作为参数传递给它，可以将 CUDA 执行提供程序添加到会话选项中。cudaOption 是一个 OrtCUDAProviderOptions 类型的对象，表示 CUDA 的选项。
		*/
		sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
	}
	else {
		std::cout << "Inference device: CPU" << std::endl;
	}

#ifdef _WIN32 //当编译环境为 Windows 时，需要将 C++ 标准字符串类型（std::string）转换为宽字符字符串类型（std::wstring）
	std::wstring w_modelPath = charToWstring(modelPath.c_str());//modelPath中的c_str()函数返回一个指向以空字符串结尾的字符数组（C风格字符串）的指针，即指向第一个字符的指针。
	session = Ort::Session(env, w_modelPath.c_str(), sessionOptions);
#else
	session = Ort::Session(env, modelPath.c_str(), sessionOptions);//Ort::Session 的第二个参数必须是指向 ONNX 模型文件路径的 C 风格字符串指针。
#endif // _WIN32

	/*
	这段代码是用于获取 ONNX 模型的输入张量形状信息。
	首先，我们通过 Ort::Session::GetInputTypeInfo() 方法获取了模型的第一个输入节点的类型信息 inputTypeInfo。这个节点的索引为0，表示第一个输入节点。如果模型有多个输入节点，我们可以根据节点的顺序依次获取它们的类型信息。
	接下来，我们通过 Ort::TypeInfo::GetTensorTypeAndShapeInfo() 方法获取了输入张量的类型和形状信息。然后，我们通过 Ort::TensorTypeAndShapeInfo::GetShape() 方法获取了张量的形状信息，并将其保存在 inputTensorShape 变量中。
	inputTensorShape 是一个 std::vector<int64_t> 类型的变量，用于保存张量的形状信息。具体来说，它是一个一维的数组，每个元素表示张量在对应维度上的长度。例如，如果一个张量的形状是 (2, 3, 4)，那么它对应的 inputTensorShape 就是一个包含三个元素的数组：[2, 3, 4]。
	需要注意的是，如果模型中存在不确定大小（dynamic shape）的张量，那么在获取张量形状信息时可能会抛出异常。此外，在某些情况下，模型的输入张量形状信息可能是不可知的，因此我们需要针对实际情况进行处理。
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
	在使用 ONNX Runtime 进行模型推理时，我们需要将输入数据（比如图像、文本等）转换为模型所需的张量格式，并将其传递给 Ort::Session::Run() 方法进行计算。在此过程中，我们需要分配一些额外的内存来保存这些张量数据，并将其传递给 ONNX Runtime 。Ort::AllocatorWithDefaultOptions 就是为这个目的而设计的。
	传递了两个参数给 GetInputName() 和 GetOutputName() 方法。其中，第一个参数表示输入/输出节点的索引，这里我们都传递了0，表示第一个输入/输出节点。第二个参数是一个内存分配器，用于分配返回名称的内存空间。
	在使用 std::vector<const char*>（即指向常量字符的指针向量）来保存名称时，我们需要确保这些名称在整个程序生命周期内都是有效的。由于这些名称是从模型中获取的，因此我们通常不能对它们进行释放或修改。若需要修改或释放这些名称，可以考虑使用 std::vector<std::string> 或其他合适的数据结构来保存它们。
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
	blob 是一个 float 类型的指针，用于存储经过预处理后的图像数据。在该方法中，会将输入的图像 image 进行一系列预处理操作，然后将处理得到的结果存储到 blob 中
	*/
	float* blob = nullptr;
	std::vector<int64_t> inputTensorShape{ 1,3,-1,-1 };
	this->preprocessing(image, blob, inputTensorShape);

	/*
	这两行代码是用来创建一维的、按行优先顺序存储的浮点数张量，即一个 std::vector<float> 对象，其中包含了输入图片的所有像素值。具体来说，第一行代码计算了张量中元素的总个数，即将各个维度的大小相乘，得到了一个整数类型的值 inputTensorSize。第二行代码，则利用前面创建的指向 float 类型数据的指针变量 blob，以及计算出的张量尺寸，创建了一个向量 inputTensorValues。
	对于具体例子，例如一个 RGB 彩色图像，其尺寸为 hxw，假设 h=224，w=224，则经过通道交换和归一化等操作后，该图像就被转换为了一个形状为 {1, 3, 224, 224} 的张量。由于 blob 指向的内存空间已经存储了处理后的图像数据，因此可以通过 blob 访问这些像素值。该张量中的总元素个数为 1×3×224×224=30198912。这时，第一行代码就会计算出 inputTensorSize 的值为 30198912。而第二行代码则将 blob 指向的内存空间中的数据存储到了 inputTensorValues 向量中，即将这个张量中所有的浮点数像素值存储到一个按行优先顺序的一维向量中。这样，在后续的推理过程中，这个向量就会被传递给 ONNX Runtime 推理引擎，用来进行目标检测任务的计算。onnxruntime 的输入是一维向量，这也是在之前车牌中很耗时的操作，车牌放在了norm中，显得归一化很耗时。

	创建一个名为 inputTensorValues 的 std::vector<float> 向量，其元素个数为输入张量中的所有元素数量（即 inputTensorSize），且初始值为从 blob 指向的内存地址开始，连续的 inputTensorSize 个浮点数值。具体来说，这个向量是通过传入指向第一个元素的指针(即 blob)和指向最后一个元素的指针 (即 blob + inputTensorSize) 来初始化的。这相当于将一个指针范围内的数据，转换成一个标准的向量容器。

	这里的 blob 并不是 inputTensorValues 的拷贝，而是指向同一块内存空间的指针，因此在 inputTensorValues 中的每次修改操作也会反映到 blob 中对应的位置上。同时，这也意味着当 blob 所指向的内存空间被释放或修改时，inputTensorValues 的有效性也会随之消失。
	*/
	size_t inputTensorSize = vectorProduct(inputTensorShape);
	std::vector<float> inputTensorValues(blob, blob + inputTensorSize);

	std::vector<Ort::Value> inputTensors;
	/*
	这行代码是使用 ONNX Runtime 库中的 Ort::MemoryInfo 类创建内存信息对象，用于存储输入张量数据的内存信息。具体而言，指定了该内存信息对象使用 CPU 内存，分配方式为 OrtArenaAllocator，使用默认的内存类型 OrtMemTypeDefault。
	Ort::MemoryInfo 类表示 ONNX Runtime 引擎中的内存信息对象，用于管理内存池、内存块等资源。通过调用 CreateCpu 静态函数创建内存信息对象，传入的两个参数分别是分配方式和内存类型。
	其中，OrtArenaAllocator 表示要在一块特定的内存池（arena）中分配内存，以避免多次分配小块内存的开销，从而提高内存访问性能；OrtMemTypeDefault 则表示使用默认的内存类型，即默认的内存地址空间（address space）。
	*/
	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault); // CreateCuda

	/*
	首先，我们调用 Ort::Value::CreateTensor<float> 函数创建一个浮点型的张量对象，并将该对象的内存信息、值数据、元素总数、形状信息数组和形状信息数组长度作为参数传递给该函数。
	其中，memoryInfo 参数表示存储输入张量数据的内存信息对象，即上一行代码创建的 Ort::MemoryInfo 类型变量 memoryInfo；
	inputTensorValues.data() 表示输入张量中的浮点数值数据，即引用传递的浮点型向量变量 inputTensorValues 的底层指针，该向量变量存储从图像像素向量中提取的数值数据；
	inputTensorSize 表示输入张量中元素的总数，即该输入张量上的像素点数，由图像矩阵的长和宽相乘计算得出；inputTensorShape.data() 和 inputTensorShape.size() 分别表示输入张量的形状信息，即一个整数数组，每个元素表示对应维度的大小，以及该数组的长度，即输入张量的阶数，它们分别对应于前面代码中通过读取图像矩阵的长、宽和通道数计算出来的三元素向量 inputTensorShape。
	*/
	inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorSize, inputTensorShape.data(), inputTensorShape.size()));

	/*
	创建了一个 Ort::RunOptions 对象，该对象指定了在运行模型时要使用的选项。在本例中，我们将其设置为 nullptr，表示使用默认选项。
	调用 this->session.Run 方法运行模型，并将输入张量集合、输出张量名字数组以及输出张量个数作为参数传递给该方法。在本例中，输入张量共有 1 个，输出张量共有 1 个，因此我们传递了值为 1 的整数作为输入和输出张量的数量参数。
	inputNames.data() 和 outputNames.data() 分别是输入张量名字数组和输出张量名字数组，它们通过 std::vector<std::string> 类型的变量 inputNames 和 outputNames 存储输入和输出张量的名字。在本例中，我们只使用了一个输入张量和一个输出张量，因此我们创建了包含单个字符串的向量来存储它们的名称。
	将运行结果以 std::vector<Ort::Value> 类型的对象 outputTensors 返回，其中存储了模型计算结果的输出张量。
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
	image：输入图像，类型为 cv::Mat &。
	blob：存储预处理后的图像数据的一维数组指针，类型为 float *&。注意，这里使用了 & 和 *，表示该参数是一个指针引用，可以修改其指向的内存地址，并且在函数外部也能够访问到修改后的值。
	inputTensorShape：输入张量的形状，类型为 std::vector<int64_t> &。
	*/
	cv::Mat resizedImage, floatImage; // 尺寸调整，类型转换
	cv::cvtColor(image, resizedImage, cv::COLOR_BGR2RGB);

	letterbox(resizedImage, resizedImage, this->inputImageShape, cv::Scalar(114, 114, 114), this->isDynamicInputShape, true, 32);

	// 更新输入张量的高度和宽度，将其设为调整后的 resizedImage 的高度和宽度,rows是高度，cols是宽度。
	inputTensorShape[2] = resizedImage.rows;
	inputTensorShape[3] = resizedImage.cols;

	// 使用 convertTo 函数将调整后的 resizedImage 数据从 uchar 类型转换为 float 类型，并且进行归一化处理，将像素值缩放到 [0, 1] 范围内
	resizedImage.convertTo(floatImage, CV_32FC3, 1 / 255.0);
	// 使用 new 运算符在堆上分配一个大小为 floatImage.cols * floatImage.rows * floatImage.channels() 的一维数组，并将其返回的指针赋给 blob 变量，用于存储预处理后的图像数据
	blob = new float[floatImage.cols * floatImage.rows * floatImage.channels()];

	cv::Size floatImageSize{ floatImage.cols,floatImage.rows };
	/*
	将 floatImage 中的数据按照通道拆分成三个 cv::Mat 类型的矩阵，并将它们存储到 blob 中。具体地，首先定义一个 vector 类型的变量 chw，其中元素个数等于 floatImage 的通道数，即 3。然后使用 cv::Mat 构造函数为每个元素创建一个大小为 floatImageSize 的单通道 cv::Mat 矩阵，并将其数据指针设为 blob + i * floatImageSize.width * floatImageSize.height，用于存储 floatImage 第 i 个通道的数据。最后，调用 cv::split 函数将 floatImage 中的数据按照通道拆分到 chw 矩阵中，并将 chw 中的数据存储到 blob 中。
	还有一种方式可以通过 cv 中的 merge 实现，但是split也足够高效。
	*/
	std::vector<cv::Mat> chw(floatImage.channels());
	for (int i = 0; i < floatImage.channels(); ++i) {
		chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
	}
	cv::split(floatImage, chw);
}

std::vector<Detection> Yolov5ORT::postprocessing(const cv::Size resizedImageShape, const cv::Size originalImageShape, std::vector<Ort::Value>& outputTensors, const float confThreshold, const float iouThreshold) {
	std::vector<cv::Rect> boxes; // 检测框
	std::vector<float> confs; // 置信度
	std::vector<int> classIds; // 类别标签

	/*
	这行代码是用来获取模型输出张量的数据指针的。首先通过索引 0 从 outputTensors 中获取第一个输出张量，然后使用 GetTensorData() 方法获取它的数据指针。	这里的 auto* 是一个指向 float 类型的指针.
	rawOutput 指针可以用于访问输出张量中存储的原始数据。在本代码中，我们后续会将其转换为 std::vector<float> 类型的对象 output，以便更方便地处理输出数据。
	*/
	auto* rawOutput = outputTensors[0].GetTensorData<float>();
	/*
	这行代码是用来获取模型输出张量的形状信息的。首先通过索引 0 从 outputTensors 中获取第一个输出张量，然后使用 GetTensorTypeAndShapeInfo() 方法获取它的类型和形状信息，最后使用 GetShape() 方法获取输出张量的形状信息。
	这里的 std::vector<int64_t> 是一个存储 int64_t 类型元素的动态数组，被赋值为 outputTensors[0].GetTensorTypeAndShapeInfo().GetShape() 的返回值。可以把它看作一个数组，其中 outputShape[i] 表示输出张量在第 i 维上的大小。
	由于在目标检测模型中，输出张量的形状和含义都是预定义好的，因此在这里我们可以通过解析输出张量的形状信息，计算出各个元素的位置，从而方便地访问输出数据。
	*/
	std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
	//这行代码是获取模型输出张量中元素的总数目。
	size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
	/*
	这行代码是将输出张量的原始数据转换为 std::vector<float> 对象的代码。使用 std::vector 类的构造函数，将 rawOutput 指针指向的输出张量数据转换为 std::vector 类型对象 output。
	这里的 std::vector<float> 是一个存储 float 类型元素的动态数组，被赋值为 std::vector<float>(rawOutput, rawOutput + count) 的返回值。这个构造函数的第一个参数是指向数据的指针，第二个参数是指向数据结束位置的指针。它会将指定范围内的数据拷贝到 std::vector 对象中。
	*/
	std::vector<float> output(rawOutput, rawOutput + count);

	/*
	outputShape 是一个存储输出张量形状信息的 std::vector<int64_t> 对象，它是一个一维数组，表示输出张量的各个维度大小。
	具体来说，输出张量形状一般为 [batch_size, num_anchors_per_cell * grid_size * grid_size, num_classes + 5]，包含三个维度。其中：
	第 1 维度 batch_size 表示输入数据中的样本数量（也称批次大小）。
	第 2 维度 num_anchors_per_cell * grid_size * grid_size 表示在每个格子上预测多少个边界框（即锚框的数量），乘以网格大小的平方，即可得到总的边界框数量。
	第 3 维度 num_classes + 5 表示每个边界框的预测结果，其中前 4 个元素是边界框的坐标信息，第 5 个元素是置信度（confidence），后面的 num_classes 个元素是代表不同类别的概率。first 5 elements are box[4] and obj confidence
	*/
	int numClasses = (int)outputShape[2] - 5;
	int elementsInBatch = int(outputShape[1] * outputShape[2]);

	/*
	从目标检测模型的输出张量中解析出概率最高的边界框，并将其坐标、置信度及类别信息存储在向量 boxes、confs 和 classIds 中。
	具体来说，代码中使用一个循环遍历张量中的每个元素。输出是个一维数组，由于每个元素对应着一个边界框，因此每次循环需要跳过张量中的一定数量的元素，这里跳过的数量为 outputShape[2]。循环中先读取第 5 个元素，即置信度，如果超过设定的阈值 confThreshold，则说明该边界框可能包含对象，可以进行后续处理。
	在读取边界框信息时，首先读取前 4 个元素，即边界框的中心点坐标和宽高，然后计算出左上角的坐标。接着调用函数 getBestClassInfo，从其中读取关于该边界框的类别概率信息，并将其与置信度相乘得到该边界框的最终置信度 confidence。最后将该边界框的信息（坐标、置信度、类别）存储在向量 boxes、confs 和 classIds 中。
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
	这段代码使用 OpenCV 的 NMSBoxes 函数对预测的边界框进行非极大值抑制（NMS）处理，以消除重叠的边界框，并保留置信度最高的那一个。
	具体来说，函数接收 5 个参数：boxes 是一个向量，存储了所有预测边界框的坐标信息；confs 是一个向量，存储了所有预测边界框的置信度信息；confThreshold 是阈值，用于过滤掉置信度低于它的边界框；iouThreshold 是 IOU 阈值，用于设置重叠度超过该值的边界框被视为同一个物体；indices 是一个输出参数，存储经过 NMS 处理后留下的边界框的索引。
	在函数内部，首先将 boxes 和 confs 中每一个元素打包成一个 cv::Rect 和一个浮点数，并存储到 rects 向量中。然后调用 cv::dnn::NMSBoxes 函数进行 NMS 处理，并将留下的边界框的索引存储到 indices 向量中。
	*/
	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confs, confThreshold, iouThreshold, indices);

	// 将目标检测的坐标还原到图上
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
	计算缩放比例 r，即将输入图像缩放到指定大小需要乘以的系数。首先，将输入图像的高宽与指定大小比较，并选择其中较小的一项作为缩放系数，保存到变量 r 中。如果 scaleUp 参数为 false，则将 r 再次与 1.0f 取最小值，确保不会对原图进行放大操作。
	*/
	float r = std::min((float)newShape.height / (float)shape.height,
		(float)newShape.width / (float)shape.width); // 和640计算最小值，按照最长边进行缩放

	if (!scaleUp)
		r = std::min(r, 1.0f); // 缩放

	//float ratio[2]{ r,r };
	/*
	根据新图像大小和缩放后的原图像大小计算出要添加的上下左右边框宽度 dw 和 dh。如果启用了自动对齐（即 auto_ 为 true），则将 dw 和 dh 调整为 stride 的倍数，以确保可被步长整除
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
	根据边框宽度和颜色使用 cv::copyMakeBorder 函数在图像周围添加边框，并将结果保存到 outImage 中。需要注意的是，在计算上下左右边框宽度时，使用了 std::round 函数进行四舍五入操作，并且减去了一个很小的数（0.1f），以避免 cv::copyMakeBorder 函数出现问题。
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
	这段代码实现了在目标检测模型输出张量中查找预测边界框对应的最佳类别及其置信度的功能。
	具体来说，函数接收三个参数：一个迭代器 it，它指向输出张量中一个预测边界框的起始位置；一个整数 numClasses，表示类别数量；以及两个引用参数 bestConf 和 bestClassId，函数将计算出的最佳类别的置信度和编号分别赋值给这两个参数。
	函数首先将初始值设置为默认值，即最佳类别 ID 为 5（因为前 5 个元素分别是边界框和对象置信度），最佳置信度为 0。然后在从第 6 个元素到第 numClasses + 5 个元素中遍历，对于每个类别，如果它的概率值大于当前最佳概率值 bestConf，则将其更新为当前概率值，并将最佳类别 ID 设为该类别的 ID。
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
	// 经过模型预测和nms之后的坐标恢复,这块和预处理保持一致
	// 输入图像与源图像在横向和纵向上的缩放比例，然后取两个比例中的较小值作为 gain 的值
	float gain = std::min((float)imageShape.height / (float)imageOriginalShape.height,
		(float)imageShape.width / (float)imageOriginalShape.width);

	// 计算缩放后水平和竖直方向的padding值
	int pad[2] = { (int)(((float)imageShape.width - (float)imageOriginalShape.width * gain) / 2.0f),
		(int)(((float)imageShape.height - (float)imageOriginalShape.height * gain) / 2.0f) };

	// 根据计算得到的缩放系数和填充值，重新计算并更新了目标物体的坐标信息,可以确保目标物体在原始图像上的位置信息得到准确还原，并且不受输入图像尺寸的影响。
	coords.x = (int)std::round(((float)(coords.x - pad[0]) / gain));//pad是原图转到目标尺寸时加的，现在从目标尺寸王原图转时要减掉，之后在除gain去还原
	coords.y = (int)std::round(((float)(coords.y - pad[1]) / gain));
	coords.width = (int)std::round(((float)coords.width / gain));
	coords.height = (int)std::round(((float)coords.height / gain));
}

std::wstring Yolov5ORT::charToWstring(const char* str) {
	/*
	*表示指针，在这个函数中，const char* str 表示一个指向 char 类型的常量指针。也就是说，str 变量存储一个指向 char 类型数据的内存地址，且该地址不可以被修改（因为使用了 const 限定符）
	该函数的功能是将 C 风格字符串转换为 std::wstring 类型的字符串。

	具体实现过程如下：
	定义一个 typedef，用于指定字符串编码方式。这里使用了 codecvt_utf8<wchar_t> 类型，表示将 UTF-8 编码的字符串转换为宽字符编码（wchar_t）的字符串。
	创建一个 wstring_convert 对象 converter，用于进行字符串类型转换。
	使用 converter.from_bytes(str) 方法将输入的 C 风格字符串 str 转换为 std::wstring 字符串，并返回该字符串。在转换过程中，from_bytes() 方法会按照 converter 对象指定的编码方式将输入字符串转换为宽字符编码的字符串。
	需要注意的是，该函数实现依赖于 C++11 及以上版本提供的 std::wstring_convert 类和 codecvt_utf8<wchar_t> 类型。这些类型可以在头文件 <codecvt> 中找到。
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
		while (std::getline(infile, line)) { // 读取文件中的每一行
			if (line.back() == '\r')  //如果该行以回车符 \r 结尾，则将其移除。这是因为 Windows 系统下的文本文件中，行末通常会包含 \r\n（即回车换行），而在 Linux 等系统中，则只包含 \n。
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

