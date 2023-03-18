#include "diffusion_slover.h"

DiffusionSlover::DiffusionSlover(int h, int w, int mode) {
	net.opt.use_vulkan_compute = false; // ncnn����gpu������ʹ��vulkan
	net.opt.lightmode = true;
	if (mode = 0) {
		net.opt.use_winograd_convolution = false; // ����ļ��ٷ���
		net.opt.use_sgemm_convolution = false;
	}
	else {
		net.opt.use_winograd_convolution = true;
		net.opt.use_sgemm_convolution = true;
	}
	net.opt.use_fp16_packed = true; // fp16����
	net.opt.use_fp16_storage = true;
	net.opt.use_fp16_arithmetic = true;
	net.opt.use_packing_layout = true; // �������ڲ��ƶ�ʱ����pack4���ڴ沼�֣��ӿ��ƶ��ٶȣ����ᵼ��Ȩ�ز�����Ӧ���ţ������Ҫ������ڴ�

	// diffussion��ģ�ͷǳ���fp16Ҳ��1.6g��fp32ģ��Ҳ��4g������onnx�ĵ�����һ�����У��ߵ���pnnx·��
	if (h == 512 && w == 512) // �˴��������ͼ�Ŀ�͸�
		net.load_param("weights/sd/UNetModel-512-512-MHA-fp16-opt.param");
	else if (h == 256 && w == 256)
		net.load_param("weights/sd/UNetModel-256-256-MHA-fp16-opt.param");
	else {
		generate_param(h, w); // �������256x256����512x512������������һ���µ�Ȩ���ļ�������
		net.load_param(("assets/sd/tmp-UNetModel-" + std::to_string(h) + "-" + std::to_string(w) + "-MHA-fp16.param").c_str());
	}
	net.load_param("weights/sd/UNetModel-MHA-fp16.bin");

	h_size = h / 8;
	w_size = w / 8;
	ifstream in("weights/sd/log_sigmas.bin", ios::in | ios::binary);
	in.read((char*)& log_sigmas, sizeof log_sigmas);
	in.close();
}

void DiffusionSlover::generate_param(int height, int width)
{
	std::string line;
	ifstream diffuser_file("assets/UNetModel-base-MHA-fp16.param");
	ofstream diffuser_file_new("assets/tmp-UNetModel-" + std::to_string(height) + "-" + std::to_string(width) + "-MHA-fp16.param");

	int cnt = 0;
	while (getline(diffuser_file, line))
	{
		if (line.substr(0, 7) == "Reshape")
		{
			switch (cnt)
			{
			case 0: line = line.substr(0, line.size() - 4) + to_string(width * height / 8 / 8); break;
			case 1: line = line.substr(0, line.size() - 7) + to_string(width / 8) + " 2=" + std::to_string(height / 8); break;
			case 2: line = line.substr(0, line.size() - 4) + to_string(width * height / 8 / 8); break;
			case 3: line = line.substr(0, line.size() - 7) + to_string(width / 8) + " 2=" + std::to_string(height / 8); break;
			case 4: line = line.substr(0, line.size() - 4) + to_string(width * height / 2 / 2 / 8 / 8); break;
			case 5: line = line.substr(0, line.size() - 7) + to_string(width / 2 / 8) + " 2=" + std::to_string(height / 2 / 8); break;
			case 6: line = line.substr(0, line.size() - 4) + to_string(width * height / 2 / 2 / 8 / 8); break;
			case 7: line = line.substr(0, line.size() - 7) + to_string(width / 2 / 8) + " 2=" + std::to_string(height / 2 / 8); break;
			case 8: line = line.substr(0, line.size() - 3) + to_string(width * height / 4 / 4 / 8 / 8); break;
			case 9: line = line.substr(0, line.size() - 7) + to_string(width / 4 / 8) + " 2=" + std::to_string(height / 4 / 8); break;
			case 10: line = line.substr(0, line.size() - 3) + to_string(width * height / 4 / 4 / 8 / 8); break;
			case 11: line = line.substr(0, line.size() - 7) + to_string(width / 4 / 8) + " 2=" + std::to_string(height / 4 / 8); break;
			case 12: line = line.substr(0, line.size() - 2) + to_string(width * height / 8 / 8 / 8 / 8); break;
			case 13: line = line.substr(0, line.size() - 5) + to_string(width / 8 / 8) + " 2=" + std::to_string(height / 8 / 8); break;
			case 14: line = line.substr(0, line.size() - 3) + to_string(width * height / 4 / 4 / 8 / 8); break;
			case 15: line = line.substr(0, line.size() - 7) + to_string(width / 4 / 8) + " 2=" + std::to_string(height / 4 / 8); break;
			case 16: line = line.substr(0, line.size() - 3) + to_string(width * height / 4 / 4 / 8 / 8); break;
			case 17: line = line.substr(0, line.size() - 7) + to_string(width / 4 / 8) + " 2=" + std::to_string(height / 4 / 8); break;
			case 18: line = line.substr(0, line.size() - 3) + to_string(width * height / 4 / 4 / 8 / 8); break;
			case 19: line = line.substr(0, line.size() - 7) + to_string(width / 4 / 8) + " 2=" + std::to_string(height / 4 / 8); break;
			case 20: line = line.substr(0, line.size() - 4) + to_string(width * height / 2 / 2 / 8 / 8); break;
			case 21: line = line.substr(0, line.size() - 7) + to_string(width / 2 / 8) + " 2=" + std::to_string(height / 2 / 8); break;
			case 22: line = line.substr(0, line.size() - 4) + to_string(width * height / 2 / 2 / 8 / 8); break;
			case 23: line = line.substr(0, line.size() - 7) + to_string(width / 2 / 8) + " 2=" + std::to_string(height / 2 / 8); break;
			case 24: line = line.substr(0, line.size() - 4) + to_string(width * height / 2 / 2 / 8 / 8); break;
			case 25: line = line.substr(0, line.size() - 7) + to_string(width / 2 / 8) + " 2=" + std::to_string(height / 2 / 8); break;
			case 26: line = line.substr(0, line.size() - 4) + to_string(width * height / 8 / 8); break;
			case 27: line = line.substr(0, line.size() - 7) + to_string(width / 8) + " 2=" + std::to_string(height / 8); break;
			case 28: line = line.substr(0, line.size() - 4) + to_string(width * height / 8 / 8); break;
			case 29: line = line.substr(0, line.size() - 7) + to_string(width / 8) + " 2=" + std::to_string(height / 8); break;
			case 30: line = line.substr(0, line.size() - 4) + to_string(width * height / 8 / 8); break;
			case 31: line = line.substr(0, line.size() - 7) + to_string(width / 8) + " 2=" + std::to_string(height / 8); break;
			default: break;
			}

			cnt++;
		}
		diffuser_file_new << line << endl;
	}
	diffuser_file_new.close();
	diffuser_file.close();
}

ncnn::Mat DiffusionSlover::randn_4(int seed) {
	cv::Mat cv_x(cv::Size(w_size, h_size), CV_32FC4);
	// RNG��opencv��c++����������������ɲ���һ��64λ��int�������Ŀǰ�ɰ����ȷֲ��͸�˹�ֲ����������
	cv::RNG rng(seed);
	rng.fill(cv_x, cv::RNG::NORMAL, 0, 1);
	ncnn::Mat x_mat(w_size, h_size, 4, (void*)cv_x.data);
	return x_mat.clone();
}

ncnn::Mat DiffusionSlover::CFGDenoiser_CompVisDenoiser(ncnn::Mat& input, float sigma, ncnn::Mat cond, ncnn::Mat uncond) {
	// get_scalings
	float c_out = -1.0 * sigma;
	float c_in = 1.0 / sqrt(sigma * sigma + 1);

	// sigma to t
	float log_sigma = log(sigma);
	std::vector<float> dists(1000);
	for (int i = 0; i < 1000; i++) {
		if (log_sigma - log_sigmas[i] >= 0)
			dists[i] = 1;
		else
			dists[i] = 0;
		if (i == 0) continue;
		dists[i] += dists[i - 1];
	}
	int low_idx = min(int(max_element(dists.begin(), dists.end()) - dists.begin()), 1000 - 2);
	int high_idx = low_idx + 1;
	float low = log_sigmas[low_idx];
	float high = log_sigmas[high_idx];
	float w = (low - log_sigma) / (low - high);
	w = max(0.f, min(1.f, w));
	float t = (1 - w) * low_idx + w * high_idx;

	ncnn::Mat t_mat(1);
	t_mat[0] = t;

	ncnn::Mat c_in_mat(1);
	c_in_mat[0] = c_in;

	ncnn::Mat c_out_mat(1);
	c_out_mat[0] = c_out;

	ncnn::Mat denoised_cond;
	{
		ncnn::Extractor ex = net.create_extractor();  // ��������ִ����
		ex.set_light_mode(true); // �Ƿ���������ģʽ
		ex.input("in0", input); // ��������blob������
		ex.input("in1", t_mat);
		ex.input("in2", cond);
		ex.input("c_in", c_in_mat);
		ex.input("c_out", c_out_mat); //
		ex.extract("output", denoised_cond); // �������������
	}


	ncnn::Mat denoised_uncond;
	{
		ncnn::Extractor ex = net.create_extractor();
		ex.set_light_mode(true);
		ex.input("in0", input);
		ex.input("in1", t_mat);
		ex.input("in2", uncond);
		ex.input("c_in", c_in_mat);
		ex.input("c_out", c_out_mat);
		ex.extract("output", denoised_uncond);
	}


	for (int c = 0; c < 4; c++) {
		float* u_ptr = denoised_uncond.channel(c);
		float* c_ptr = denoised_cond.channel(c);
		for (int hw = 0; hw < h_size * w_size; hw++) {
			(*u_ptr) = (*u_ptr) + 7 * ((*c_ptr) - (*u_ptr)); //  7Ӧ����scale
			u_ptr++;
			c_ptr++;
		}
	}
	return denoised_uncond;
}


ncnn::Mat DiffusionSlover::sampler(int seed, int step, ncnn::Mat& c, ncnn::Mat& uc) {
	// diffusion�������������������£�ʹ��opencv��������������
	ncnn::Mat x_mat = randn_4(seed % 1000);

	// ��������
	// sampling method/diffusion sampler��ɢ������������������ͼ������ж�ͼ�����ȥ��ķ�����
	// ���ڲ�ͬ����ɢ�������ڼ���ͼ����һ���ķ�ʽ��ͬ�����������Ҫ��ͬ�ĳ���ʱ��Ͳ��������ɿ��õ�ͼ��
	// �����߽���DDIM���ٶȿ죬10������
	// t to sigma
	std::vector<float> sigma(step); //step:����������������

	float delta = 0.0 - 999.0 / (step - 1); // 999:timesteps
	for (int i = 0; i < step; i++) { // make_ddim_sampling_parameters 
		float t = 999.0 + i * delta;
		int low_idx = floor(t);
		int high_idx = ceil(t);
		float w = t - low_idx;
		sigma[i] = exp((1 - w) * log_sigmas[low_idx] + w * log_sigmas[high_idx]);
	}
	sigma.push_back(0.f);//sigma��step��ͬ����

	float _norm_[4] = { sigma[0],sigma[0],sigma[0],sigma[0] };
	x_mat.substract_mean_normalize(0, _norm_); //sigma��step���

	// eular ancestral
	for (int i = 0; i < sigma.size() - 1; i++) { // ѭ��step
		cout << "step:" << i << "\t\t";

		//double t1 = ncnn::get_current_time();
		ncnn::Mat denoised = CFGDenoiser_CompVisDenoiser(x_mat, sigma[i], c, uc); // ȥ��
		//double t2 = ncnn::get_

		float sigma_up = min(sigma[i + 1], sqrt(sigma[i + 1] * sigma[i + 1] * (sigma[i] * sigma[i] - sigma[i + 1] * sigma[i + 1]) / (sigma[i] * sigma[i])));
		float sigma_down = sqrt(sigma[i + 1] * sigma[i + 1] - sigma_up * sigma_up);

		srand(time(NULL) + i);
		ncnn::Mat randn = randn_4(rand() % 1000);
		for (int c = 0; c < 4; c++) {
			float* x_ptr = x_mat.channel(c);
			float* d_ptr = denoised.channel(c);
			float* r_ptr = randn.channel(c);
			for (int hw = 0; hw < h_size * w_size; hw++) {
				*x_ptr = *x_ptr + ((*x_ptr - *d_ptr) / sigma[i]) * (sigma_down - sigma[i]) + *r_ptr * sigma_up;
				x_ptr++;
				d_ptr++;
				r_ptr++;
			}
		}
	}

	ncnn::Mat y_mat;
	y_mat.clone_from(x_mat);
	return y_mat;
}








