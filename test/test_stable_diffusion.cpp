#include <iostream>
#include <string>
#include <fstream>
#include <time.h>
#include "diffusion_slover.h"
#include "prompt_slover.h"
#include "decoder_slover.h"
#include "getmem.h"

using namespace std;

int main() {
	int height, width, mode, step, seed;
	string positive_prompt, negative_prompt;

	// height x width��ͼƬ�ֱ���
	height = 256;
	width = 256;
	// Ai���ݵ��㷨��һ��Euler a��Euler ��DDIM
	mode = 0;
	// ���������AI���ݶ��ٲ���һ����˵����17�������ܿ��ˣ�����Խ�࣬�����е�ϸ�ھ�Խ�࣬
	// ����Ҫ��ʱ��Ҳ��Խ�ã�һ��20~30��һ���Ƚ����׵��趨�����������ı仭�����ݣ�
	// ֻ�������ݸ��Ӿ�ϸ������20����������һ��������ʯ����50����������ͬ����������ʯ��
	// ֻ����ʯ�ϻ��и��Ӹ��ӵ�����
	step = 15;
	// ��������ӣ�AI������ԭ������ʵ������һ�����������ͼ�����ƻ�ͼ��
	// ����Ϊ�������Ҳû������������ʵ���ϣ�AI��������ʼ�������ǿ�������Ϊһ���������ġ�
	seed = 42;
	// ֧��ʹ��Բ����()������prompt����Ҫ�ȣ�ʹ�÷�����[]������prompt����Ҫ��
	// ��Ҫ��������
	positive_prompt = "floating hair, portrait, ((loli)), ((one girl)), cute face, hidden hands, asymmetrical bangs, beautiful detailed eyes, eye shadow, hair ornament, ribbons, bowties, buttons, pleated skirt, (((masterpiece))), ((best quality)), colorful";
	// ����Ҫ��������
	negative_prompt = "((part of the head)), ((((mutated hands and fingers)))), deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, blurry, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, Octane renderer, lowres, bad anatomy, bad hands, text";

	// ��һ��magic.txt���ı���prompt
	ifstream magic;
	magic.open("data/magic.txt");
	if (!magic) {
		cout << "can not find magic.txt, using the default setting" << endl;
	}
	else {
		string content = "";
		int i = 0;
		for (i = 0; i < 7; i++) {
			if (getline(magic, content)) {
				switch (i)
				{
				case 0:height = stoi(content); // stoi��int��ʽ����
				case 1:width = stoi(content);
				case 2:mode = stoi(content);
				case 3:step = stoi(content);
				case 4:seed = stoi(content);
				case 5:positive_prompt = content;
				case 6:negative_prompt = content;
				default:
					break;
				}
			}
			else {
				break;
			}
		}
		if (i != 7) {
			cout << "magic.txt has wrong format, please fix it" << endl;
			return 0;
		}
	}
	if (seed == 0) {
		seed = (unsigned)time(NULL);
	}
	magic.close();

	// stable diffusion
	cout << "----------------[init]--------------------";
	PromptSlover prompt_slover;
	DiffusionSlover diffusion_slover(height, width, mode);
	DecoderSlover decoder_slover(height, width);
	printf(" %.2lfG / %.2lfG\n", getCurrentRSS() / 1024.0 / 1024.0 / 1024.0, getPeakRSS() / 1024.0 / 1024.0 / 1024.0);

	cout << "----------------[prompt]------------------";
	//conditioning:��������ģ��Ӧ������ͼ����ı���ʾ
	//string��Ϊ�������ݣ����û��߷����ã����������ʲô���������ĺ�� ����ֻ�ǣ���Ϊ���ò���ʱ��
	//ֱ�������˸ö��󣬶Ը��βε��޸�ͬ����Ӱ��ʵ�Σ����������β���ֻ��ʵ�ε�һ��������
	//���βε��޸Ĳ���Ӱ�쵽ʵ�� ���Ǵ���ʱ���ǻ���������Ϊ���ò������ݣ����������Ǳ���ģ���
	//�������ܼ��ٶ�ʵ�εĿ�����������ʵ���ַ����ܳ��ܳ��������
	ncnn::Mat cond = prompt_slover.get_conditioning(positive_prompt);
	ncnn::Mat uncond = prompt_slover.get_conditioning(negative_prompt);
	printf(" %.2lfG / %.2lfG\n", getCurrentRSS() / 1024.0 / 1024.0 / 1024.0, getPeakRSS() / 1024.0 / 1024.0 / 1024.0);

	cout << "----------------[diffusion]---------------" << endl;
	ncnn::Mat sample = diffusion_slover.sampler(seed, step, cond, uncond);
	cout << "----------------[diffusion]---------------";
	printf(" %.2lfG / %.2lfG\n", getCurrentRSS() / 1024.0 / 1024.0 / 1024.0, getPeakRSS() / 1024.0 / 1024.0 / 1024.0);

	cout << "----------------[decode]------------------";
	ncnn::Mat x_samples_ddim = decoder_slover.decode(sample);
	printf(" %.2lfG / %.2lfG\n", getCurrentRSS() / 1024.0 / 1024.0 / 1024.0, getPeakRSS() / 1024.0 / 1024.0 / 1024.0);

	cout << "----------------[save]--------------------" << endl;
	cv::Mat image(height, width, CV_8UC3);
	x_samples_ddim.to_pixels(image.data,ncnn::Mat::PIXEL_RGB2BGR);
	cv::imwrite("result_" + std::to_string(step) + "_" + std::to_string(seed) + "_" + std::to_string(height) + "x" + std::to_string(width) + ".png", image);

	cout << "----------------[close]-------------------" << endl;

	return 0;
}

