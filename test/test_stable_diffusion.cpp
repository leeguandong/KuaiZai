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

	// height x width，图片分辨率
	height = 256;
	width = 256;
	// Ai推演的算法，一般Euler a，Euler ，DDIM
	mode = 0;
	// 可以理解让AI推演多少步，一般来说超过17基本就能看了，步数越多，画面中的细节就越多，
	// 但需要的时间也就越久，一般20~30是一个比较稳妥的设定。这个数不会改变画面内容，
	// 只会让内容更加精细，比如20的项链就是一个心形钻石，而50的项链还是同样的心形钻石，
	// 只是钻石上会有更加复杂的线条
	step = 15;
	// 随机数种子，AI作画从原理上其实就是用一个随机的噪声图，反推回图像。
	// 但因为计算机里也没有真随机嘛，所以实际上，AI作画的起始噪声，是可以量化为一个种子数的。
	seed = 42;
	// 支持使用圆括号()来提升prompt的重要度，使用方括号[]来降低prompt的重要度
	// 想要的特征点
	positive_prompt = "floating hair, portrait, ((loli)), ((one girl)), cute face, hidden hands, asymmetrical bangs, beautiful detailed eyes, eye shadow, hair ornament, ribbons, bowties, buttons, pleated skirt, (((masterpiece))), ((best quality)), colorful";
	// 不想要的特征点
	negative_prompt = "((part of the head)), ((((mutated hands and fingers)))), deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, blurry, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, Octane renderer, lowres, bad anatomy, bad hands, text";

	// 做一个magic.txt的文本来prompt
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
				case 0:height = stoi(content); // stoi以int形式返回
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
	//conditioning:用来描述模型应该生成图像的文本提示
	//string作为参数传递（引用或者非引用）并不会产生什么不可描述的后果 区别只是，作为引用参数时，
	//直接引用了该对象，对该形参的修改同样会影响实参，而非引用形参则只是实参的一个拷贝，
	//对形参的修改不会影响到实参 但是传参时又是会倾向于作为引用参数传递（不过并不是必须的），
	//这样就能减少对实参的拷贝，尤其是实参字符串很长很长的情况下
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

