#pragma once
#ifndef DIFFUSION_SLOVER
#define DIFFUSION_SLOVER

#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <net.h>

using namespace std;

class DiffusionSlover {
public:
	DiffusionSlover(int h, int w, int mode);
	ncnn::Mat sampler(int seed, int step, ncnn::Mat& c, ncnn::Mat& uc);
private:
	void generate_param(int h, int w);

	ncnn::Mat randn_4(int seed);
	ncnn::Mat CFGDenoiser_CompVisDenoiser(ncnn::Mat& input, float sigma, ncnn::Mat cond, ncnn::Mat uncond);

	float log_sigmas[1000] = { 0 };
	ncnn::Net net;
	int h_size, w_size;
};
#endif // !DIFFUSION_SLOVER

