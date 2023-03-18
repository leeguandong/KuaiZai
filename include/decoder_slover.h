#pragma once
#ifndef DECODER_SLOVER
#define DECODER_SLOVER
#include <net.h>
#include <fstream>
#include <string>
#include <vector>


class DecoderSlover {
public:
	DecoderSlover(int h, int w);
	ncnn::Mat decode(ncnn::Mat sample);
private:
	void generate_param(int height, int width);
	const float factor[4] = { 5.48998f, 5.48998f, 5.48998f, 5.48998f };
	const float _mean_[3] = { -1.0f, -1.0f, -1.0f };
	const float _norm_[3] = { 127.5f, 127.5f, 127.5f };

	ncnn::Net net;
};

#endif // !DECODER_SLOVER
