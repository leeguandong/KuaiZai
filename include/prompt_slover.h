#pragma once
#ifndef PROMPT_SLOVER
#define PROMPT_SLOVER
#include <iostream>
#include <string>
#include <map>
#include <stack>
#include <vector>
#include <fstream>
#include <net.h>
#include <math.h>

//using namespace std;

class PromptSlover {
public:
	PromptSlover();
	ncnn::Mat get_conditioning(std::string& prompt); // 字符串长的时候，传引用更合适
private:
	std::vector<std::string> split(std::string str);
	std::vector<std::pair<std::string, float>> parse_prompt_attention(std::string& texts);

	std::map<std::string, int> tokenizer_token2idx;
	std::map<int, std::string> tokenizer_idx2token;

	ncnn::Net net;
};


#endif // !PROMPT_SLOVER




