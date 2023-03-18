#include "prompt_slover.h"

PromptSlover::PromptSlover() {
	// load clip model
	net.opt.use_vulkan_compute = false;
	net.opt.use_winograd_convolution = false;
	net.opt.use_sgemm_convolution = false;
	net.opt.use_fp16_packed = true;
	net.opt.use_fp16_storage = true;
	net.opt.use_fp16_arithmetic = true;
	net.opt.use_packing_layout = true;

	net.load_param("weights/sd/FrozenCLIPEmbedder-fp16.param"); // clip
	net.load_model("weights/sd/FrozenCLIPEmbedder-fp16.bin"); // 模型转换用的pnnx，不是onnx，通过pnnx再转成ncnn的模型结构数据

	// read tokenizer dict
	std::ifstream infile;
	std::string pathname = "weights/sd/vocab.txt";
	infile.open(pathname.data());
	std::string s;
	int idx = 0;
	while (getline(infile, s)) {
		tokenizer_token2idx.insert(std::pair<std::string, int>(s, idx));
		tokenizer_idx2token.insert(std::pair<int, std::string>(idx, s));
		idx++;
	}
	infile.close();
}

ncnn::Mat PromptSlover::get_conditioning(std::string& prompt) {
	// 重要度计算可以匹配"()"和"[]"，圆括号是加重要度，方括号是减重要度
	std::vector<std::pair<std::string, float>> parsed = parse_prompt_attention(prompt);

	// token转ids
	std::vector<std::vector<int>> tokenized;
	for (auto p : parsed) {
		std::vector<std::string> tokens = split(p.first); // 按照单词维度进行切分，空格和逗号作为切分的依据
		std::vector<int> ids;
		for (std::string token : tokens) {
			ids.push_back(tokenizer_token2idx[token]); // 英文的单词转索引
		}
		tokenized.push_back(ids);
	}

	// 一些处理
	std::vector<int> remade_tokens; // 这里是指进行padding之后的特征
	std::vector<float> multipliers; // 这里指特征对应的权重
	int last_comma = -1;
	for (int it_tokenized = 0; it_tokenized < tokenized.size(); it_tokenized++) {
		std::vector<int> tokens = tokenized[it_tokenized]; // tokens是编码之后单个词，词这里是指调整了优先级的小段落
		float weight = parsed[it_tokenized].second;

		int i = 0;
		// 对输入的prompt进行一些筛选
		while (i < tokens.size()) {
			int token = tokens[i];
			if (token == 267) { // 267是，
				last_comma = remade_tokens.size();
			}
			//这里做的和下面做的是一致的
			else if ((max(int(remade_tokens.size()), 1) % 75 == 0) && (last_comma != -1) && (remade_tokens.size() - last_comma <= 20)) {
				last_comma += 1;
				std::vector<int> reloc_tokens(remade_tokens.begin() + last_comma, remade_tokens.end());
				std::vector<float> reloc_mults(multipliers.begin() + last_comma, multipliers.end());
				std::vector<int> _remade_tokens(remade_tokens.begin(), remade_tokens.begin() + last_comma);
				remade_tokens = _remade_tokens;
				int length = remade_tokens.size();
				int rem = ceil(length / 75.0) * 75 - length; // 
				std::vector<int> tmp_token(rem, 49407); // 49407是vocat中的embedding单词数量
				remade_tokens.insert(remade_tokens.end(), tmp_token.begin(), tmp_token.end());
				remade_tokens.insert(remade_tokens.end(), reloc_tokens.begin(), reloc_tokens.end());
				std::vector<float> _multipliers_(multipliers.begin(), multipliers.end() + last_comma);
				std::vector<int> tmp_multipliers(rem, 1.0f);
				_multipliers_.insert(_multipliers_.end(), tmp_multipliers.begin(), tmp_multipliers.end());
				_multipliers_.insert(_multipliers_.end(), reloc_mults.begin(), reloc_mults.end());
				multipliers = _multipliers_;

			}
			remade_tokens.push_back(token);
			multipliers.push_back(weight);
			i += 1;
		}
	}
	//std::cout << int(remade_tokens.size()) << std::endl;
	//std::cout << ceil(max(int(remade_tokens.size()), 1) / 75.0) << std::endl;
	//std::cout << max(int(remade_tokens.size()), 1) << std::endl;
	// prompt是要做padding对齐的，进入clip的token一定是1个开始符+75个词+1个结束符=77个符，输入clip的weight维度是49408x768,data维度是77x768
	int prompt_target_length = ceil(max(int(remade_tokens.size()), 1) / 75.0) * 75; // 1*75=75
	int tokens_to_add = prompt_target_length - remade_tokens.size(); // 不够75的，初始化一个输入矩阵进行填充
	std::vector<int> tmp_token(tokens_to_add, 49407);
	remade_tokens.insert(remade_tokens.end(), tmp_token.begin(), tmp_token.end());
	std::vector<int> tmp_multipliers(tokens_to_add, 1.0f);
	multipliers.insert(multipliers.end(), tmp_multipliers.begin(), tmp_multipliers.end());

	// remade_tokens是做了对齐的prompt,multiplier是做了对齐的weights
	ncnn::Mat conds(768, 0);
	while (remade_tokens.size() > 0) {
		std::vector<int> rem_tokens(remade_tokens.begin() + 75, remade_tokens.end()); // 从end取到end
		std::vector<float> rem_multipliers(multipliers.begin() + 75, multipliers.end());

		std::vector<int> current_tokens; // 相当于只取了75个输入的current_tokens/multipliers
		std::vector<float> current_multipliers;
		// remade_tokens中有值就按照第一种取，无值就设定一个值
		if (remade_tokens.size() > 0) {
			current_tokens.insert(current_tokens.end(), remade_tokens.begin(), remade_tokens.begin() + 75);
			current_multipliers.insert(current_multipliers.end(), multipliers.begin(), multipliers.begin() + 75);
		}
		else {
			std::vector<int> tmp_token(75, 49407);
			current_tokens.insert(current_tokens.end(), tmp_token.begin(), tmp_token.end());
			std::vector<int> tmp_multipliers(75, 1.0f);
			current_multipliers.insert(current_multipliers.end(), tmp_multipliers.begin(), tmp_multipliers.end());
		}

		ncnn::Mat token_mat = ncnn::Mat(77); // 75+2
		token_mat.fill(int(49406));
		ncnn::Mat multiplier_mat = ncnn::Mat(77);
		multiplier_mat.fill(1.0f); // 初始化ncnn:Mat为全1

		int* token_ptr = token_mat; // 转成指针,这个指针是77维度
		float* multiplier_ptr = multiplier_mat;
		for (int i = 0; i < 75; i++) {
			token_ptr[i + 1] = int(current_tokens[i]); // current_tokens是75维度的
			multiplier_ptr[i + 1] = current_multipliers[i];
		}

		ncnn::Extractor ex = net.create_extractor(); // 模型载入之后，新建一个EXtractor，设置输入，获取输出
		ex.set_light_mode(true);
		ex.input("token", token_mat);
		ex.input("multiplier", multiplier_mat);
		ex.input("cond", conds);

		// 模型出来的condition特征最后拼接起来，总的特征shape将会是(77*i,768)，第一个维度是77的维度
		ncnn::Mat new_conds;
		ex.extract("conds", new_conds);
		conds = new_conds;

		remade_tokens = rem_tokens;
		multipliers = rem_multipliers;
	}
	return conds;
}



std::vector<std::pair<std::string, float>> PromptSlover::parse_prompt_attention(std::string& texts) {
	std::vector<std::pair<std::string, float>> res; // dict数据
	std::stack<int> round_brackets;
	std::stack<int> square_brackets;
	const float round_bracket_multiplier = 1.1; // 圆括号
	const float square_bracket_multiplier = 1 / 1.1;  // 方括号

	std::vector<std::string> ms;
	// ["floating hair, portrait, ","(","(","loli",")",")"...]
	// 循环字符串，因为字符串会根据(),[]对字符串进行重新拼接，形成vector
	for (char c : texts) {
		std::string s = std::string(1, c);
		if (s == "(" || s == "[" || s == ")" || s == "]") {
			ms.push_back(s); // 次数是添加一个元素
		}
		else {
			if (ms.size() < 1)
				ms.push_back("");
			std::string last = ms[ms.size() - 1];
			if (last == "(" || last == "[" || last == ")" || last == "]")
				ms.push_back("");
			ms[ms.size() - 1] += s; // 此处是添加字符
		}
	}

	// round_brackets、square_brackets中是存储res.size()，只是一个占位符
	// res是存储prompt
	for (std::string text : ms) {
		if (text == "(") {
			round_brackets.push(res.size());
		}
		else if (text == "[") {
			square_brackets.push(res.size());
		}
		else if (text == ")" && round_brackets.size() > 0) {
			for (int p = round_brackets.top(); p < res.size(); p++) {
				res[p].second *= round_bracket_multiplier; // second是键对应值，圆括号中值降权 ("loli",1.21;"one girl",1.0)
			}
			round_brackets.pop(); // res计算完权重就弹出
		}
		else if (text == "]" and square_brackets.size() > 0) {
			for (int p = square_brackets.top(); p < res.size(); p++) {
				res[p].second *= square_bracket_multiplier;
			}
			square_brackets.pop();
		}
		else {
			res.push_back(std::make_pair(text, 1.0));
		}
	}

	// 一些过滤逻辑
	while (!round_brackets.empty()) {
		for (int p = round_brackets.top(); p < res.size(); p++) {
			res[p].second *= round_bracket_multiplier;
		}
		round_brackets.pop();
	}

	while (!square_brackets.empty()) {
		for (int p = square_brackets.top(); p < res.size(); p++) {
			res[p].second *= square_bracket_multiplier;
		}
		square_brackets.pop();
	}

	int i = 0;
	while (i + 1 < res.size()) {
		if (res[i].second == res[i + 1].second) {
			res[i].first += res[i + 1].first;
			auto it = res.begin();
			res.erase(it + i + 1);
		}
		else {
			i += 1;
		}
	}
	return res;
}


std::vector<std::string> PromptSlover::split(std::string str) {
	std::string::size_type pos;
	std::vector<std::string> result;
	str += " ";
	int size = str.size();
	for (int i = 0; i < size; i++) {
		pos = min(str.find(" ", i), str.find(",", i));
		if (pos < size) {
			std::string s = str.substr(i, pos - i);
			std::string pat = std::string(1, str[pos]);
			if (s.length() > 0)
				result.push_back(s + "</w>");
			if (pat != " ")
				result.push_back(pat + "</w>");
			i = pos;
		}
	}
	return result;
}



