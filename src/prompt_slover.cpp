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
	net.load_model("weights/sd/FrozenCLIPEmbedder-fp16.bin"); // ģ��ת���õ�pnnx������onnx��ͨ��pnnx��ת��ncnn��ģ�ͽṹ����

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
	// ��Ҫ�ȼ������ƥ��"()"��"[]"��Բ�����Ǽ���Ҫ�ȣ��������Ǽ���Ҫ��
	std::vector<std::pair<std::string, float>> parsed = parse_prompt_attention(prompt);

	// tokenתids
	std::vector<std::vector<int>> tokenized;
	for (auto p : parsed) {
		std::vector<std::string> tokens = split(p.first); // ���յ���ά�Ƚ����з֣��ո�Ͷ�����Ϊ�зֵ�����
		std::vector<int> ids;
		for (std::string token : tokens) {
			ids.push_back(tokenizer_token2idx[token]); // Ӣ�ĵĵ���ת����
		}
		tokenized.push_back(ids);
	}

	// һЩ����
	std::vector<int> remade_tokens; // ������ָ����padding֮�������
	std::vector<float> multipliers; // ����ָ������Ӧ��Ȩ��
	int last_comma = -1;
	for (int it_tokenized = 0; it_tokenized < tokenized.size(); it_tokenized++) {
		std::vector<int> tokens = tokenized[it_tokenized]; // tokens�Ǳ���֮�󵥸��ʣ���������ָ���������ȼ���С����
		float weight = parsed[it_tokenized].second;

		int i = 0;
		// �������prompt����һЩɸѡ
		while (i < tokens.size()) {
			int token = tokens[i];
			if (token == 267) { // 267�ǣ�
				last_comma = remade_tokens.size();
			}
			//�������ĺ�����������һ�µ�
			else if ((max(int(remade_tokens.size()), 1) % 75 == 0) && (last_comma != -1) && (remade_tokens.size() - last_comma <= 20)) {
				last_comma += 1;
				std::vector<int> reloc_tokens(remade_tokens.begin() + last_comma, remade_tokens.end());
				std::vector<float> reloc_mults(multipliers.begin() + last_comma, multipliers.end());
				std::vector<int> _remade_tokens(remade_tokens.begin(), remade_tokens.begin() + last_comma);
				remade_tokens = _remade_tokens;
				int length = remade_tokens.size();
				int rem = ceil(length / 75.0) * 75 - length; // 
				std::vector<int> tmp_token(rem, 49407); // 49407��vocat�е�embedding��������
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
	// prompt��Ҫ��padding����ģ�����clip��tokenһ����1����ʼ��+75����+1��������=77����������clip��weightά����49408x768,dataά����77x768
	int prompt_target_length = ceil(max(int(remade_tokens.size()), 1) / 75.0) * 75; // 1*75=75
	int tokens_to_add = prompt_target_length - remade_tokens.size(); // ����75�ģ���ʼ��һ���������������
	std::vector<int> tmp_token(tokens_to_add, 49407);
	remade_tokens.insert(remade_tokens.end(), tmp_token.begin(), tmp_token.end());
	std::vector<int> tmp_multipliers(tokens_to_add, 1.0f);
	multipliers.insert(multipliers.end(), tmp_multipliers.begin(), tmp_multipliers.end());

	// remade_tokens�����˶����prompt,multiplier�����˶����weights
	ncnn::Mat conds(768, 0);
	while (remade_tokens.size() > 0) {
		std::vector<int> rem_tokens(remade_tokens.begin() + 75, remade_tokens.end()); // ��endȡ��end
		std::vector<float> rem_multipliers(multipliers.begin() + 75, multipliers.end());

		std::vector<int> current_tokens; // �൱��ֻȡ��75�������current_tokens/multipliers
		std::vector<float> current_multipliers;
		// remade_tokens����ֵ�Ͱ��յ�һ��ȡ����ֵ���趨һ��ֵ
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
		multiplier_mat.fill(1.0f); // ��ʼ��ncnn:MatΪȫ1

		int* token_ptr = token_mat; // ת��ָ��,���ָ����77ά��
		float* multiplier_ptr = multiplier_mat;
		for (int i = 0; i < 75; i++) {
			token_ptr[i + 1] = int(current_tokens[i]); // current_tokens��75ά�ȵ�
			multiplier_ptr[i + 1] = current_multipliers[i];
		}

		ncnn::Extractor ex = net.create_extractor(); // ģ������֮���½�һ��EXtractor���������룬��ȡ���
		ex.set_light_mode(true);
		ex.input("token", token_mat);
		ex.input("multiplier", multiplier_mat);
		ex.input("cond", conds);

		// ģ�ͳ�����condition�������ƴ���������ܵ�����shape������(77*i,768)����һ��ά����77��ά��
		ncnn::Mat new_conds;
		ex.extract("conds", new_conds);
		conds = new_conds;

		remade_tokens = rem_tokens;
		multipliers = rem_multipliers;
	}
	return conds;
}



std::vector<std::pair<std::string, float>> PromptSlover::parse_prompt_attention(std::string& texts) {
	std::vector<std::pair<std::string, float>> res; // dict����
	std::stack<int> round_brackets;
	std::stack<int> square_brackets;
	const float round_bracket_multiplier = 1.1; // Բ����
	const float square_bracket_multiplier = 1 / 1.1;  // ������

	std::vector<std::string> ms;
	// ["floating hair, portrait, ","(","(","loli",")",")"...]
	// ѭ���ַ�������Ϊ�ַ��������(),[]���ַ�����������ƴ�ӣ��γ�vector
	for (char c : texts) {
		std::string s = std::string(1, c);
		if (s == "(" || s == "[" || s == ")" || s == "]") {
			ms.push_back(s); // ���������һ��Ԫ��
		}
		else {
			if (ms.size() < 1)
				ms.push_back("");
			std::string last = ms[ms.size() - 1];
			if (last == "(" || last == "[" || last == ")" || last == "]")
				ms.push_back("");
			ms[ms.size() - 1] += s; // �˴�������ַ�
		}
	}

	// round_brackets��square_brackets���Ǵ洢res.size()��ֻ��һ��ռλ��
	// res�Ǵ洢prompt
	for (std::string text : ms) {
		if (text == "(") {
			round_brackets.push(res.size());
		}
		else if (text == "[") {
			square_brackets.push(res.size());
		}
		else if (text == ")" && round_brackets.size() > 0) {
			for (int p = round_brackets.top(); p < res.size(); p++) {
				res[p].second *= round_bracket_multiplier; // second�Ǽ���Ӧֵ��Բ������ֵ��Ȩ ("loli",1.21;"one girl",1.0)
			}
			round_brackets.pop(); // res������Ȩ�ؾ͵���
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

	// һЩ�����߼�
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



