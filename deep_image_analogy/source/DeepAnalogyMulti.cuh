#ifndef DEEPANALOGY_H
#define DEEPANALOGY_H
#include <string>
#include "opencv2/opencv.hpp"
#include <iostream>
#include <fstream>
#include "Classifier.h"

using namespace std;
class DeepAnalogyMulti
{
public:

	DeepAnalogyMulti();
	~DeepAnalogyMulti();
	
	void SetRatio(float ratio = 0.5);
	void SetBlendWeight(int level = 3);
	void UsePhotoTransfer(bool flag = false);
	void SetModel(string path);
	void SetA(string list_c);
	void SetBPrime(string list_s);
	void SetOutputDir(string dir_o);
	void SetGPU(int no);
	void LoadInputs();
	void ComputeAnn();
	void ReadFilePath();
	void CheckImageSize(std::vector<cv::Mat> &ori_M,std::vector<cv::Mat> &img_M,float &R);
	void MakeOutputDir();

private:
	float resizeRatio, weightBi;
	int weightLevel, ori_A_rows, ori_A_cols, ori_BP_cols, ori_BP_rows, cur_A_rows, cur_A_cols, cur_BP_rows, cur_BP_cols;
	bool photoTransfer;
	string path_model, path_output, list_A, list_BP;
	std::vector<cv::Mat> img_AL_set, img_BPL_set;
	std::vector<std::string> file_A_set, file_BP_set, name_A_set, name_BP_set;
	string path_result_AB, path_result_BA, path_refine_AB, path_refine_BA;
};

#endif