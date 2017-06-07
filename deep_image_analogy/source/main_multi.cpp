#include "DeepAnalogyMulti.cuh"

int main(int argc, char** argv) {

	DeepAnalogyMulti dpm;

	if (argc!=9) {

		string model = "models/";
	
		string A_list = "datasets/content_list.txt";
		string BP_list = "datasets/style_list.txt";
		string output = "output/";

		dpm.SetModel(model);
		dpm.SetA(A_list);
		dpm.SetBPrime(BP_list);
		dpm.SetOutputDir(output);
		dpm.SetGPU(0);
		dpm.SetRatio(0.5);
		dpm.SetBlendWeight(2);
		dpm.UsePhotoTransfer(false);
		dpm.LoadInputs();
		dpm.ComputeAnn();
		
	}
	else{
		dpm.SetModel(argv[1]);
		dpm.SetA(argv[2]);
		dpm.SetBPrime(argv[3]);
		dpm.SetOutputDir(argv[4]);
		dpm.SetGPU(atoi(argv[5]));
		dpm.SetRatio(atof(argv[6]));
		dpm.SetBlendWeight(atoi(argv[7]));
		if (atoi(argv[8]) == 1) {
			dpm.UsePhotoTransfer(true);
		}
		else{
			dpm.UsePhotoTransfer(false);
		}
		dpm.LoadInputs();
		dpm.MakeOutputDir();
		dpm.ComputeAnn();
	}



	return 0;
}
