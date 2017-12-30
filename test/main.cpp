#include"run_vo.h"


int main(int argc, char** argv)
{
	testSE3QuatError();
	//run_vo(argc, argv);
	validate_result(argc, argv);
}