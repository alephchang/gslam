#pragma once

#include <fstream>
//#include <boost/timer.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/viz.hpp> 
#include <opencv2/imgproc/imgproc.hpp>

#include "gslam/config.h"
#include "gslam/visual_odometry.h"

int run_vo(int argc, char** argv);
int validate_result(int argc, char** argv);
void testSE3QuatError();