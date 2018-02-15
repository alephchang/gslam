#pragma once

#include"common_include.h"
#include"frame.h"
#include"map.h"


namespace gslam {

class Optimizer
{
public:
    Optimizer();
    ~Optimizer();
    //optimize the camera poses and the map     
    //@param frames: the camera to be optimized, fix the first one
    //@param map: the local map storing the map points
    static void localBA(vector<unsigned long>& frame_ids, Map::Ptr map);
    static int optId;
    static std::string logPath;
    static int poseOptimization(Frame::Ptr pFrame);
};

}