#pragma once

#include<vector>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"gslam/mappoint.h"
#include"gslam/frame.h"

namespace gslam{
class ORBmatcher{
public:
    ORBmatcher(float nnratio=0.6, bool checkOri=true);
    // Computes the Hamming distance between two ORB descriptors
    static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
    int searchByBoW(Frame::Ptr pKF, Frame::Ptr F, std::vector<MapPoint::Ptr> &vpMapPointMatches);
    int searchByProjection(Frame::Ptr F, const vector<MapPoint::Ptr> &vpMapPoints, const float th);
    
public:

    static const int TH_LOW;
    static const int TH_HIGH;
    static const int HISTO_LENGTH;
protected:
    float mfNNratio;
    bool mbCheckOrientation;
};

}//namespace