/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef FRAME_H
#define FRAME_H

#include "gslam/common_include.h"
#include "gslam/camera.h"
#include "gslam/ORBVocabulary.h"
#include "gslam/mappoint.h"
#include "3rdparty/DBoW2/DBoW2/BowVector.h"
#include "3rdparty/DBoW2/DBoW2/FeatureVector.h"

using gslam::MapPoint;
namespace gslam 
{
    
// forward declare 
class Frame
{
public:
    typedef std::shared_ptr<Frame> Ptr;
    unsigned long                  id_;         // id of this frame
    double                         time_stamp_; // when it is recorded
    SE3<double>                    T_c_w_;      // transform from world to camera
    Camera::Ptr                    camera_;     // Pinhole RGBD Camera model 
    Mat                            color_, depth_; // color and depth image 
    std::vector<gslam::MapPoint::Ptr>     vpMapPoints_;  // MapPoints associated to keypoints, NULL pointer if no association.
    std::vector<cv::KeyPoint>      vKeys_;
    cv::Mat                        descriptors_;
    // Bag of Words Vector structures.
    DBoW2::BowVector               BowVec_;
    DBoW2::FeatureVector           featVec_;
    static shared_ptr<ORBVocabulary>  pORBvocab_;
    vector<bool>                   vbOutlier_;
    int                            N_;
    vector<float>                  vInvLevelSigma2_;
    
    
public: // data members 
    Frame();
    Frame( long id, double time_stamp=0, SE3<double> T_c_w=SE3<double>(), Camera::Ptr camera=nullptr, Mat color=Mat(), Mat depth=Mat() );
    ~Frame();
    
    static Frame::Ptr createFrame(); 
    
    // find the depth in depth map
    double findDepth( const cv::KeyPoint& kp );
    
    // Get Camera Center
    Vector3d getCamCenter() const;
    
    void setPose( const SE3<double>& T_c_w );
    
    // check if a point is in this frame 
    bool isInFrame( const Vector3d& pt_world );
    
    vector<size_t> getFeaturesInAera(float x, float y, float r) const;

    bool isInFrustum(MapPoint::Ptr pMp);
    void addMapPoint(MapPoint::Ptr pMp, size_t i);
    void sortMapPoint2d();
    void computeBoW();
    bool isBad(){return false;}
};

}

#endif // FRAME_H
