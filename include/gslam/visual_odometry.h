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

#ifndef VISUALODOMETRY_H
#define VISUALODOMETRY_H

#include "gslam/common_include.h"
#include "gslam/map.h"
#include "ORBextractor.h"

#include <opencv2/features2d/features2d.hpp>
#include <fstream>
namespace gslam 
{
class VisualOdometry
{
public:
    typedef shared_ptr<VisualOdometry> Ptr;
    enum VOState {
        INITIALIZING=-1,
        OK=0,
        LOST
    };
    
    VOState     state_;     // current VO status
    Map::Ptr    map_;       // map with all frames and map points
    
    Frame::Ptr  ref_;       // reference key-frame 
    Frame::Ptr  curr_;      // current frame 
    vector<unsigned long> key_frame_ids_;
    cv::Ptr<ORB_SLAM2::ORBextractor> orb_;  // orb detector and computer 
    
    cv::FlannBasedMatcher   matcher_flann_;     // flann matcher
    vector<MapPoint::Ptr>   match_3dpts_;       // matched 3d points 
    vector<int>             match_2dkp_index_;  // matched 2d pixels (index of kp_curr)
   
    //SE3<double> T_c_w_estimated_;    // the estimated pose of current frame 
    int num_inliers_;        // number of inlier features in icp
    int num_lost_;           // number of lost times
    
    // parameters 
    int num_of_features_;   // number of features
    double scale_factor_;   // scale in image pyramid
    int level_pyramid_;     // number of pyramid levels
    float match_ratio_;     // ratio for selecting  good matches
    int max_num_lost_;      // max number of continuous lost times
    int min_inliers_;       // minimum inliers
    double key_frame_min_rot;   // minimal rotation of two key-frames
    double key_frame_min_trans; // minimal translation of two key-frames
    double map_point_erase_ratio_; // remove map point ratio

    vector<Frame::Ptr> local_key_frames_;
    vector<MapPoint::Ptr> local_map_pts_;
    //log file output
    std::ofstream flog_;
    
public: // functions 
    VisualOdometry();
    ~VisualOdometry();
    
    bool addFrame( Frame::Ptr frame );      // add a new frame 

    bool setLogFile(const std::string& logpath);
    void dumpMapAndKeyFrames();
    
protected:  
    // inner operation 
    void detectAndComputeFeatures();
    void featureMatching();
    int featureMatchingWithRef();
    void poseEstimationPnP(); 
    int poseEstimationOptimization();
    bool trackLocalMap(); //based on the pose estimation, find more match between map and keypoints;
    void optimizeMap();
    
    void addKeyFrame();
    void recordKeyFrameForMapPoint();
    void triangulateForNewKeyFrame();
    void addMapPoints();
    bool checkEstimatedPose(); 
    bool checkKeyFrame();
    
    double getViewAngle( Frame::Ptr frame, MapPoint::Ptr point );

    void validateProjection();
    void optimizePnP(const vector<cv::Point3f>& pts3d, const vector<cv::Point2f>& pts2d, Mat& inliers,
            const Mat& rvec, const Mat& tvec);
    void reInitializeFrame();
    
    void updateLocalKeyFrames();
    void updateLocalMapPoints();
    void searchLocalMapPoints();
    
};
}

#endif // VISUALODOMETRY_H
