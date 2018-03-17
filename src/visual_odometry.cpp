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

//#include "stdafx.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <algorithm>
#include <boost/timer.hpp>
#include <boost/concept_check.hpp>

#include "gslam/config.h"
#include "gslam/visual_odometry.h"
#include "gslam/g2o_types.h"
#include "gslam/Optimizer.h"
#include "gslam/ORBmatcher.h"
namespace gslam
{

const int TH_HIGH = 100;
const int TH_LOW = 50;
const int HISTO_LENGTH = 30;
VisualOdometry::VisualOdometry() :
    state_ ( INITIALIZING ), ref_ ( nullptr ), curr_ ( nullptr ), map_ ( new Map ), num_lost_ ( 0 ), num_inliers_ ( 0 ), matcher_flann_ ( new cv::flann::LshIndexParams ( 5,10,2 ) )
{
    num_of_features_    = Config::get<int> ( "number_of_features" );
    scale_factor_       = Config::get<double> ( "scale_factor" );
    level_pyramid_      = Config::get<int> ( "level_pyramid" );
    match_ratio_        = Config::get<float> ( "match_ratio" );
    max_num_lost_       = Config::get<float> ( "max_num_lost" );
    min_inliers_        = Config::get<int> ( "min_inliers" );
    key_frame_min_rot   = Config::get<double> ( "keyframe_rotation" );
    key_frame_min_trans = Config::get<double> ( "keyframe_translation" );
    map_point_erase_ratio_ = Config::get<double> ( "map_point_erase_ratio" );
    orb_ = new ORB_SLAM2::ORBextractor(1000,1.2,8,20,7);
}

VisualOdometry::~VisualOdometry()
{
    flog_.close();
}
void VisualOdometry::initialize()
{
    if( curr_->vKeys_.size() < 400) return;
    for ( size_t i=0; i<curr_->vKeys_.size(); i++ )
    {
        double d = curr_->findDepth ( curr_->vKeys_[i] );
        if ( d < 0 ) 
            continue;
        Vector3d p_world = ref_->camera_->pixel2world (
            Vector2d ( curr_->vKeys_[i].pt.x, curr_->vKeys_[i].pt.y ), curr_->T_c_w_, d
        );
        Vector3d n = p_world - ref_->getCamCenter();
        n.normalize();
        MapPoint::Ptr pNewMp= MapPoint::createMapPoint(
            p_world, n, curr_->descriptors_.row(i).clone(), curr_);
        pNewMp->addObservation(curr_, i);
        curr_->addMapPoint(pNewMp, i);
        pNewMp->computeDistinctiveDescriptors();
        pNewMp->updateNormalAndDepth();
        map_->insertMapPoint(pNewMp);
    }
    key_frame_ids_.push_back(curr_->id_);
    map_->insertKeyFrame ( curr_ );
    curr_->computeBoW();
    state_ = OK;
}

bool VisualOdometry::addFrame ( Frame::Ptr frame )
{
    flog_ << "Frame ID: " << frame->id_ << std::endl;
    switch ( state_ )
    {
    case INITIALIZING:
    {
        //state_ = OK;
        curr_ = ref_ = frame;
        // extract features from first frame and add them into map
        detectAndComputeFeatures();
        initialize();
        break;
    }
    case OK:
    {
        curr_ = frame;
        curr_->T_c_w_ = ref_->T_c_w_;
        detectAndComputeFeatures();
        //featureMatching();
        int nmatches = featureMatchingWithRef();
        int ngoodmatches = poseEstimationOptimization();
        flog_ << "inliers: " << ngoodmatches << " of matches: " << nmatches <<endl;
        trackLocalMap();
        if ( checkEstimatedPose() == true ) // a good estimation
        {
            validateProjection(); //for validation
            optimizeMap();
            num_lost_ = 0;
            if ( checkKeyFrame() == true ) // is a key-frame
            {
                //triangulate for key points in key frames
                //triangulateForNewKeyFrame();
                addKeyFrame();
            }
        }
        else // bad estimation due to various reasons
        {
            reInitializeFrame();
            curr_->T_c_w_ = ref_->T_c_w_;
            num_lost_++;
            if ( num_lost_ > max_num_lost_ )
            {
                state_ = LOST;
            }
            flog_ << "==========There are " << map_->map_points_.size() << " map points, "
                << "and " << map_->keyframes_.size() << " key frames" << endl;
            return false;
        }
        break;
    }
    case LOST:
    {
        flog_<<"vo has lost."<<endl;
        break;
    }
    }
    flog_ << "==========There are "<< map_->map_points_.size()<<" map points, "
        <<"and "<< map_->keyframes_.size()<< " key frames"<< endl;
    if(map_->map_points_.empty()){
        state_ = LOST;
        flog_ << "tracking lost because of empty map at frame " << curr_->id_ << endl;
        return false;
    }
    else
        return true;
}

bool VisualOdometry::setLogFile(const std::string& logpath)
{
    flog_.open(logpath, fstream::out);
    return flog_.good();
}

void VisualOdometry::detectAndComputeFeatures()
{
    (*orb_)(curr_->color_, cv::Mat(), curr_->vKeys_, curr_->descriptors_);
    curr_->vpMapPoints_.assign(curr_->vKeys_.size(), nullptr);
    curr_->vbOutlier_.assign(curr_->vKeys_.size(), true);
    curr_->N_ = curr_->vpMapPoints_.size();
    curr_->vInvLevelSigma2_ = orb_->GetInverseScaleSigmaSquares();
    //descriptors_curr_.convertTo(descriptors_curr_, CV_32F);
}

void VisualOdometry::featureMatching()
{
    boost::timer timer;
    vector<cv::DMatch> matches;
    // select the candidates in map 
    Mat desp_map;
    vector<MapPoint::Ptr> candidate;
    for ( auto& allpoints: map_->map_points_ )
    {
        MapPoint::Ptr& p = allpoints.second;
        // check if p in curr frame image 
        if ( curr_->isInFrame(p->pos_) )
        {
            // add to candidate 
            p->visible_times_++;
            candidate.push_back( p );
            desp_map.push_back( p->descriptor_ );
        }
    }
    matcher_flann_.match ( desp_map, curr_->descriptors_, matches );
    // select the best matches
    float min_dis = std::min_element (
                        matches.begin(), matches.end(),
                        [] ( const cv::DMatch& m1, const cv::DMatch& m2 )
    {
        return m1.distance < m2.distance;
    } )->distance;

    match_3dpts_.clear();
    match_2dkp_index_.clear();
    flog_ << "match with matcher_flann_: " <<std::endl;
    for ( cv::DMatch& m : matches )
    {
        if ( m.distance < max<float> ( min_dis*match_ratio_, 30.0 ) )
        {
            match_3dpts_.push_back( candidate[m.queryIdx] );
            match_2dkp_index_.push_back( m.trainIdx );
            flog_ << match_3dpts_.back()->id_ << " " << match_2dkp_index_.back() << std::endl;
        }
    }
    flog_ << "end of match with matcher_flann_ " <<std::endl;

    //flog_<<"match cost time: "<<timer.elapsed() <<endl;
}
int VisualOdometry::featureMatchingWithRef()
{
    ORBmatcher matcher(0.8,true);
    vector<MapPoint::Ptr> vpMapPointMatches;
    int nmatch = matcher.searchByBoW(ref_, curr_, vpMapPointMatches);
    //flog_ << "match with ORBmatcher :" <<std::endl;
    match_3dpts_.clear();
    match_2dkp_index_.clear();
    for(size_t i = 0; i < vpMapPointMatches.size(); ++i){
        if(vpMapPointMatches[i]!=nullptr){
            match_3dpts_.push_back(vpMapPointMatches[i]);
            match_2dkp_index_.push_back(i);
           //flog_ << vpMapPointMatches[i]->id_ << " " << i << std::endl;
        }
    }
    curr_->vpMapPoints_ = vpMapPointMatches;
    //flog_ << "end of match with ORBmatcher " <<std::endl;
    return nmatch;
}

int VisualOdometry::poseEstimationOptimization()
{
    num_inliers_ = Optimizer::poseOptimization(curr_);
    // Discard outliers
    int nmatches = 0;
    for(int i =0; i<curr_->N_; i++)
    {
        if(curr_->vpMapPoints_[i])
        {
            if(curr_->vbOutlier_[i])
            {
                MapPoint::Ptr pMP = curr_->vpMapPoints_[i];

                curr_->vpMapPoints_[i]=nullptr;
                curr_->vbOutlier_[i]=false;
                //nmatches--;
            }
            else{
                nmatches++;
            }
        }
    }
    return nmatches;
}

bool VisualOdometry::checkEstimatedPose()
{
    // check if the estimated pose is good
    if ( num_inliers_ < min_inliers_ )
    {
        flog_<<"reject because inlier is too small: "<<num_inliers_<<endl;
        return false;
    }
    // if the motion is too large, it is probably wrong
    SE3<double> T_r_c = ref_->T_c_w_ * curr_->T_c_w_.inverse();
    Sophus::Vector6d d = T_r_c.log();
    flog_ << "motion change " << d.norm() << endl;
    if ( d.norm() > 3.0 )
    {
        flog_<<"reject because motion is too large: "<<d.norm() <<endl;
        return false;
    }
    return true;
}

bool VisualOdometry::checkKeyFrame()
{
    SE3<double> T_r_c = ref_->T_c_w_ * curr_->T_c_w_.inverse();
    Sophus::Vector6d d = T_r_c.log();
    Vector3d trans = d.head<3>();
    Vector3d rot = d.tail<3>();
    if ( rot.norm() >key_frame_min_rot || trans.norm() >key_frame_min_trans )
        return true;
    return false;
}
void VisualOdometry::reInitializeFrame()
{
    ref_ = curr_;
    map_->map_points_.clear();
    for (size_t i = 0; i<curr_->vKeys_.size(); i++)
    {
        double d = curr_->findDepth(curr_->vKeys_[i]);
        if (d < 0)
            continue;
        Vector3d p_world = ref_->camera_->pixel2world(
            Vector2d(curr_->vKeys_[i].pt.x, curr_->vKeys_[i].pt.y), curr_->T_c_w_, d
        );
        Vector3d n = p_world - ref_->getCamCenter();
        n.normalize();
        MapPoint::Ptr pNewMp = MapPoint::createMapPoint(
            p_world, n, curr_->descriptors_.row(i).clone(), curr_);
        map_->insertMapPoint(pNewMp);
        curr_->vpMapPoints_[i] = pNewMp;
        curr_->addMapPoint(pNewMp,i);
        pNewMp->addObservation(curr_, i);
        pNewMp->computeDistinctiveDescriptors();
    }
    key_frame_ids_.push_back(curr_->id_);
    map_->insertKeyFrame(curr_);
    flog_ << "re init frame " << endl;
}

void VisualOdometry::addKeyFrame()
{
    //1. sort the point by depth
    vector<pair<double, size_t> > vDepthIdx;
    for(size_t i = 0; i < curr_->N_; ++i){
        double z = curr_->findDepth(curr_->vKeys_[i]);
        if(z>0)
            vDepthIdx.push_back(make_pair(z, i));
    }
    //2. add the point
    int nPts = 0;
    flog_ <<"map point num before add keyframe: " << map_->map_points_.size() << " ";
    if(!vDepthIdx.empty()){
        sort(vDepthIdx.begin(),vDepthIdx.end());
        for(size_t j = 0; j < vDepthIdx.size(); ++j){
            int i = vDepthIdx[j].second;
            bool bCreateNew = false;
            MapPoint::Ptr pMp = curr_->vpMapPoints_[i];
            if(pMp==nullptr)
                bCreateNew = true;
            else if(pMp->observations_.empty()){
                bCreateNew = true;
                curr_->vpMapPoints_[i] = nullptr;
            }
            if(bCreateNew){
                Vector3d p_world = curr_->camera_->pixel2world(
                    Vector2d( curr_->vKeys_[i].pt.x, curr_->vKeys_[i].pt.y ),
                    curr_->T_c_w_, vDepthIdx[j].first );
                Vector3d n = p_world - curr_->getCamCenter();
                n.normalize();
                MapPoint::Ptr pNewMp = MapPoint::createMapPoint(p_world, n, curr_->descriptors_.row(i).clone(), curr_);
                pNewMp->addObservation(curr_, i);
                curr_->addMapPoint(pNewMp, i);
                pNewMp->computeDistinctiveDescriptors();
                pNewMp->updateNormalAndDepth();
                map_->insertMapPoint(pNewMp);
            }
            nPts++;
            if(nPts>100)
                break;
        }
    }
    flog_ << " after create new key frame: " << map_->map_points_.size()<< endl;
    
    key_frame_ids_.push_back(curr_->id_);
    map_->insertKeyFrame ( curr_ );
    ref_ = curr_;
    curr_->computeBoW();
    if (key_frame_ids_.size() > 1) {
        vector<unsigned long> frame_ids;
        if (key_frame_ids_.size() == 2)
            frame_ids = key_frame_ids_;
        else
            frame_ids.assign(key_frame_ids_.begin() + key_frame_ids_.size() - 3, key_frame_ids_.end());
        Optimizer::localBA(frame_ids, map_);
    }
    
}

void VisualOdometry::validateProjection()
{
    for(size_t i = 0; i < curr_->vpMapPoints_.size(); ++i){
        MapPoint::Ptr pMp = curr_->vpMapPoints_[i];
        if(pMp == nullptr) continue;
        Eigen::Vector3d pos = pMp->pos_;
        Eigen::Vector2d pix1 = curr_->camera_->world2pixel(pos, curr_->T_c_w_);
        Eigen::Vector2d pix0(curr_->vKeys_[i].pt.x, curr_->vKeys_[i].pt.y);
        if ((pix1 - pix0).norm() > 5.0)
            flog_ << "large error for projection: " << pix1.transpose()
            << " " << pix0.transpose() << endl;
    }
}

void VisualOdometry::addMapPoints()
{
    // add the new map points into map
    vector<bool> matched(curr_->vKeys_.size(), false); 
    for ( int index:match_2dkp_index_ )
        matched[index] = true;
    int add_num = 0;
    for ( int i=0; i<curr_->vKeys_.size(); i++ )
    {
        if ( matched[i] == true )   
            continue;
        double d = ref_->findDepth ( curr_->vKeys_[i] );
        if ( d<0 )  
            continue;
        Vector3d p_world = ref_->camera_->pixel2world (
            Vector2d ( curr_->vKeys_[i].pt.x, curr_->vKeys_[i].pt.y ), 
            curr_->T_c_w_, d
        );
        Vector3d n = p_world - ref_->getCamCenter();
        n.normalize();
        MapPoint::Ptr map_point = MapPoint::createMapPoint(
            p_world, n, curr_->descriptors_.row(i).clone(), curr_);
        map_->insertMapPoint( map_point );
        curr_->vpMapPoints_[i] = map_point;
        add_num++;
    }
    flog_<< "new map points are added: " << add_num <<endl;
}

void VisualOdometry::optimizeMap()
{
    // remove the hardly seen and no visible points 
    size_t map_points_sz = map_->map_points_.size();
    int outframe = 0, lowmatchratio = 0, largeangle = 0;
    for ( auto iter = map_->map_points_.begin(); iter != map_->map_points_.end(); )
    {
        MapPoint::Ptr pMp = iter->second;
        if ( !curr_->isInFrame(iter->second->pos_) )  {
            iter = map_->map_points_.erase(iter);
            outframe++;
            continue;
        }
        float match_ratio = float(pMp->found_times_)/pMp->visible_times_;
        if ( match_ratio < map_point_erase_ratio_ ) {
            iter = map_->map_points_.erase(iter);
            lowmatchratio++;
            continue;
        }
        if (pMp->last_frame_seen_ + 10 < curr_->id_){
            iter = map_->map_points_.erase(iter);
            continue;
        }
        
        double angle = getViewAngle( curr_, pMp );
        if ( angle > M_PI/6. )
        {
            iter = map_->map_points_.erase(iter);
            largeangle++;
            continue;
        }
        iter++;
    }
    
    if (match_2dkp_index_.size() < 100 || num_inliers_ * 2 < match_2dkp_index_.size()) {
        addMapPoints();
        if (match_2dkp_index_.size() > 100 && num_inliers_ * 2 < match_2dkp_index_.size())
            flog_ << "add map points because of low inliers ratio" << endl;
    }
    if ( map_->map_points_.size() > 1000 && map_point_erase_ratio_<0.30)  
    {
        map_point_erase_ratio_ += 0.05;
    }
    else 
        map_point_erase_ratio_ = 0.1;
    flog_ << "map points size change: " << map_points_sz << " to " << map_->map_points_.size() 
        << " outframe "<<outframe << " lowmatchratio "<<lowmatchratio << " largeangle " <<largeangle<< endl;
}

double VisualOdometry::getViewAngle ( Frame::Ptr frame, MapPoint::Ptr point )
{
    Vector3d n = point->pos_ - frame->getCamCenter();
    n.normalize();
    return acos( n.transpose()*point->norm_ );
}

void VisualOdometry::updateLocalKeyFrames()
{
    map<Frame::Ptr, int> keyframeCounter;
    for(size_t i = 0; i < curr_->N_; ++i){
        if(curr_->vpMapPoints_[i]!=nullptr){
            MapPoint::Ptr pMp = curr_->vpMapPoints_[i];
            if(!pMp->isBad()){
                const map<Frame::Ptr,size_t>& observations = pMp->observations_;
                for(map<Frame::Ptr,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else{
                curr_->vpMapPoints_[i]=nullptr;
            }
        }
    }
    if(keyframeCounter.empty()) return;
    int max = 0; 
    Frame::Ptr pKFmax = nullptr;
    local_key_frames_.clear();
    local_key_frames_.reserve(3*keyframeCounter.size());
    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<Frame::Ptr,int>::const_iterator it = keyframeCounter.begin(); it != keyframeCounter.end(); ++it){
        Frame::Ptr pKF = it->first;
        if(it->second>max){
            max = it->second;
            pKFmax = pKF;
        }
        local_key_frames_.push_back(pKF);
    }
}

void VisualOdometry::updateLocalMapPoints()
{
    local_map_pts_.clear();
    for(auto kf : local_key_frames_){
        const vector<MapPoint::Ptr> vMPs = kf->vpMapPoints_;
        for(auto mp : vMPs){
            if(mp==nullptr) continue;
            if(!mp->isBad())
                local_map_pts_.push_back(mp);
        }
    }
}

void VisualOdometry::searchLocalMapPoints()
{
    for(size_t i = 0; i < curr_->vpMapPoints_.size(); ++i){
        MapPoint::Ptr pMp = curr_->vpMapPoints_[i];
        if(pMp==nullptr) continue;
        if(pMp->isBad()){
            curr_->vpMapPoints_[i]==nullptr;
        }
        else{
            pMp->increaseVisible();
            pMp->last_frame_seen_ = curr_->id_;
            pMp->track_in_view_ = false;
        }
    }
    int nTomatch = 0;
    for(vector<MapPoint::Ptr>::iterator vit=local_map_pts_.begin(), vend=local_map_pts_.end(); vit!=vend; vit++){
        MapPoint::Ptr pMp = *vit;
        if(pMp->last_frame_seen_ == curr_->id_) continue;
        if(pMp->isBad()) continue;
        if(curr_->isInFrustum(pMp)){
            pMp->increaseVisible();
            nTomatch++;
        }
    }
    if(nTomatch>0){
        ORBmatcher matcher(0.8);
        matcher.searchByProjection(curr_,local_map_pts_,5);
    }
}

bool VisualOdometry::trackLocalMap()
{
    updateLocalKeyFrames();
    updateLocalMapPoints();
    searchLocalMapPoints();
    num_inliers_ = Optimizer::poseOptimization(curr_);
    int nmatch = 0;
    for(size_t i = 0; i < curr_->vpMapPoints_.size(); ++i){
        MapPoint::Ptr pMp = curr_->vpMapPoints_[i];
        if(pMp){
            if(curr_->vbOutlier_[i]){
                curr_->vpMapPoints_[i]=nullptr;
            }
            else{
                if(pMp->observations_.empty()==false)
                    nmatch++;
                pMp->found_times_++;
                pMp->last_frame_seen_ = curr_->id_;
            }
        }
    }
    if(nmatch < 30)
        return false;
    else
        return true;
}


}
