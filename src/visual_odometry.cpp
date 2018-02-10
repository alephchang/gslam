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

bool VisualOdometry::addFrame ( Frame::Ptr frame )
{
    flog_ << "Frame ID: " << frame->id_ << std::endl;
    switch ( state_ )
    {
    case INITIALIZING:
    {
        state_ = OK;
        curr_ = ref_ = frame;
        // extract features from first frame and add them into map
        detectAndComputeFeatures();
        addKeyFrame();      // the first frame is a key-frame
        break;
    }
    case OK:
    {
        curr_ = frame;
        curr_->T_c_w_ = ref_->T_c_w_;
        detectAndComputeFeatures();
        //featureMatching();
        int nmatches = featureMatchingWithRef();
        //poseEstimationPnP();
        int ngoodmatches = poseEstimationOptimization();
        flog_ << "inliers: " << ngoodmatches << "of matches: " << nmatches <<endl;
        if ( checkEstimatedPose() == true ) // a good estimation
        {
            curr_->T_c_w_ = T_c_w_estimated_;
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
            //reInitializeFrame();
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

void VisualOdometry::poseEstimationPnP()
{
    // construct the 3d 2d observations
    vector<cv::Point3f> pts3d;
    vector<cv::Point2f> pts2d;

/*    for ( int index:match_2dkp_index_ )
    {
        pts2d.push_back ( curr_->vKeys_[index].pt );
    }
    for ( MapPoint::Ptr pt:match_3dpts_ )
    {
        pts3d.push_back( pt->getPositionCV() );
    }
*/
    for(size_t i = 0; i < curr_->N_; ++i){
        if(curr_->vpMapPoints_[i]!=nullptr){
            pts2d.push_back(curr_->vKeys_[i].pt);
            pts3d.push_back(curr_->vpMapPoints_[i]->getPositionCV());
        }
    }
    Mat K = ( cv::Mat_<float> ( 3,3 ) <<
              ref_->camera_->fx_, 0, ref_->camera_->cx_,
              0, ref_->camera_->fy_, ref_->camera_->cy_,
              0,0,1
            );
    Eigen::AngleAxisd rotvec0(curr_->T_c_w_.rotationMatrix());

    Mat rotM = (cv::Mat_<float>(3, 3) <<
        curr_->T_c_w_.rotationMatrix()(0, 0), curr_->T_c_w_.rotationMatrix()(0, 1), curr_->T_c_w_.rotationMatrix()(0, 2),
        curr_->T_c_w_.rotationMatrix()(1, 0), curr_->T_c_w_.rotationMatrix()(1, 1), curr_->T_c_w_.rotationMatrix()(1, 2),
        curr_->T_c_w_.rotationMatrix()(2, 0), curr_->T_c_w_.rotationMatrix()(2, 1), curr_->T_c_w_.rotationMatrix()(2, 2));
    Mat rvec;
    Rodrigues(rotM, rvec);
    Mat tvec = (cv::Mat_<float>(3, 1) << curr_->T_c_w_.translation()[0],
        curr_->T_c_w_.translation()[1], curr_->T_c_w_.translation()[2]);
    Mat inliers;
    cv::solvePnPRansac(pts3d, pts2d, K, Mat(), rvec, tvec, true, 100, 4.0, 0.99, inliers, cv::SOLVEPNP_ITERATIVE);

    num_inliers_ = 0;// inliers.rows;

    optimizePnP(pts3d, pts2d, inliers, rvec, tvec);

    //add map points to the current frame
    curr_->vpMapPoints_.assign(curr_->N_, nullptr);//zero all the map points, and only record the inliers
    for (int i = 0; i < inliers.rows; ++i)
    {
        int index = inliers.at<int>(i, 0);
        if (index > 0) {
            curr_->addMapPoint2d(match_3dpts_[index]->id_, pts2d[index]);
            curr_->vpMapPoints_[match_2dkp_index_[index]] = match_3dpts_[index];
            num_inliers_++;
        }
    }
//    flog_ << "key point range: " << curr_->map_points_2d_.front().first
//        << " " << curr_->map_points_2d_.back().first << endl;
    //flog_<<"T_c_w_estimated_ after g2o: "<<endl<<T_c_w_estimated_.matrix()<<endl;
    flog_<<"pnp inliers: "<<num_inliers_
             << " good matches: " << match_3dpts_.size() << endl;
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

void VisualOdometry::optimizePnP(const vector<cv::Point3f>& pts3d, const vector<cv::Point2f>& pts2d, Mat& inliers,
    const Mat& rvec, const Mat& tvec)
{
    Eigen::Vector3d vec3(rvec.at<double>(0, 0), rvec.at<double>(1, 0), rvec.at<double>(2, 0));
    Eigen::AngleAxisd rotvec(vec3.norm(), vec3.normalized());

    T_c_w_estimated_ = SE3<double>(Sophus::Matrix3d(rotvec),
        Sophus::Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0)));

    // using bundle adjustment to optimize the pose
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 2>> Block;
    typedef g2o::LinearSolverDense<Block::PoseMatrixType> Linear;

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<Block>(g2o::make_unique<Linear>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setId(0);

    Eigen::Quaterniond se3_r(T_c_w_estimated_.rotationMatrix());
    g2o::SE3Quat Tcw;// = g2o::SE3Quat(se3_r, T_c_w_estimated_.translation());
    Tcw.setRotation(se3_r);
    Tcw.setTranslation(T_c_w_estimated_.translation());
    pose->setEstimate(Tcw);
    optimizer.addVertex(pose);


    // edges
    for (int i = 0; i<inliers.rows; i++)
    {
        int index = inliers.at<int>(i, 0);
        // 3D -> 2D projection
        EdgeProjectXYZ2UVPoseOnly* edge = new EdgeProjectXYZ2UVPoseOnly();
        edge->setId(i);
        edge->setVertex(0, pose);
        edge->camera_ = curr_->camera_.get();
        edge->point_ = Vector3d(pts3d[index].x, pts3d[index].y, pts3d[index].z);
        edge->setMeasurement(Vector2d(pts2d[index].x, pts2d[index].y));
        edge->setInformation(Eigen::Matrix2d::Identity());
        edge->setRobustKernel(new g2o::RobustKernelHuber());
        optimizer.addEdge(edge);
        // set the inlier map points 
        match_3dpts_[index]->matched_times_++;
        //flog_ << edge->point_.transpose() << " " << pts2d[index].x << " " << pts2d[index].y << endl;
    }

    optimizer.initializeOptimization();
    optimizer.optimize(10);
    auto edges = optimizer.edges();
    for (auto it = edges.begin(); it != edges.end(); ++it) {
        if (inliers.rows> 50 && dynamic_cast<EdgeProjectXYZ2UVPoseOnly*>(*it)->error().norm() > 4.0) {
            inliers.at<int>((*it)->id(), 0) = -1;
            flog_ << "point id: " << (*it)->vertex(0)->id() << " camera id: " << curr_->id_
                << " pixel locatoin: " << dynamic_cast<EdgeProjectXYZ2UVPoseOnly*>(*it)->measurement().transpose()
                << " error: " << dynamic_cast<EdgeProjectXYZ2UVPoseOnly*>(*it)->error().transpose() << std::endl;
        }
    }
    T_c_w_estimated_ = SE3<double>(
        pose->estimate().rotation(),
        pose->estimate().translation()
        );

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
    SE3<double> T_r_c = ref_->T_c_w_ * T_c_w_estimated_.inverse();
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
    SE3<double> T_r_c = ref_->T_c_w_ * T_c_w_estimated_.inverse();
    Sophus::Vector6d d = T_r_c.log();
    Vector3d trans = d.head<3>();
    Vector3d rot = d.tail<3>();
    if ( rot.norm() >key_frame_min_rot || trans.norm() >key_frame_min_trans )
        return true;
    return false;
}
void VisualOdometry::reInitializeFrame()
{
    curr_->T_c_w_ = T_c_w_estimated_;
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
        MapPoint::Ptr map_point = MapPoint::createMapPoint(
            p_world, n, curr_->descriptors_.row(i).clone(), curr_.get()
        );
        map_->insertMapPoint(map_point);
        curr_->addMapPoint2d(map_point->id_, curr_->vKeys_[i].pt);
        curr_->vpMapPoints_[i] = map_point;
    }
    key_frame_ids_.push_back(curr_->id_);
    map_->insertKeyFrame(curr_);
    recordKeyFrameForMapPoint();
    flog_ << "re init frame " << endl;
}

void VisualOdometry::addKeyFrame()
{
    if ( map_->keyframes_.empty() )
    {
        // first key-frame, add all 3d points into map
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
            MapPoint::Ptr map_point = MapPoint::createMapPoint(
                p_world, n, curr_->descriptors_.row(i).clone(), curr_.get()
            );
            map_->insertMapPoint( map_point );
            curr_->addMapPoint2d(map_point->id_, curr_->vKeys_[i].pt);
            curr_->vpMapPoints_[i] = map_point;
        }
    }
    key_frame_ids_.push_back(curr_->id_);
    map_->insertKeyFrame ( curr_ );
    ref_ = curr_;
    curr_->computeBoW();
    recordKeyFrameForMapPoint();
    if (key_frame_ids_.size() > 1) {
        vector<unsigned long> frame_ids;
        if (key_frame_ids_.size() == 2)
            frame_ids = key_frame_ids_;
        else
            frame_ids.assign(key_frame_ids_.begin() + key_frame_ids_.size() - 3, key_frame_ids_.end());
        Optimizer::localBA(frame_ids, map_);
    }
    
}

void VisualOdometry::recordKeyFrameForMapPoint()
{
    for (auto item : ref_->map_points_2d_) {
        auto it = map_->map_points_.find(item.first);
        if (it!= map_->map_points_.end())
            it->second->observed_frames_.push_back(ref_.get());
    }
}

void VisualOdometry::triangulateForNewKeyFrame()
{
    //1. find the matched key points between curr key frame and the previous key frame
    const Frame::Map_Point_2d& mp2d0 = ref_->map_points_2d_;
    const Frame::Map_Point_2d& mp2d1 = curr_->map_points_2d_;
    size_t i = 0,  j = 0;
    std::vector<unsigned long> map_point_idx;
    std::vector<cv::Point2f> pts0, pts1;
    for(auto it0 = mp2d0.begin(); it0 != mp2d0.end(); ++it0){
        auto it1 = mp2d1.find(it0->first);
        if(it1 != mp2d1.end()){
            pts0.push_back(ref_->camera_->pixel2camera(it0->second));
            pts1.push_back(curr_->camera_->pixel2camera(it1->second));
            map_point_idx.push_back(it0->first);
        }
    }

    if (pts0.empty()) {
        flog_ << "no key point match between current frame and ref frame" << endl;
        return;
    }
    flog_ << "triangulate matche points " << pts0.size() << endl;
    //2. triangulatePoints
    const Sophus::SE3d& Tref = ref_->T_c_w_;
    const Sophus::SE3d& Tcur = curr_->T_c_w_;
    cv::Mat T0 = (cv::Mat_<float>(3, 4) <<
        Tref.rotationMatrix()(0, 0), Tref.rotationMatrix()(0, 1), Tref.rotationMatrix()(0, 2), Tref.translation()(0),
        Tref.rotationMatrix()(1, 0), Tref.rotationMatrix()(1, 1), Tref.rotationMatrix()(1, 2), Tref.translation()(1),
        Tref.rotationMatrix()(2, 0), Tref.rotationMatrix()(2, 1), Tref.rotationMatrix()(2, 2), Tref.translation()(2)
        );
    cv::Mat T1 = (cv::Mat_<float>(3, 4) <<
        Tcur.rotationMatrix()(0, 0), Tcur.rotationMatrix()(0, 1), Tcur.rotationMatrix()(0, 2), Tcur.translation()(0),
        Tcur.rotationMatrix()(1, 0), Tcur.rotationMatrix()(1, 1), Tcur.rotationMatrix()(1, 2), Tcur.translation()(1),
        Tcur.rotationMatrix()(2, 0), Tcur.rotationMatrix()(2, 1), Tcur.rotationMatrix()(2, 2), Tcur.translation()(2)
        );

    cv::Mat pts4d;
    cv::triangulatePoints(T0, T1, pts0, pts1, pts4d);

    //update the map points
    for (i = 0; i < map_point_idx.size(); ++i)
    {
        Mat x = pts4d.col(i);
        x /= x.at<float>(3, 0);
        Eigen::Vector3d tri_pos = Eigen::Vector3d(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0));
        
        std::unordered_map<unsigned long, MapPoint::Ptr>::iterator it = map_->map_points_.find(map_point_idx[i]);
        if (it != map_->map_points_.end()) {
            if ((it->second->pos_ - tri_pos).norm() < 0.1 && false) {
                it->second->pos_ = tri_pos;
            }
        }    
    }
    //validate
/*    for (i = 0; i < map_point_idx.size(); ++i) {
        if(map_->map_points_.find(map_point_idx[i])== map_->map_points_.end())
            continue;
        Eigen::Vector3d pos = map_->map_points_[map_point_idx[i]]->pos_;
        Eigen::Vector3d p0 = ref_->camera_->world2camera(pos, ref_->T_c_w_);
        Eigen::Vector3d p1 = curr_->camera_->world2camera(pos, curr_->T_c_w_);
        p0 /= p0(2, 0);
        p1 /= p1(2, 0);
        flog_ << "triangulation error: " << (p0 - Eigen::Vector3d(pts0[i].x, pts0[i].y, 1.0)).norm()
            << " " << (p1 - Eigen::Vector3d(pts1[i].x, pts1[i].y, 1.0)).norm() << endl;
        //T0*cv::Vec3f(pos(0,0), pos(1,0), pos(2,0))
    }
*/
}

void VisualOdometry::validateProjection()
{
    for(auto it = curr_->map_points_2d_.begin(); it != curr_->map_points_2d_.end(); ++it){
        if(map_->map_points_.find(it->first) == map_->map_points_.end()) continue;
        Eigen::Vector3d pos = map_->map_points_[it->first]->pos_;
        Eigen::Vector2d pix1 = curr_->camera_->world2pixel(pos, curr_->T_c_w_);
        Eigen::Vector2d pix0(it->second.x, it->second.y);
        if ((pix1 - pix0).norm() > 5.0)
            flog_ << "large error for projection: " << pix1.transpose()
            << " " << pix0.transpose() << endl;
    }
}
void VisualOdometry::dumpMapAndKeyFrames()
{
    unordered_map<unsigned long, MapPoint::Ptr >::const_iterator it = map_->map_points_.begin();
    flog_ << "== Map Information == "<< map_->map_points_.size() << endl;

    for(auto it = map_->map_points_.begin(); it != map_->map_points_.end(); ++it){
        if (it->second->observed_frames_.size() < 2) continue;
        flog_ << "Map Point ID: " << it->first <<" oberved times: "<< it->second->observed_frames_.size() << endl;
        for (auto it_frame = it->second->observed_frames_.begin();
                    it_frame != it->second->observed_frames_.end(); ++it_frame) {
            flog_ << (*it_frame)->id_ << " ";
        }
        flog_ << endl;
    }
    flog_ << "== Key Frames ==" << map_->keyframes_.size() << endl;
    for (auto it_frame = map_->keyframes_.begin(); it_frame != map_->keyframes_.end(); ++it_frame) {
        flog_ << (it_frame->second)->id_ << " ";
    }
    flog_ << endl;
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
            p_world, n, curr_->descriptors_.row(i).clone(), curr_.get()
        );
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
        if ( !curr_->isInFrame(iter->second->pos_) )
        {
            iter = map_->map_points_.erase(iter);
            outframe++;
            continue;
        }
        float match_ratio = float(iter->second->matched_times_)/iter->second->visible_times_;
        if ( match_ratio < map_point_erase_ratio_ )
        {
            iter = map_->map_points_.erase(iter);
            lowmatchratio++;
            continue;
        }
        
        double angle = getViewAngle( curr_, iter->second );
        if ( angle > M_PI/6. )
        {
            iter = map_->map_points_.erase(iter);
            largeangle++;
            continue;
        }
        if ( iter->second->good_ == false )
        {
            // TODO try triangulate this map point 
        }
        iter++;
    }
    
    if (match_2dkp_index_.size() < 100 || num_inliers_ * 2 < match_2dkp_index_.size()) {
        addMapPoints();
        if (match_2dkp_index_.size() > 100 && num_inliers_ * 2 < match_2dkp_index_.size())
            flog_ << "add map points because of low inliers ratio" << endl;
    }
    if ( map_->map_points_.size() > 1000 )  
    {
        // TODO map is too large, remove some one 
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


}
