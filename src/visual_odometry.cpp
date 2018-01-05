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

namespace gslam
{

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
    orb_ = cv::ORB::create ( num_of_features_, scale_factor_, level_pyramid_ );
}

VisualOdometry::~VisualOdometry()
{
	flog_.close();
}

bool VisualOdometry::addFrame ( Frame::Ptr frame )
{
    switch ( state_ )
    {
    case INITIALIZING:
    {
        state_ = OK;
        curr_ = ref_ = frame;
        // extract features from first frame and add them into map
        extractKeyPoints();
        computeDescriptors();
        addKeyFrame();      // the first frame is a key-frame
        break;
    }
    case OK:
    {
        curr_ = frame;
        curr_->T_c_w_ = ref_->T_c_w_;
        extractKeyPoints();
        computeDescriptors();
        featureMatching();
        poseEstimationPnP();
		flog_<<"pnp inliers: "<<num_inliers_<<endl;
        if ( checkEstimatedPose() == true ) // a good estimation
        {
            curr_->T_c_w_ = T_c_w_estimated_;
			//validateProjection(); //for validation
            optimizeMap();
            num_lost_ = 0;
            if ( checkKeyFrame() == true ) // is a key-frame
            {
				//triangulate for key points in key frames
				triangulateForNewKeyFrame();
                addKeyFrame();
            }
        }
        else // bad estimation due to various reasons
        {
            num_lost_++;
            if ( num_lost_ > max_num_lost_ )
            {
                state_ = LOST;
            }
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
	flog_ << "Frame id: " << curr_->id_ <<" has "<< map_->map_points_.size()<<" map points, "
		<<"has "<< map_->keyframes_.size()<< " key frames"<< endl;
    return true;
}

bool VisualOdometry::setLogFile(const std::string& logpath)
{
	flog_.open(logpath, fstream::out);
	return flog_.good();
}

void VisualOdometry::extractKeyPoints()
{
    boost::timer timer;
    orb_->detect ( curr_->color_, keypoints_curr_ );
    flog_<<"extract keypoints cost time: "<<timer.elapsed() <<endl;
}

void VisualOdometry::computeDescriptors()
{
    boost::timer timer;
    orb_->compute ( curr_->color_, keypoints_curr_, descriptors_curr_ );
    //flog_<<"descriptor computation cost time: "<<timer.elapsed() <<endl;
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
    
    matcher_flann_.match ( desp_map, descriptors_curr_, matches );
    // select the best matches
    float min_dis = std::min_element (
                        matches.begin(), matches.end(),
                        [] ( const cv::DMatch& m1, const cv::DMatch& m2 )
    {
        return m1.distance < m2.distance;
    } )->distance;

    match_3dpts_.clear();
    match_2dkp_index_.clear();
    for ( cv::DMatch& m : matches )
    {
        if ( m.distance < max<float> ( min_dis*match_ratio_, 30.0 ) )
        {
            match_3dpts_.push_back( candidate[m.queryIdx] );
            match_2dkp_index_.push_back( m.trainIdx );
        }
    }
    flog_<<"good matches: "<<match_3dpts_.size() <<endl;
    //flog_<<"match cost time: "<<timer.elapsed() <<endl;
}

void VisualOdometry::poseEstimationPnP()
{
    // construct the 3d 2d observations
    vector<cv::Point3f> pts3d;
    vector<cv::Point2f> pts2d;

    for ( int index:match_2dkp_index_ )
    {
        pts2d.push_back ( keypoints_curr_[index].pt );
    }
    for ( MapPoint::Ptr pt:match_3dpts_ )
    {
        pts3d.push_back( pt->getPositionCV() );
    }

    Mat K = ( cv::Mat_<double> ( 3,3 ) <<
              ref_->camera_->fx_, 0, ref_->camera_->cx_,
              0, ref_->camera_->fy_, ref_->camera_->cy_,
              0,0,1
            );
	Eigen::AngleAxisd rotvec0(curr_->T_c_w_.rotationMatrix());
	double angle = rotvec0.angle();
	Mat rvec = (cv::Mat_<double>(3, 1) << angle * rotvec0.axis()[0] ,
					angle * rotvec0.axis()[1] , angle * rotvec0.axis()[2]);
	Mat tvec = (cv::Mat_<double>(3, 1) << curr_->T_c_w_.translation()[0],
		curr_->T_c_w_.translation()[1], curr_->T_c_w_.translation()[2]);
    Mat inliers;
    cv::solvePnPRansac ( pts3d, pts2d, K, Mat(), rvec, tvec, true, 100, 4.0, 0.99, inliers );
    num_inliers_ = inliers.rows;

	Eigen::Vector3d vec3(rvec.at<double>(0, 0), rvec.at<double>(1, 0), rvec.at<double>(2, 0));
	Eigen::AngleAxisd rotvec(vec3.norm(), vec3.normalized());

    T_c_w_estimated_ = SE3<double>(Sophus::Matrix3d(rotvec), 
		Sophus::Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0)));

    // using bundle adjustment to optimize the pose
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,2>> Block;
	typedef g2o::LinearSolverDense<Block::PoseMatrixType> Linear;

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
						g2o::make_unique<Block>(g2o::make_unique<Linear>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );

    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setId ( 0 );

	Eigen::Quaterniond se3_r(T_c_w_estimated_.rotationMatrix());
	g2o::SE3Quat Tcw;// = g2o::SE3Quat(se3_r, T_c_w_estimated_.translation());
	Tcw.setRotation(se3_r);
	Tcw.setTranslation(T_c_w_estimated_.translation());
	pose->setEstimate(Tcw);
	optimizer.addVertex(pose);


    // edges
    for ( int i=0; i<inliers.rows; i++ )
    {
        int index = inliers.at<int> ( i,0 );
        // 3D -> 2D projection
        EdgeProjectXYZ2UVPoseOnly* edge = new EdgeProjectXYZ2UVPoseOnly();
        edge->setId ( i );
        edge->setVertex ( 0, pose );
        edge->camera_ = curr_->camera_.get();
        edge->point_ = Vector3d ( pts3d[index].x, pts3d[index].y, pts3d[index].z );
        edge->setMeasurement ( Vector2d ( pts2d[index].x, pts2d[index].y ) );
        edge->setInformation ( Eigen::Matrix2d::Identity() );
		edge->setRobustKernel(new g2o::RobustKernelHuber());
        optimizer.addEdge ( edge );
        // set the inlier map points 
        match_3dpts_[index]->matched_times_++;
    }

    optimizer.initializeOptimization();
    optimizer.optimize ( 10 );

    T_c_w_estimated_ = SE3<double>(
        pose->estimate().rotation(),
        pose->estimate().translation()
    );
    
	//add map points to the current frame
	for (int i = 0; i < inliers.rows; ++i)
	{
		int index = inliers.at<int>(i, 0);
		curr_->addMapPoint2d(match_3dpts_[index]->id_, pts2d[index]);
	}
	curr_->sortMapPoint2d();
	flog_ << "key point range: " << curr_->map_points_2d_.front().first
		<< " " << curr_->map_points_2d_.back().first << endl;
//    flog_<<"T_c_w_estimated_: "<<endl<<T_c_w_estimated_.matrix()<<endl;
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

void VisualOdometry::addKeyFrame()
{
    if ( map_->keyframes_.empty() )
    {
        // first key-frame, add all 3d points into map
        for ( size_t i=0; i<keypoints_curr_.size(); i++ )
        {
            double d = curr_->findDepth ( keypoints_curr_[i] );
            if ( d < 0 ) 
                continue;
            Vector3d p_world = ref_->camera_->pixel2world (
                Vector2d ( keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y ), curr_->T_c_w_, d
            );
            Vector3d n = p_world - ref_->getCamCenter();
            n.normalize();
            MapPoint::Ptr map_point = MapPoint::createMapPoint(
                p_world, n, descriptors_curr_.row(i).clone(), curr_.get()
            );
            map_->insertMapPoint( map_point );
			curr_->addMapPoint2d(map_point->id_, keypoints_curr_[i].pt);
        }
		curr_->sortMapPoint2d();
    }
    
    map_->insertKeyFrame ( curr_ );
    ref_ = curr_;
	recordKeyFrameForMapPoint();
}

void VisualOdometry::recordKeyFrameForMapPoint()
{
	for (auto item : ref_->map_points_2d_) {
		map_->map_points_[item.first]->observed_frames_.push_back(ref_.get());
	}
}

void VisualOdometry::triangulateForNewKeyFrame()
{
	//1. find the matched key points between curr key frame and the previous key frame
	const vector<Frame::KeyPoint2d>& mp2d0 = ref_->map_points_2d_;
	const vector<Frame::KeyPoint2d>& mp2d1 = curr_->map_points_2d_;
	size_t i = 0,  j = 0;
	std::vector<unsigned long> map_point_idx;
	std::vector<cv::Point2f> pts0, pts1;
	while (i < mp2d0.size() && j < mp2d1.size()) {
		if (mp2d0[i].first == mp2d1[j].first) {
			pts0.push_back(ref_->camera_->pixel2camera(mp2d0[i].second));
			pts1.push_back(curr_->camera_->pixel2camera(mp2d1[j].second));
			map_point_idx.push_back(mp2d0[i].first);
			i++;
			j++;
		}
		else if (mp2d0[i].first < mp2d1[j].first) {
			i++;
		}
		else
			j++;
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
			if ((it->second->pos_ - tri_pos).norm() < 0.1) {
				it->second->pos_ = tri_pos;
			}
		}	
	}
	//validate
/*	for (i = 0; i < map_point_idx.size(); ++i) {
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
	for (size_t i = 0; i < curr_->map_points_2d_.size(); ++i) {
		Eigen::Vector3d pos = map_->map_points_[curr_->map_points_2d_[i].first]->pos_;
		Eigen::Vector2d pix1 = curr_->camera_->world2pixel(pos, curr_->T_c_w_);
		Eigen::Vector2d pix0(curr_->map_points_2d_[i].second.x, curr_->map_points_2d_[i].second.y);
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
		flog_ << "Map Point ID: " << it->first << endl;
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
    vector<bool> matched(keypoints_curr_.size(), false); 
    for ( int index:match_2dkp_index_ )
        matched[index] = true;
	int add_num = 0;
    for ( int i=0; i<keypoints_curr_.size(); i++ )
    {
        if ( matched[i] == true )   
            continue;
        double d = ref_->findDepth ( keypoints_curr_[i] );
        if ( d<0 )  
            continue;
        Vector3d p_world = ref_->camera_->pixel2world (
            Vector2d ( keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y ), 
            curr_->T_c_w_, d
        );
        Vector3d n = p_world - ref_->getCamCenter();
        n.normalize();
        MapPoint::Ptr map_point = MapPoint::createMapPoint(
            p_world, n, descriptors_curr_.row(i).clone(), curr_.get()
        );
        map_->insertMapPoint( map_point );
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
    
    if ( match_2dkp_index_.size()<100 )
        addMapPoints();
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
