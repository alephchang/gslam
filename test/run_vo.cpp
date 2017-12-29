// -------------- test the visual odometry -------------
#include <fstream>
//#include <boost/timer.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/viz.hpp> 
#include <opencv2/imgproc/imgproc.hpp>

#include "gslam/config.h"
#include "gslam/visual_odometry.h"

int main ( int argc, char** argv )
{
    if ( argc != 2 )
    {
        cout<<"usage: run_vo parameter_file"<<endl;
        return 1;
    }

    gslam::Config::setParameterFile ( argv[1] );
    gslam::VisualOdometry::Ptr vo ( new gslam::VisualOdometry );

	string dataset_dir = gslam::Config::get<string>("dataset_dir");
    cout<<"dataset: "<<dataset_dir<<endl;
    ifstream fin ( dataset_dir+"/associate.txt" );
    if ( !fin )
    {
        cout<<"please generate the associate file called associate.txt!"<<endl;
        return 1;
    }

    vector<string> rgb_files, depth_files, rgb_times_str;
    vector<double> rgb_times, depth_times;
    while ( !fin.eof() )
    {
        string rgb_time, rgb_file, depth_time, depth_file;
        fin>>rgb_time>>rgb_file>>depth_time>>depth_file;
        rgb_times.push_back ( atof ( rgb_time.c_str() ) );
		rgb_times_str.push_back(rgb_time);
        depth_times.push_back ( atof ( depth_time.c_str() ) );
        rgb_files.push_back ( dataset_dir+"/"+rgb_file );
        depth_files.push_back ( dataset_dir+"/"+depth_file );

        if ( fin.good() == false )
            break;
    }

    gslam::Camera::Ptr camera ( new gslam::Camera );

    // visualization
    cv::viz::Viz3d vis ( "Visual Odometry" );
    cv::viz::WCoordinateSystem world_coor ( 1.0 ), camera_coor ( 0.5 );
    cv::Point3d cam_pos ( 0, -1.0, -1.0 ), cam_focal_point ( 0,0,0 ), cam_y_dir ( 0,1,0 );
    cv::Affine3d cam_pose = cv::viz::makeCameraPose ( cam_pos, cam_focal_point, cam_y_dir );
    vis.setViewerPose ( cam_pose );

    world_coor.setRenderingProperty ( cv::viz::LINE_WIDTH, 2.0 );
    camera_coor.setRenderingProperty ( cv::viz::LINE_WIDTH, 1.0 );
    vis.showWidget ( "World", world_coor );
    vis.showWidget ( "Camera", camera_coor );

    cout<<"read total "<<rgb_files.size() <<" entries"<<endl;
	vector<SE3<double> > estimated_pose;
    for ( int i=0; i<rgb_files.size(); i++ )
    {
        cout<<"****** loop "<<i<<" ******"<<endl;
        Mat color = cv::imread ( rgb_files[i] );
        Mat depth = cv::imread ( depth_files[i], -1 );
        if ( color.data==nullptr || depth.data==nullptr )
            break;
        gslam::Frame::Ptr pFrame = gslam::Frame::createFrame();
        pFrame->camera_ = camera;
        pFrame->color_ = color;
        pFrame->depth_ = depth;
        pFrame->time_stamp_ = rgb_times[i];

//        boost::timer timer;
        vo->addFrame ( pFrame );
//        cout<<"VO costs time: "<<timer.elapsed() <<endl;

        if ( vo->state_ == gslam::VisualOdometry::LOST )
            break;
        Sophus::SE3<double> Twc = pFrame->T_c_w_.inverse();
		estimated_pose.push_back(pFrame->T_c_w_);
        // show the map and the camera pose
        cv::Affine3d M (
            cv::Affine3d::Mat3 (
                Twc.rotationMatrix() ( 0,0 ), Twc.rotationMatrix() ( 0,1 ), Twc.rotationMatrix() ( 0,2 ),
                Twc.rotationMatrix() ( 1,0 ), Twc.rotationMatrix() ( 1,1 ), Twc.rotationMatrix() ( 1,2 ),
                Twc.rotationMatrix() ( 2,0 ), Twc.rotationMatrix() ( 2,1 ), Twc.rotationMatrix() ( 2,2 )
            ),
            cv::Affine3d::Vec3 (
                Twc.translation() ( 0,0 ), Twc.translation() ( 1,0 ), Twc.translation() ( 2,0 )
            )
        );

        Mat img_show = color.clone();
        for ( auto& pt:vo->map_->map_points_ )
        {
            gslam::MapPoint::Ptr p = pt.second;
            Vector2d pixel = pFrame->camera_->world2pixel ( p->pos_, pFrame->T_c_w_ );
            cv::circle ( img_show, cv::Point2f ( pixel ( 0,0 ),pixel ( 1,0 ) ), 5, cv::Scalar ( 0,255,0 ), 2 );
        }

        cv::imshow ( "image", img_show );
        cv::waitKey ( 1 );
        vis.setWidgetPose ( "Camera", M );
        vis.spinOnce ( 1, false );

        cout<<endl;
    }
	ofstream fo(dataset_dir + "/estimatedpose.txt");
	for (size_t i = 0; i < estimated_pose.size(); ++i) {
		const SE3<double>& se3(estimated_pose[i]);
		fo << rgb_times_str[i] << "\t" << se3.translation().x()<<" "
			<< se3.translation().y() << " "
			<< se3.translation().z() << " "
			<< se3.unit_quaternion().x()<< " " 
			<< se3.unit_quaternion().y() << " " 
			<< se3.unit_quaternion().z() << " " 
			<< se3.unit_quaternion().w() << endl;
	}
	fo.close();
    return 0;
}

