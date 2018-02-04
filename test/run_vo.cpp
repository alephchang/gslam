#include"run_vo.h"

#include "gslam/g2o_types.h"
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include<boost/timer.hpp>
#include<fstream>

// -------------- test the visual odometry -------------


int run_vo ( int argc, char** argv )
{
    if ( argc != 2 )
    {
        for (int i = 0; i < argc; ++i)
        cout << argv[i] << endl;
        cout<<"usage: run_vo parameter_file"<<endl;
        return 1;
    }

    gslam::Config::setParameterFile ( argv[1] );
    

    string dataset_dir = gslam::Config::get<string>("dataset_dir");
    cout<<"dataset: "<<dataset_dir<<endl;
    ifstream fin ( dataset_dir+"/associate.txt" );
    if ( !fin )
    {
        cout<<"please generate the associate file called associate.txt!"<<endl;
        return 1;
    }
    
    gslam::VisualOdometry::Ptr vo(new gslam::VisualOdometry);
    if(vo->setLogFile(dataset_dir + "/log.txt")==false){
        cout << "Faile to create the log file: " << dataset_dir + "log.txt" << endl;
        return 1;
    }

    vector<string> rgb_files, depth_files;
    vector<double> rgb_times, depth_times;
    while ( !fin.eof() )
    {
        string rgb_time, rgb_file, depth_time, depth_file;
        fin>>rgb_time>>rgb_file>>depth_time>>depth_file;
        if(rgb_file.empty()) break;
        rgb_times.push_back ( atof ( rgb_time.c_str() ) );
        depth_times.push_back ( atof ( depth_time.c_str() ) );
        rgb_files.push_back ( dataset_dir+"/"+rgb_file );
        depth_files.push_back ( dataset_dir+"/"+depth_file );
        cout << "rgb_file: "<<rgb_file <<endl;
        if ( fin.good() == false )
            break;
    }

    //Load ORB Vocabulary
    string orbVocabDir= gslam::Config::get<string>("orb_vocab_dir");
    cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;

    gslam::Frame::pORBvocab_ = std::make_shared<gslam::ORBVocabulary>();
    bool bVocLoad = gslam::Frame::pORBvocab_->loadFromTextFile(orbVocabDir);
    if(!bVocLoad)
    {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Falied to open at: " << orbVocabDir << endl;
        exit(-1);
    }
    cout << "Vocabulary loaded!" << endl << endl;
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
    Mat gray;
    for ( int i=0; i<rgb_files.size(); i++ )
    {
        cout<<"****** loop "<<i<<" ******"<<endl;
        Mat color = cv::imread ( rgb_files[i] );
        cvtColor(color, gray, CV_RGB2GRAY);
        Mat depth = cv::imread ( depth_files[i], -1 );
        if ( color.data==nullptr || depth.data==nullptr )
            break;
        gslam::Frame::Ptr pFrame = gslam::Frame::createFrame();
        pFrame->id_ = i;
        pFrame->camera_ = camera;
        pFrame->color_ = gray;
        pFrame->depth_ = depth;
        pFrame->time_stamp_ = rgb_times[i];

        boost::timer timer;
        vo->addFrame ( pFrame );
        cout<<"VO costs time: "<<timer.elapsed() <<endl;

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
    fo.precision(15);
    for (size_t i = 0; i < estimated_pose.size(); ++i) {
        const SE3<double>& se3(estimated_pose[i]);
        fo << rgb_times[i] << "\t" << se3.translation().x()<<" "
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

bool load_pose_file(const string& path, vector<pair<double, SE3<double> > >& poses)
{
	ifstream fin(path);
	if(!fin){
		cout << "cannot load pose file: " << path << endl;
		return false;
	}
	string line;
	vector<string> strs;
	while (!fin.eof())
	{
		getline(fin, line);
		boost::split(strs, line, boost::is_any_of("\t "));
		if (strs.empty() || strs[0].empty() || strs[0][0] == '#')
			continue;
		double time_stamp;
		Eigen::Vector3d t(boost::lexical_cast<double>(strs[1]), 
			boost::lexical_cast<double>(strs[2]), boost::lexical_cast<double>(strs[3]) );
		Eigen::Quaternion<double> r(boost::lexical_cast<double>(strs[7]),
									boost::lexical_cast<double>(strs[4]), 
									boost::lexical_cast<double>(strs[5]),
									boost::lexical_cast<double>(strs[6]) );
		time_stamp = boost::lexical_cast<double>(strs[0]);

		pair<double, SE3<double> > time_pose;
		time_pose.first = time_stamp;
		time_pose.second = SE3<double>(r, t);
		poses.push_back(time_pose);
	}
	return true;
}
///we assume that truth_pose has higher frequence than est_pose
///and for each item in est_pose, pick the first larger one in truth_pose.
void filter_ground_truth(const vector<pair<double, SE3<double> > >&est_pose,
        vector<pair<double, SE3<double> > >& truth_pose)
{
    vector<int> truth_idx;
    if (est_pose.empty() || truth_pose.empty() || est_pose[0].first > truth_pose.back().first) {
        truth_pose.clear();
        return;
    }
    int j = 0;
    for(size_t i = 0; i < est_pose.size(); ++i){
        double current = est_pose[i].first;
        while(j!=truth_pose.size()){
            if(current < truth_pose[j].first){
                truth_idx.push_back(j);
                break;
            }
            j++;
        }
    }
    for(size_t i = 0; i < truth_idx.size(); ++i){
        truth_pose[i] = truth_pose[truth_idx[i]];
    }
    truth_pose.resize(truth_idx.size());
}

int validate_result(int argc, char** argv)
{
    //load estimatedpose.txt and groundtruth.txt and compare
    if (argc != 2){
        cout << "usage: run_vo parameter_file" << endl;
        return 1;
    }

	gslam::Config::setParameterFile(argv[1]);
	string dataset_dir = gslam::Config::get<string>("dataset_dir");
	vector<pair<double, SE3<double> > > est_pose, truth_pose;
	
	if (load_pose_file(dataset_dir + "/groundtruth.txt", truth_pose) == false
		|| load_pose_file(dataset_dir + "/estimatedpose.txt", est_pose) == false) {
		return 1;
	}


	cout.precision(4);

	
	filter_ground_truth(est_pose, truth_pose);

	cout << est_pose.size() << " " << truth_pose.size() << endl;
	// visualization
	cv::viz::Viz3d vis("Visual Odometry");
	cv::viz::WCoordinateSystem world_coor(1.0), camera_coor(0.5);
	cv::Point3d cam_pos(0, -1.0, -1.0), cam_focal_point(0, 0, 0), cam_y_dir(0, 1, 0);
	cv::Affine3d cam_pose = cv::viz::makeCameraPose(cam_pos, cam_focal_point, cam_y_dir);
	vis.setViewerPose(cam_pose);

	world_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);
	camera_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 1.0);
	vis.showWidget("World", world_coor);
	vis.showWidget("Camera", camera_coor);
	const SE3<double> ref = truth_pose[0].second.inverse();
	for(size_t i = 0; i < est_pose.size(); ++i){
		// show the map and the camera pose
		const SE3<double>& Twc = est_pose[i].second.inverse();
		cv::Affine3d M(
			cv::Affine3d::Mat3(
				Twc.rotationMatrix() (0, 0), Twc.rotationMatrix() (0, 1), Twc.rotationMatrix() (0, 2),
				Twc.rotationMatrix() (1, 0), Twc.rotationMatrix() (1, 1), Twc.rotationMatrix() (1, 2),
				Twc.rotationMatrix() (2, 0), Twc.rotationMatrix() (2, 1), Twc.rotationMatrix() (2, 2)
			),
			cv::Affine3d::Vec3(
				Twc.translation() (0, 0), Twc.translation() (1, 0), Twc.translation() (2, 0)
			)
		);
		const SE3<double>& Twc_truth = ref*truth_pose[i].second;
		cv::Affine3d M_t(
			cv::Affine3d::Mat3(
				Twc_truth.rotationMatrix() (0, 0), Twc_truth.rotationMatrix() (0, 1), Twc_truth.rotationMatrix() (0, 2),
				Twc_truth.rotationMatrix() (1, 0), Twc_truth.rotationMatrix() (1, 1), Twc_truth.rotationMatrix() (1, 2),
				Twc_truth.rotationMatrix() (2, 0), Twc_truth.rotationMatrix() (2, 1), Twc_truth.rotationMatrix() (2, 2)
			),
			cv::Affine3d::Vec3(
				Twc_truth.translation() (0, 0), Twc_truth.translation() (1, 0), Twc_truth.translation() (2, 0)
			)
		);
		vis.setWidgetPose("Camera", M);
		vis.setWidgetPose("World", M_t);
		//Sleep(100);
		cout << "Time diff: "<<truth_pose[i].first - est_pose[i].first << " Pose diff:"
			<<(truth_pose[i].second*est_pose[i].second.inverse()).log().norm()<< endl;
		vis.spinOnce(1, false);
	}

	return 0;
}

void testSE3QuatError()
{
	Eigen::Quaterniond se3_r;
	g2o::SE3Quat Tcw;
	Tcw.setRotation(se3_r);
}

int generate_associate_txt(const char* dir)
{
	string rgbpath("rgb\\rgb.txt");	//record all the rgb file names in order
	string depthpath("depth\\depth.txt"); //record all the depth file names in order
	ifstream fin_rgb(dir + rgbpath);
	ifstream fin_depth(dir + depthpath);
	vector<string> rgb_files, depth_files;
	string buf;
	while(!fin_rgb.eof())
	{	
		fin_rgb >> buf;
		rgb_files.push_back(buf);
	}
	while (!fin_depth.eof())
	{
		fin_depth >> buf;
		depth_files.push_back(buf);
	}
	size_t len = rgb_files.size() < depth_files.size() ? rgb_files.size() : depth_files.size();
	ofstream fou_asso(dir + string("associate.txt"));
	for(size_t i = 0; i < len; ++i){
		size_t pos0 = rgb_files[i].find_last_of('.');
		size_t pos1 = depth_files[i].find_last_of('.');
		//format£º rgb_time rgb_file_path depth_time depth_file_path
		fou_asso << rgb_files[i].substr(0, pos0) << " rgb\\"<<rgb_files[i] << " "
			<< depth_files[i].substr(0,pos1) <<" depth\\" << depth_files[i] << endl;
	}
	fou_asso.close();
	return 0;
}
