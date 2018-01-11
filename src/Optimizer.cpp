#include "gslam/Optimizer.h"
#include "gslam/g2o_types.h"
#include<fstream>

namespace gslam {
	Optimizer::Optimizer()
	{
	}


	Optimizer::~Optimizer()
	{
	}

	int Optimizer::optId = 0;
	std::string Optimizer::logPath;

	void Optimizer::localBA(vector<unsigned long>& frame_ids, Map::Ptr map)
	{
		vector<Frame::Ptr> frames(frame_ids.size());
		for (size_t i = 0; i < frames.size(); ++i) frames[i] = map->keyframes_[frame_ids[i]];
		typedef g2o::BlockSolver< g2o::BlockSolverTraits<6, 3> > BlockSolver_6_3;
		typedef g2o::LinearSolverDense<BlockSolver_6_3::PoseMatrixType> Linear;
		g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
			g2o::make_unique<BlockSolver_6_3>(g2o::make_unique<Linear>()));

		g2o::SparseOptimizer optimizer;
		optimizer.setAlgorithm(solver);
		//nodes of camera pose
		for(size_t i = 0; i < frames.size(); ++i){
			g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
			pose->setId(i);
			Eigen::Quaterniond se3_r(frames[i]->T_c_w_.rotationMatrix());
			g2o::SE3Quat Tcw = g2o::SE3Quat(se3_r, frames[i]->T_c_w_.translation());
			//Tcw.setRotation(se3_r);
			//Tcw.setTranslation(T_c_w_estimated_.translation());
			pose->setEstimate(Tcw);
			optimizer.addVertex(pose);
			if (i == 0)
				pose->setFixed(true);
		}

		//nodes of map point
		// parameter: camera intrinsics
		g2o::CameraParameters* camera = new g2o::CameraParameters(
			frames[0]->camera_->fx_, Eigen::Vector2d(frames[0]->camera_->cx_, frames[0]->camera_->cy_), 0);
		camera->setId(0);

		optimizer.addParameter(camera);
		int index = frames.size();
		int edge_id = 1;
		std::vector<MapPoint::Ptr> map_points;
		for(auto it = map->map_points_.begin(); it != map->map_points_.end(); ++it){
			int id = it->first;
			std::list<std::pair<int, cv::Point2f> > edge_candidate;
			for(size_t i = 0; i < frames.size(); ++i){
				auto it2d = frames[i]->map_points_2d_.find(id);
				if(it2d != frames[i]->map_points_2d_.end()){
					edge_candidate.push_back(std::pair<int, cv::Point2f>(i, it2d->second));
				}
			}
			///TODO: complete the edge 
			if(edge_candidate.size()>1){//the map point is observed more than once
				//add the map point as node
				g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
				point->setId(index);
				point->setEstimate(it->second->pos_);
				point->setMarginalized(true);
				optimizer.addVertex(point);
				map_points.push_back(it->second);
				//add edges

				for(auto itedge = edge_candidate.begin(); itedge != edge_candidate.end(); ++itedge){
					g2o::EdgeProjectXYZ2UV* e = new g2o::EdgeProjectXYZ2UV();
					e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(index)));
					e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(itedge->first)));
					e->setMeasurement(Eigen::Vector2d(itedge->second.x, itedge->second.y));
					g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
					e->setRobustKernel(rk);
					e->setParameterId(0, 0);
					e->setInformation(Eigen::Matrix2d::Identity());
					e->setId(edge_id++);
					optimizer.addEdge(e);
				}
				index++;
			}
		}
		std::ofstream fou;
		if (!logPath.empty()) {
			std::string path = logPath + "/opt_" + std::to_string(optId) + ".txt";
			fou.open(path.c_str(), std::ios::out);
		}

		if (fou.good()) {
			fou << "pose and points before optimization: " << std::endl;
			fou << " camera pose: " << std::endl;
			for (size_t i = 0; i < frames.size(); ++i) {
				const g2o::SE3Quat& Tcw = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(i))->estimate();
				fou << "camera " << i << " : " << Tcw.rotation().coeffs().transpose() << "\t" << Tcw.translation().transpose() << std::endl;
			}
/*			fou << " map points: " << std::endl;
			for (size_t i = frames.size(); i < index; ++i) {
				const Eigen::Vector3d& vPoint = dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(i))->estimate();
				fou << "point " << i << " : " << vPoint.transpose() << std::endl;
			}*/
			auto edges = optimizer.edges();
			for (auto it = edges.begin(); it != edges.end(); ++it) {
				dynamic_cast<g2o::EdgeProjectXYZ2UV*>(*it)->computeError();
				if (dynamic_cast<g2o::EdgeProjectXYZ2UV*>(*it)->error().norm() < 4.0) continue;
				fou << "point id: " << (*it)->vertex(0)->id() << " camera id: " << (*it)->vertex(1)->id()
					<< " pixel locatoin: " << dynamic_cast<g2o::EdgeProjectXYZ2UV*>(*it)->measurement().transpose() 
					<< " error: "<< dynamic_cast<g2o::EdgeProjectXYZ2UV*>(*it)->error().transpose()<< std::endl;
			}
		}

		optimizer.setVerbose(true);
		optimizer.initializeOptimization();
		optimizer.optimize(10);

		if (fou.good()) {
			fou << "pose and points after optimization: " << std::endl;
			fou << " camera pose: " << std::endl;
			for (size_t i = 0; i < frames.size(); ++i) {
				const g2o::SE3Quat& Tcw = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(i))->estimate();
				fou << "camera " << i << " : " << Tcw.rotation().coeffs().transpose() << "\t" << Tcw.translation().transpose() << std::endl;
			}
/*			fou << " map points: " << std::endl;
			for (size_t i = frames.size(); i < index; ++i) {
				const Eigen::Vector3d& vPoint = dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(i))->estimate();
				fou << "point " << i << " : " << vPoint.transpose() << std::endl;
			}*/
			auto edges = optimizer.edges();
			for (auto it = edges.begin(); it != edges.end(); ++it) {
				if (dynamic_cast<g2o::EdgeProjectXYZ2UV*>(*it)->error().norm() < 4.0) continue;
				fou << "point id: " << (*it)->vertex(0)->id() << " camera id: " << (*it)->vertex(1)->id()
					<< " pixel locatoin: " << dynamic_cast<g2o::EdgeProjectXYZ2UV*>(*it)->measurement().transpose()
					<< " error: " << dynamic_cast<g2o::EdgeProjectXYZ2UV*>(*it)->error().transpose() << std::endl;
			}
		}
		fou.close();
		optId++;
		for (size_t i = 0; i < frames.size(); ++i) {
			g2o::SE3Quat Tcw = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(i))->estimate();
			frames[i]->T_c_w_ = Sophus::SE3d(Tcw.rotation(), Tcw.translation());
		}
		for (size_t i = frames.size(); i < index; ++i) {
			const Eigen::Vector3d& vPoint = dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(i))->estimate();
			map_points[i - frames.size()]->pos_ = vPoint;
		}
	}
}