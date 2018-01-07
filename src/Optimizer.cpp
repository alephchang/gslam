#include "gslam/Optimizer.h"
#include "gslam/g2o_types.h"

namespace gslam {
	Optimizer::Optimizer()
	{
	}


	Optimizer::~Optimizer()
	{
	}

	void Optimizer::localBA(vector<Frame::Ptr>& frames, Map::Ptr map)
	{
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
			//K.at<double>(0, 0), Eigen::Vector2d(K.at<double>(0, 2), K.at<double>(1, 2)), 0
		);
		camera->setId(0);
		optimizer.addParameter(camera);
		int index = frames.size();
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
				for(auto itedge = edge_candidate.begin(); itedge != edge_candidate.end(); ++itedge){
					g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
					//g2o::EdgeProjectXYZ2UV
					e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(index)));
					e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(itedge->first)));
					Eigen::Matrix<double, 2, 1> obs;
					obs << itedge->second.x, itedge->second.y;
					e->setMeasurement(obs);
					g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
					e->setRobustKernel(rk);
				}
				index++;
				//add edges
			}
		}

	}
}