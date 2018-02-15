#include "gslam/Optimizer.h"
#include "gslam/g2o_types.h"
#include "gslam/frame.h"
#include "gslam/Converter.h" 
#include"Eigen/Dense"
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
/*            fou << " map points: " << std::endl;
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
/*            fou << " map points: " << std::endl;
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
    
int Optimizer::poseOptimization(Frame::Ptr pFrame)
{
    g2o::SparseOptimizer optimizer;
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6, 3> > BlockSolver_6_3;
    typedef g2o::LinearSolverDense<BlockSolver_6_3::PoseMatrixType> Linear;
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolver_6_3>(g2o::make_unique<Linear>()));
//    BlockSolver_6_3::LinearSolverType * linearSolver;
//    linearSolver = new g2o::LinearSolverDense<BlockSolver_6_3::PoseMatrixType>();
//    BlockSolver_6_3 * solver_ptr = new BlockSolver_6_3(linearSolver);
//    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences=0;

    // Set Frame vertex
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    //vSE3->setEstimate(Converter::toSE3Quat(pFrame->T_c_w_));
    Eigen::Quaterniond se3_r(pFrame->T_c_w_.rotationMatrix());
    g2o::SE3Quat Tcw = g2o::SE3Quat(se3_r, pFrame->T_c_w_.translation());
    vSE3->setEstimate(Tcw);
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = pFrame->N_;

    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
    vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    const float deltaMono = sqrt(5.991);

    for(int i=0; i<N; i++)
    {
        MapPoint::Ptr pMP = pFrame->vpMapPoints_[i];
        if(pMP!=nullptr)
        {
            nInitialCorrespondences++;
            pFrame->vbOutlier_[i] = false;

            Eigen::Matrix<double,2,1> obs;
            const cv::KeyPoint &kpUn = pFrame->vKeys_[i];
            obs << kpUn.pt.x, kpUn.pt.y;

            g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
            e->setMeasurement(obs);
            const float invSigma2 = pFrame->vInvLevelSigma2_[kpUn.octave];
            e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(deltaMono);

            e->fx = pFrame->camera_->fx_;
            e->fy = pFrame->camera_->fy_;
            e->cx = pFrame->camera_->cx_;
            e->cy = pFrame->camera_->cy_;
            cv::Point3f Xw = pMP->getPositionCV();
            e->Xw[0] = Xw.x;
            e->Xw[1] = Xw.y;
            e->Xw[2] = Xw.z;

            optimizer.addEdge(e);

            vpEdgesMono.push_back(e);
            vnIndexEdgeMono.push_back(i);
        }
    }

    if(nInitialCorrespondences<3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    const int its[4]={10,10,10,10};    

    int nBad=0;
    for(size_t it=0; it<4; it++)
    {
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->T_c_w_));
        optimizer.initializeOptimization(0);
        optimizer.setVerbose(true);
        optimizer.optimize(its[it]);

        nBad=0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if(pFrame->vbOutlier_[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Mono[it])
            {                
                pFrame->vbOutlier_[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                pFrame->vbOutlier_[idx]=false;
                e->setLevel(0);
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        if(optimizer.edges().size()<10)
            break;
    }    

    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    Tcw = vSE3_recov->estimate();
    pFrame->T_c_w_ = Sophus::SE3d(Tcw.rotation(), Tcw.translation());
    for(size_t i = 0; i < pFrame->N_; ++i){
        if(!pFrame->vbOutlier_[i]){
            pFrame->addMapPoint2d(pFrame->vpMapPoints_[i]->id_, pFrame->vKeys_[i].pt);
        }
    }
    
#ifdef VO_DEBUG    
    std::ofstream fou;
    logPath = "/work/data/fr1xyz";
    if (!logPath.empty()) {
        std::string path = logPath + "/opt_" + std::to_string(optId) + ".txt";
        fou.open(path.c_str(), std::ios::out);
    }

    if(fou.good()){
        g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
        fou << "vSE3: " << vSE3_recov->estimate() <<endl;
        auto edges = optimizer.edges();
        for (auto it = edges.begin(); it != edges.end(); ++it) {
            g2o::EdgeSE3ProjectXYZOnlyPose* itc = dynamic_cast<g2o::EdgeSE3ProjectXYZOnlyPose*>(*it);
            itc->computeError();
            Eigen::Matrix3d K;
            K << itc->fx, 0.0, itc->cx, 0.0, itc->fy, itc->cy, 0.0, 0.0, 1.0;
            Eigen::Vector3d pt3(itc->Xw[0], itc->Xw[1], itc->Xw[2]);
            pt3 = K * vSE3_recov->estimate()* pt3;
            Eigen::Vector2d projerr(itc->measurement()[0]-pt3[0]/pt3[2], itc->measurement()[1]-pt3[1]/pt3[2]); 
            fou << itc->measurement().transpose() << " -> "
                << itc->Xw.transpose() << " edge error: "
                << itc->error().transpose() << " proj error: " << projerr.transpose() << endl;
            
        }
        fou.close();
    }
#endif    
    return nInitialCorrespondences-nBad;
}
}
