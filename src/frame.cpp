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
#include "gslam/frame.h"
#include <boost/concept_check.hpp>

namespace gslam
{
shared_ptr<ORBVocabulary>  Frame::pORBvocab_=nullptr;
Frame::Frame()
: id_(-1), time_stamp_(-1), camera_(nullptr)
{

}

Frame::Frame ( long id, double time_stamp, SE3<double> T_c_w, Camera::Ptr camera, Mat color, Mat depth )
: id_(id), time_stamp_(time_stamp), T_c_w_(T_c_w), camera_(camera), color_(color), depth_(depth) 
{

}

Frame::~Frame()
{

}

Frame::Ptr Frame::createFrame()
{
    static long factory_id = 0;
    return Frame::Ptr( new Frame(factory_id++) );
}

double Frame::findDepth ( const cv::KeyPoint& kp )
{
    int x = cvRound(kp.pt.x);
    int y = cvRound(kp.pt.y);
    ushort d = depth_.ptr<ushort>(y)[x];
    if ( d!=0 )
    {
        return double(d)/camera_->depth_scale_;
    }
    else 
    {
        // check the nearby points 
        int dx[4] = {-1,0,1,0};
        int dy[4] = {0,-1,0,1};
        for ( int i=0; i<4; i++ )
        {
            d = depth_.ptr<ushort>( y+dy[i] )[x+dx[i]];
            if ( d!=0 )
            {
                return double(d)/camera_->depth_scale_;
            }
        }
    }
    return -1.0;
}

void Frame::setPose ( const SE3<double>& T_c_w )
{
    T_c_w_ = T_c_w;
}


Vector3d Frame::getCamCenter() const
{
    return T_c_w_.inverse().translation();
}

bool Frame::isInFrame ( const Vector3d& pt_world )
{
    Vector3d p_cam = camera_->world2camera( pt_world, T_c_w_ );
    // cout<<"P_cam = "<<p_cam.transpose()<<endl;
    if ( p_cam(2,0)<0 ) return false;
    Vector2d pixel = camera_->world2pixel( pt_world, T_c_w_ );
    return pixel(0,0)>0 && pixel(1,0)>0 
        && pixel(0,0)<color_.cols 
        && pixel(1,0)<color_.rows;
}
bool Frame::isInFrustum(MapPoint::Ptr pMp)
{
    pMp->track_in_view_ = false;
    Vector3d P = pMp->pos_;
    Vector3d Pc = T_c_w_*P;
    const double &PcX = Pc(0);
    const double &PcY= Pc(1);
    const double &PcZ = Pc(2);
    if(PcZ<0.0) return false;
    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    const float u=camera_->fx_*PcX*invz+camera_->cx_;
    const float v=camera_->fy_*PcY*invz+camera_->cy_;
    if(u<0.0 || u > static_cast<double>(color_.cols)) return false;
    if(v<0.0 || v > static_cast<double>(color_.rows)) return false;
    
    Vector3d Pn = pMp->norm_;
    assert(std::abs<double>(Pn.norm()-1.0)<1e-10);
    Vector3d Ow = -T_c_w_.rotationMatrix().transpose()*T_c_w_.translation();
    Vector3d PO = P - Ow;
    const double viewCos = PO.dot(Pn)/PO.norm();
    if(viewCos < 0.5) return false;
    pMp->track_in_view_ = true;
    pMp->track_proj_x_ = u;
    pMp->track_proj_y_ = v;
    pMp->track_view_cos_= viewCos;  
    return true;
}

vector<size_t> Frame::getFeaturesInAera(float x, float y, float r) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N_);
    for(size_t i = 0; i < N_; ++i){
        const cv::KeyPoint &kp = vKeys_[i];
        float dx = kp.pt.x - x;
        float dy = kp.pt.y - y;
        if(fabs(dx)<r && fabs(dy) < r)
            vIndices.push_back(i);
    }
    return vIndices;
}




void Frame::addMapPoint(MapPoint::Ptr pMp, size_t i)
{
    vpMapPoints_[i] = pMp;
}
std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors)
{
    std::vector<cv::Mat> vDesc;
    vDesc.reserve(Descriptors.rows);
    for (int j=0;j<Descriptors.rows;j++)
        vDesc.push_back(Descriptors.row(j));

    return vDesc;
}

void Frame::computeBoW()
{
    if(BowVec_.empty())
    {
        vector<cv::Mat> vCurrentDesc = toDescriptorVector(descriptors_);
        pORBvocab_->transform(vCurrentDesc,BowVec_,featVec_,4);
    }
}
}//namespace