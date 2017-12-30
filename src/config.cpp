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

#include "gslam/config.h"

namespace gslam 
{
    
void Config::setParameterFile( const std::string& filename )
{
    if ( config_ == nullptr )
        config_ = shared_ptr<Config>(new Config);
    config_->file_ = cv::FileStorage( filename.c_str(), cv::FileStorage::READ );
    if ( config_->file_.isOpened() == false )
    {
        std::cerr<<"parameter file "<<filename<<" does not exist."<<std::endl;
        config_->file_.release();
        return;
    }
}

Config::~Config()
{
    if ( file_.isOpened() )
        file_.release();
}

shared_ptr<Config> Config::config_ = nullptr;

}
/*
g2o_core_d.lib
g2o_stuff_d.lib
g2o_solver_csparse_d.lib
g2o_types_slam3d_d.lib
g2o_opengl_helper_d.lib
g2o_csparse_extension_d.lib
g2o_ext_csparse_d.lib
g2o_types_icp_d.lib
g2o_types_sba_d.lib
opencv_core330d.lib
opencv_highgui330d.lib
opencv_imgproc330d.lib
opencv_imgcodecs330d.lib
opencv_calib3d330d.lib
opencv_features2d330d.lib
opencv_xfeatures2d330d.lib
opencv_flann330d.lib
opencv_viz330d.lib
*/