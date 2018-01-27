

#include<opencv/highgui.h>
#include<opencv/cxcore.h>
#include<opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include<opencv2/xfeatures2d/nonfree.hpp>
#include <boost/concept_check.hpp>
#include<iostream>
#include "../../ORB_SLAM2/include/ORBextractor.h"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

void detectAndMatchFeatures(int argc, char** argv)
{
        if (argc < 3) {
                std::cout << "Pleae specify two images!" << std::endl;
                return ;
        }
        Mat img0 = imread(argv[1], IMREAD_GRAYSCALE);
        Mat img1 = imread(argv[2], IMREAD_GRAYSCALE);
        if (img0.empty() || img1.empty()) {
                std::cout << "Fail to load the two images!" << std::endl;
                return;
        }

        //1. detect features with ORB
        //FastFeatureDetector fast;
        Ptr<ORB_SLAM2::ORBextractor> orb_fea = new ORB_SLAM2::ORBextractor(1000,1.2,8,20,7);
        vector<KeyPoint> orb_keyPts0, orb_keyPts1;
        Mat orb_desc0, orb_desc1;
        (*orb_fea)(img0, cv::Mat(), orb_keyPts0, orb_desc0);
        (*orb_fea)(img1, cv::Mat(), orb_keyPts1, orb_desc1);
        
        cout << "img0--number of keypoints: " << orb_keyPts0.size() << endl;
        cout << "img1--number of keypoints: " << orb_keyPts1.size() << endl;

        //3. matching
        FlannBasedMatcher matcher;
        vector< DMatch > orb_matches;
        if(orb_desc0.type() != CV_32F){
            orb_desc0.convertTo(orb_desc0, CV_32F);
            orb_desc1.convertTo(orb_desc1, CV_32F);
        }
        matcher.match(orb_desc0, orb_desc1, orb_matches);
        cout << "number of orb matches: " << orb_matches.size() << endl;

        double max_dist = 0; double min_dist = 100;
        for (int i = 0; i < orb_desc0.rows; i++){
                double dist = orb_matches[i].distance;
                if (dist < min_dist) min_dist = dist;
                if (dist > max_dist) max_dist = dist;
        }
        std::vector< DMatch > orb_good_matches;
        for (int i = 0; i < orb_desc0.rows; i++){
                if (orb_matches[i].distance <= max(5 * min_dist, 0.02)){
                        orb_good_matches.push_back(orb_matches[i]);
                }
        }
        //4. show result
        Mat matchImg;
        drawMatches(img0, orb_keyPts0, img1, orb_keyPts1, orb_good_matches, matchImg,
                Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        imshow("matching result", matchImg);
        imwrite("match_result.png", matchImg);
        waitKey(0);
}

int main(int argc, char** argv)
{    
    detectAndMatchFeatures(argc, argv);
    return 0;
}
