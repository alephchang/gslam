

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

        //1. detect features with FAST
        //FastFeatureDetector fast;
        Ptr<SURF> fea = SURF::create();
        Ptr<ORB_SLAM2::ORBextractor> orb_fea = new ORB_SLAM2::ORBextractor(1000,1.2,8,20,7);
        vector<KeyPoint> keyPts0, keyPts1;
        fea->detect(img0, keyPts0);  
        fea->detect(img1, keyPts1);  
        vector<KeyPoint> orb_keyPts0, orb_keyPts1;
        Mat orb_desc0, orb_desc1;
        (*orb_fea)(img0, cv::Mat(), orb_keyPts0, orb_desc0);
        (*orb_fea)(img1, cv::Mat(), orb_keyPts1, orb_desc1);
        
        cout << "img0--number of keypoints: " << keyPts0.size() << endl;
        cout << "img1--number of keypoints: " << keyPts1.size() << endl;
        cout << "img0--number of keypoints: " << orb_keyPts0.size() << endl;
        cout << "img1--number of keypoints: " << orb_keyPts1.size() << endl;

        //2. detect and compute feature descriptor
        Mat desc0, desc1;
        fea->compute(img0, keyPts0, desc0); //detectAndCompute(img0, Mat(), keyPts0, desc0);
        fea->compute(img1, keyPts1, desc1); //detectAndCompute(img1, Mat(), keyPts1, desc1);
        
        //3. matching
        //BruteForceMatcher< L2<float> > matcher; //FlannBasedMatcher matcher;  
        FlannBasedMatcher matcher;
        vector< DMatch > matches, orb_matches;
        cout<<desc0.size <<endl;
        cout << orb_desc0.size <<endl;
        if(orb_desc0.type() != CV_32F){
            orb_desc0.convertTo(orb_desc0, CV_32F);
            orb_desc1.convertTo(orb_desc1, CV_32F);
        }
        matcher.match(orb_desc0, orb_desc1, orb_matches);
        matcher.match(desc0, desc1, matches);
        cout << "number of matches: " << matches.size() << endl;
        cout << "number of orb matches: " << orb_matches.size() << endl;

        double max_dist = 0; double min_dist = 100;
        //-- Quick calculation of max and min distances between keypoints
        for (int i = 0; i < desc0.rows; i++){
                double dist = matches[i].distance;
                if (dist < min_dist) min_dist = dist;
                if (dist > max_dist) max_dist = dist;
        }
        std::vector< DMatch > good_matches;
        for (int i = 0; i < desc0.rows; i++){
                if (matches[i].distance <= max(2 * min_dist, 0.02)){
                        good_matches.push_back(matches[i]);
                }
        }
        max_dist = 0;  min_dist = 100;
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


void homographyRansac(int argc, char** argv)
{
        //1.load a pair of images
        if (argc < 3) {
                std::cout << "Pleae specify two images!" << std::endl;
                return;
        }
        Mat img0 = imread(argv[1], IMREAD_GRAYSCALE);
        Mat img1 = imread(argv[2], IMREAD_GRAYSCALE);
        if (img0.empty() || img1.empty()) {
                std::cout << "Fail to load the two images!" << std::endl;
                return;
        }
        //2. detect and compute features
        Ptr<Feature2D> orb = ORB::create();
        vector<KeyPoint> keyPts0, keyPts1;
        Mat desc0, desc1;
        orb->detectAndCompute(img0, Mat(), keyPts0, desc0);
        orb->detectAndCompute(img1, Mat(), keyPts1, desc1);

        if (desc0.type() != CV_32F) desc0.convertTo(desc0, CV_32F);
        if (desc1.type() != CV_32F) desc1.convertTo(desc1, CV_32F);
        //3. match
        //FlannBasedMatcher matcher;
        //Ptr<cv::DescriptorMatcher> matcher(new cv::BFMatcher(cv::NORM_HAMMING, true));
        Ptr<cv::DescriptorMatcher> matcher(new FlannBasedMatcher);
        vector< DMatch > matches;
        matcher->match(desc0, desc1, matches);

        double max_dist = 0; double min_dist = 100;
        //-- Quick calculation of max and min distances between keypoints
        for (int i = 0; i < matches.size(); i++) {
                double dist = matches[i].distance;
                if (dist < min_dist) min_dist = dist;
                if (dist > max_dist) max_dist = dist;
        }

        std::vector< DMatch > good_matches;
        for (int i = 0; i < matches.size(); i++) {
                if (matches[i].distance <= max(3 * min_dist, 0.02)) {
                        good_matches.push_back(matches[i]);
                }
        }
        cout << "Good Matches Size: " << good_matches.size() << endl;
        const int minNumberMatchesAllowed = 4;
        if (good_matches.size() < minNumberMatchesAllowed) {
                cout << "Feature matches are too few" << endl;
                return;
        }

        vector<Point2f> pts0, pts1;
        for(size_t i = 0; i < good_matches.size(); ++i){
                pts0.push_back(keyPts0[good_matches[i].queryIdx].pt);
                pts1.push_back(keyPts1[good_matches[i].trainIdx].pt);
        }
        float reprojectionThreshold = 1.0;
        std::vector<unsigned char> inliersMask(matches.size());
        Mat H = cv::findHomography(pts0,
                pts1,
                CV_FM_RANSAC,
                reprojectionThreshold,
                inliersMask);
        std::vector<cv::DMatch> inliers;
        for (size_t i = 0; i<inliersMask.size(); i++)
        {
                if (inliersMask[i])
                        inliers.push_back(matches[i]);
        }
        cout << "match size: " << good_matches.size() << "\t"
                << "inliers size: " << inliers.size() << endl;
        Mat imgRlt(img0.size(), img0.type());
        warpPerspective(img0, imgRlt, H, imgRlt.size());
        if(argc>3)
                imwrite(argv[3], imgRlt);
}

int main(int argc, char** argv)
{    
    detectAndMatchFeatures(argc, argv);
    return 0;
}
