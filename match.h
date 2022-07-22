//
// Created by zgq on 2022/7/21.
//

#ifndef CVNN3_MATCH_H
#define CVNN3_MATCH_H
#include "iostream"
#include "vector"
#include "opencv2/opencv.hpp"
#include "matching2D.h"
#include "dataStructures.h"
#include "map"


using namespace std;

class Match {
public:
    struct DataFrame { // represents the available sensor information at the same time instance

        cv::Mat cameraImg; // camera image
        std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
        cv::Mat descriptors; // keypoint descriptors
        std::vector<cv::DMatch> kptMatches; // keypoint matches between previous and current frame
//        std::vector<BoundingBoxes> boundingBoxes;
        std::vector<YoloDetSt> yolodetst;
        std::map<int,int> bbMatches; // bounding box matches between previous and current frame
    };

private:
    int dateBufferSize_=2;
    vector<DataFrame> dataBuffer_; // list of data frames which are held in memory at the same time
    enum Descriptors{
        BRIEF, ORB, FREAK, AKAZE, SIFT, BRISK
    };
public:
    void pushFrame(cv::Mat imgGray,const std::vector<YoloDetSt> detst );
    void detect_keyPoints(cv::Mat imgGray);
    void extract_keyPoints();

    void match_keyPoints();

    void matchBoundingBoxes(vector<cv::DMatch> &matches, map<int, int> &bbBestMatches, DataFrame &prevFrame,
                            DataFrame &currFrame);
};


#endif //CVNN3_MATCH_H
