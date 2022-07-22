//
// Created by zgq on 2022/7/21.
//

#ifndef CVNN3_DATASTRUCTURES_H
#define CVNN3_DATASTRUCTURES_H

#include "opencv2/opencv.hpp"
#include "iostream"
 struct YoloDetSt
{
    std::string label;
    float confidences;
    cv::Rect rect;
    int classID; // ID based on class file provided to YOLO framework
    int boxID;

} ;
struct LidarPoint { // single lidar point in space
    double x,y,z,r; // x,y,z in [m], r is point reflectivity
};
struct BoundingBoxes{
    YoloDetSt yoloDetSt;
    std::vector<LidarPoint> lidarPoints; // Lidar 3D points which project into 2D image roi
    std::vector<cv::KeyPoint> keypoints; // keypoints enclosed by 2D roi
    std::vector<cv::DMatch> kptMatches; // keypoint matches enclosed by 2D roi
};
#endif //CVNN3_DATASTRUCTURES_H
