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
//std::vector<YoloDetSt> keepBBs;//the result data foa all process
/*void search(const std::vector<int> As,const std::vector<YoloDetSt> &Xs){
    bool result;
    std::set<int> set_a;
    int k=0;
    for(auto it=Xs.begin();it!=Xs.end();++it){
set_a.insert(it->boxID);
    }
    for(int i=0;i<As.size();i++){
        if(set_a.find(As[i] != set_a.end())){
            set_a.erase(As[i]);
        }
    }

}*/
template <typename T>
std::vector<T> deff_cmp(std::vector<T>& first,std::vector<T>& second){
    std::vector<T> diff;
    std::sort(first.begin(),first.end(),std::less<T>());
    std::sort(second.begin(),second.end(),std::less<T>());
std::set_difference(first.begin(),first.end(),second.begin(),second.end(),std::inserter(diff,diff.begin()));
    return diff;
}

#endif //CVNN3_DATASTRUCTURES_H
