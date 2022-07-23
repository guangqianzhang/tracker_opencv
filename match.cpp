//
// Created by zgq on 2022/7/21.
//

#include "match.h"

void Match::pushFrame(cv::Mat imgGray, const std::vector<YoloDetSt> detst) {
    DataFrame frame;
    frame.yolodetst = detst;
    frame.cameraImg = imgGray;
    if (dataBuffer_.size() < dateBufferSize_) {
        dataBuffer_.push_back(frame);
        cout << "LOAD IMAGE INTO BUFFER done" << endl;
    } else {
        dataBuffer_.erase(dataBuffer_.begin());
        dataBuffer_.push_back(frame);
        cout << "REPLACE IMAGE IN BUFFER done" << endl;
    }


    cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;
}

void Match::detect_keyPoints(cv::Mat imgGray) {

    // extract 2D keypoints from current image
    vector<cv::KeyPoint> keypoints; // create empty feature list for current image
    string detectorType = "SHITOMASI";  // -> SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

    if (detectorType.compare("SHITOMASI") == 0) {
        detKeypointsShiTomasi(keypoints, imgGray, false);
    } else if (detectorType.compare("HARRIS") == 0) {
        detKeypointsHarris(keypoints, imgGray, false);
    } else {
        detKeypointsModern(keypoints, imgGray, detectorType, false);
    }

    // only keep keypoints on the preceding vehicle
    bool bFocusOnVehicle = true;
    vector<cv::KeyPoint>::iterator keypoint;
    vector<cv::KeyPoint> keypoints_roi;

    std::vector<YoloDetSt> detRct = (dataBuffer_.end() - 1)->yolodetst;

    if (bFocusOnVehicle) {
        for (auto it = detRct.begin(); it != detRct.end(); ++it) {
            std::cout << it->label << std::endl;
            const cv::Rect roi = it->rect;
            for (keypoint = keypoints.begin(); keypoint != keypoints.end(); ++keypoint) {
                if (roi.contains(keypoint->pt)) {
                    cv::KeyPoint newKetPoint;;
                    newKetPoint.pt = cv::Point2f(keypoint->pt);
                    newKetPoint.size = 1;
                    keypoints_roi.push_back(newKetPoint);
                }
            }

            cout << "IN ROI n= " << keypoints_roi.size() << " keypoints" << endl;
        }
        //all obj in keyPoints
        keypoints = keypoints_roi;
    }
    // optional : limit number of keypoints (helpful for debugging and learning)
    bool bLimitKpts = false;
    if (bLimitKpts) {
        int maxKeypoints = 200;

        if (detectorType.compare("SHITOMASI") ==
            0) { // there is no response info, so keep the first 50 as they are sorted in descending quality order
            keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
        }
        cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
        cout << " NOTE: Keypoints have been limited!" << endl;
    }
    // push keypoints and descriptor for current frame to end of data buffer
    (dataBuffer_.end() - 1)->keypoints = keypoints;
    cout << "#2 : DETECT KEYPOINTS done" << endl;

}

void Match::extract_keyPoints() {
    /* EXTRACT KEYPOINT DESCRIPTORS */

    //// STUDENT ASSIGNMENT
    //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
    //// -> BRIEF, ORB, FREAK, AKAZE, SIFT
    cv::Mat descriptors;
    string descriptorType = "BRISK"; // BRIEF, ORB, FREAK, AKAZE, SIFT, BRISK
    descKeypoints((dataBuffer_.end() - 1)->keypoints, (dataBuffer_.end() - 1)->cameraImg, descriptors, descriptorType);
    //// EOF STUDENT ASSIGNMENT
    // push descriptors for current frame to end of data buffer
    (dataBuffer_.end() - 1)->descriptors = descriptors;
    cout << "#3 : EXTRACT DESCRIPTORS done" << endl;
}

void Match::match_keyPoints(std::vector<YoloDetSt> &trackRet) {
    size_t currSize = (dataBuffer_.end() - 1)->yolodetst.size();
    size_t prevSize = (dataBuffer_.end() - 2)->yolodetst.size();
    int detSize = max(currSize, prevSize);
    if (dataBuffer_.size() > 1) {
        /* MATCH KEYPOINT DESCRIPTORS */

        vector<cv::DMatch> matches;
        string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
        string descriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG
        string selectorType = "SEL_NN";       // SEL_NN, SEL_KNN

        //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
        //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

        matchDescriptors((dataBuffer_.end() - 2)->keypoints, (dataBuffer_.end() - 1)->keypoints,
                         (dataBuffer_.end() - 2)->descriptors, (dataBuffer_.end() - 1)->descriptors,
                         matches, descriptorType, matcherType, selectorType);



        // store matches in current data frame
        (dataBuffer_.end() - 1)->kptMatches = matches;
        cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;



        // visualize matches between current and previous image
        bool bVis = false;
        if (bVis) {
            cv::Mat matchImg = ((dataBuffer_.end() - 1)->cameraImg).clone();
            cv::drawMatches((dataBuffer_.end() - 2)->cameraImg, (dataBuffer_.end() - 2)->keypoints,
                            (dataBuffer_.end() - 1)->cameraImg, (dataBuffer_.end() - 1)->keypoints,
                            matches,
                            matchImg,
                            cv::Scalar::all(-1), cv::Scalar::all(-1),
                            vector<char>(), cv::DrawMatchesFlags::DEFAULT);

            string windowName = "Matching keypoints between two camera images";
            cv::namedWindow(windowName, 7);
            cv::imshow(windowName, matchImg);

            //if(imgIndex == 2)
            //{
            //    string Img_Name = "../images/fast_brisk.png";
            //    imwrite(Img_Name, matchImg);
            //}

            cout << "Press key to continue to next image" << endl;
            cv::waitKey(10); // wait for key to be pressed
        }
        bVis = false;

        std::map<int, int> bbBestMatches;
        // associate bounding boxes between current and previous frame using keypoint matches
        matchBoundingBoxes(matches, bbBestMatches, *(dataBuffer_.end() - 2), *(dataBuffer_.end() -1));


        (dataBuffer_.end() - 1)->bbMatches = bbBestMatches;

        // init  think about the condition of init and the part of track!!
        if(trackRet.size()==0){
            trackRet=(dataBuffer_.end()-2)->yolodetst;
        }

/*        std::vector<int> PreVecId=getVector((dataBuffer_.end()-2)->yolodetst);
        std::vector<int> CurVecId=getVector((dataBuffer_.end()-1)->yolodetst);
        std::vector<int> MatchSecondId;
        std::vector<int> theDiffId;
        for (auto it3 = (dataBuffer_.end() - 1)->bbMatches.begin();
             it3 != (dataBuffer_.end() - 1)->bbMatches.end(); ++it3) {
            MatchSecondId.push_back(it3->second);
        }*/

        //        add some
        if(currSize>trackRet.size()) {
            if(prevSize<currSize){
//                trackRet=(dataBuffer_.end()-2)->yolodetst;

//                theDiffId=deff_cmp(CurVecId,MatchSecondId);
//                for(size_t i=0;i<theDiffId.size();i++)
//                    trackRet.push_back((dataBuffer_.end()-1)->yolodetst[i]);
            }
        }
//            delete some
        else if(currSize<trackRet.size()){
//            trackRet=(dataBuffer_.end()-2)->yolodetst;
//            if(it1->second==0){}
        }


        // loop over all BB match pairs
        std::vector<YoloDetSt> currBBs, preBBs;
        for (auto it1 = (dataBuffer_.end() - 1)->bbMatches.begin();
             it1 != (dataBuffer_.end() - 1)->bbMatches.end(); ++it1) {
            // find bounding boxes associates with current match
            for (size_t it2 = 0; it2 <(dataBuffer_.end()-2)->yolodetst.size(); ++it2) {
                if (trackRet[it2].boxID == it1->first) {
                    if (trackRet[it2].label == (dataBuffer_.end() - 1)->yolodetst[it1->second].label) {
                        std::cout << "label match!!" << endl;
                        trackRet[it2].rect = (dataBuffer_.end() - 1)->yolodetst[it1->second].rect;
                    }

                }
            }

/*//            for (auto it2 = (dataBuffer_.end() - 1)->yolodetst.begin();
//                                 it2 != (dataBuffer_.end() - 1)->yolodetst.end(); ++it2) {
//                if (it1->second == it2->boxID) // check wether current match partner corresponds to this BB
//                {
//                    currBB = &(*it2);
//                    currBBs.push_back(*currBB);
//                }
//            }
//
//            for (auto it2 = (dataBuffer_.end() - 2)->yolodetst.begin();
//                                  it2 != (dataBuffer_.end() - 2)->yolodetst.end(); ++it2) {
//                if (it1->first == it2->boxID) // check wether current match partner corresponds to this BB
//                {
//                    prevBB = &(*it2);
//                    preBBs.push_back(*prevBB);
//                }
//            }*/


        }

//        this->matchShow(currBBs,preBBs);
        this->matchShow(trackRet);


    }


}

void Match::matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches,
                               DataFrame &prevFrame, DataFrame &currFrame) {
/*    // NOTE: After calling a cv::DescriptorMatcher::match function, each DMatch
    // contains two keypoint indices, queryIdx and trainIdx, based on the order of image arguments to match.
    // https://docs.opencv.org/4.1.0/db/d39/classcv_1_1DescriptorMatcher.html#a0f046f47b68ec7074391e1e85c750cba
    // prevFrame.keypoints is indexed by queryIdx
    // currFrame.keypoints is indexed by trainIdx*/
    int p = prevFrame.yolodetst.size();
    int c = currFrame.yolodetst.size();
    int pt_counts[p][c] = {};
    for (auto it = matches.begin(); it != matches.end(); ++it) {
        cv::KeyPoint query = prevFrame.keypoints[it->queryIdx];
        auto query_pt = cv::Point(query.pt.x, query.pt.y);
        bool query_found = false;

        cv::KeyPoint train = currFrame.keypoints[it->trainIdx];
        auto train_pt = cv::Point(train.pt.x, train.pt.y);
        bool train_found = false;

        std::vector<int> query_id, train_id;
        for (int i = 0; i < p; i++) {
            if (prevFrame.yolodetst[i].rect.contains(query_pt)) {
                query_found = true;
                query_id.push_back(i);
            }
        }
        for (int i = 0; i < c; i++) {
            if (currFrame.yolodetst[i].rect.contains(train_pt)) {
                train_found = true;
                train_id.push_back(i);
            }
        }

        if (query_found && train_found) {
            for (auto id_prev: query_id)
                for (auto id_curr: train_id)
                    pt_counts[id_prev][id_curr] += 1;
        }
    }
    for (int i = 0; i < p; i++) {
        int max_count = 0;
        int id_max = 0;
        for (int j = 0; j < c; j++)//choose the
            if (pt_counts[i][j] > max_count) {
                max_count = pt_counts[i][j];
                id_max = j;
            }
        bbBestMatches[i] = id_max;
    }

}

void Match::matchShow(const std::vector<YoloDetSt> currBBs) {


    if (currBBs.size() != 0) {
        cv::Mat visImg = (dataBuffer_.end() - 1)->cameraImg.clone();
//                showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImg);
        for (auto it = currBBs.begin(); it != currBBs.end(); ++it) {
            cv::rectangle(visImg, cv::Point(it->rect.x, it->rect.y),
                          cv::Point(it->rect.x + it->rect.width, it->rect.y + it->rect.height),
                          cv::Scalar(255, 255, 0), 2);

            std::string str = it->label + cv::format(": %d", it->boxID);
            int baseLine;
            cv::Size labelSize = cv::getTextSize(str, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            int top = cv::max(it->rect.y, labelSize.height);
            putText(visImg, str, cv::Point(it->rect.x, top), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 0, 255));
        }
        string windowName = "Final Results : tracking";
        cv::namedWindow(windowName, 4);
        cv::imshow(windowName, visImg);
        cv::waitKey(10);
    }

}

void Match::matchShow(const std::vector<YoloDetSt> currBBs, const std::vector<YoloDetSt> ptreBBs) {


    if (currBBs.size() != 0) {
        cv::Mat visImg = (dataBuffer_.end() - 1)->cameraImg.clone();
//                showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImg);
        for (auto it = currBBs.begin(); it != currBBs.end(); ++it) {
            cv::rectangle(visImg, cv::Point(it->rect.x, it->rect.y),
                          cv::Point(it->rect.x + it->rect.width, it->rect.y + it->rect.height),
                          cv::Scalar(255, 255, 0), 2);

            std::string str = it->label + cv::format(": %d", it->boxID);
            int baseLine;
            cv::Size labelSize = cv::getTextSize(str, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            int top = cv::max(it->rect.y, labelSize.height);
            putText(visImg, str, cv::Point(it->rect.x, top), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 0, 255));
        }
        for (auto it = ptreBBs.begin(); it != ptreBBs.end(); ++it) {
            cv::rectangle(visImg, cv::Point(it->rect.x, it->rect.y),
                          cv::Point(it->rect.x + it->rect.width, it->rect.y + it->rect.height),
                          cv::Scalar(0, 255, 0), 2);

            std::string str = it->label + cv::format(": %d", it->boxID);
            int baseLine;
            cv::Size labelSize = cv::getTextSize(str, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            int top = cv::max(it->rect.y, labelSize.height);
            putText(visImg, str, cv::Point(it->rect.x + labelSize.width + 10, top), cv::FONT_HERSHEY_PLAIN, 2,
                    cv::Scalar(0, 255, 0));
        }
        string windowName = "Final Results : tracking";
        cv::namedWindow(windowName, 4);
        cv::imshow(windowName, visImg);
        cout << "Press key to continue to next frame" << endl;
        cv::waitKey(10);
    }

}