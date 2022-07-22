#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include "Yolo_plug.hpp"
#include "tracker.hpp"
#include "MultiTemplateTracker.hpp"

#include "match.h"
using namespace cv;
using namespace std;
namespace global
{
    bool isRoiReady = false;
    bool isImgReady= false;
    bool selectObject = false;
    Rect selectedRoi;
    std::vector<YoloDetSt> trackRet;

}
int main(int argc, char *argv[])
{
    std::cout << "Hello, World!" << std::endl;
    cout << argv[1] << endl;
    int cap_num_ = atoi(argv[1]);
    VideoCapture capture(cap_num_);
    boost::shared_ptr<Yolo> yolo_;
    yolo_.reset(new Yolo);
    if (!capture.isOpened())
    {
        printf("could not read this video file...\n");
        return -1;
    }
    Size S = Size((int)capture.get(CAP_PROP_FRAME_WIDTH),
                  (int)capture.get(CAP_PROP_FRAME_HEIGHT));
    const Rect FrameArea(0, 0, S.width, S.height);
    int fps = capture.get(CAP_PROP_FPS);
    printf("current fps : %d \n", fps);
    std::cout << "size:" << S << std::endl;
    Mat frame;
    Mat currentFrame;
    Mat workFrame;
    const string winName = "Trackering";
    namedWindow(winName, 2);
    capture >> frame;

    CV_Assert(!frame.empty());
    global::isImgReady=true;
    cout << "声明跟踪对象实例化，初始化目标跟踪器。。。" << endl;
//    Track::MTTracker::Params mtparams = Track::MTTracker::Params();
//    Ptr<Track::Tracker> tracker =new Track::MTTracker(mtparams);
    Match match;
    while (global::isImgReady)
    {
        // frame.copyTo(displayImg_);
        frame.copyTo(workFrame);
         std::vector<YoloDetSt> yoloRet;
        cv::Mat current_frame;
        current_frame = yolo_->detect(workFrame, yoloRet);
        //获取 jiancekuang
        global::trackRet=yoloRet;
        cv::Mat imgGray;
        cv::cvtColor(frame,imgGray,cv::COLOR_BGR2GRAY);
        match.pushFrame(imgGray,global::trackRet);
        match.detect_keyPoints(imgGray);
        match.extract_keyPoints();
        match.match_keyPoints();


        imshow(winName, current_frame);
        capture >> frame;

        waitKey(10);
    }
    return 0;
}
