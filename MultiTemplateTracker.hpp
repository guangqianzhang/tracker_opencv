#pragma once
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "tracker.hpp"

using namespace std;
using namespace cv;
namespace Track
{
    class MultiTemplateTracker : public Track::Tracker
    {
    public:
        enum MatchMethod
        {
            SQDIFF = 0,
            SADIFF = 1
        };
        enum MatchStrategy
        {
            UNIFORM = 0,
            NORMAL = 1
        };
        struct Params
        {

            int expandWidth;             //局部扩展
            MatchStrategy matchStrategy; //匹配策略 1 随机；0 局部
            MatchMethod matchMethod;     //匹配方法
            double alpha;                //模板更新速度
            int numPoints;               //随机采样点数
            Point2i sigma;               //正态分布标准差
            Vec2i xyStep;                //模板内采样不长
            Vec2i xyStride;              //模板在图像内滑动不长
            Params()
            {
                expandWidth = 50;
                matchMethod = MatchMethod::SADIFF;
                matchStrategy = MatchStrategy::NORMAL;
                alpha = 0.7;
                numPoints = 500;
                sigma = Point2d(0.5, 0.5);
                xyStep = Vec2i(2, 2);
                xyStride = Vec2i(1, 1);
            }
        };
        // Mat TargetTemplate;                    //目标模板
        vector<Mat> MultiScaleTargetTemplates; //多尺度目标模板
        Rect CurrentBoundingBox;               //当前帧找到的目标框
        Mat CurrentTargetPatch;                //当前帧找到的图像块
        Rect FrameArea;                        //视频帧矩形
        Rect NextSearchArea;                   //下一帧搜索范围
        vector<Point2d> SamplePoints;          //标准正态分布采样点
        Params params;

    private:
        float MatchTemplate(const Mat &src, const Mat &temp, Rect2i &match_location,
                            MatchMethod match_method, Vec2i &xy_step, Vec2i &xy_stride)
        {
            CV_Assert((src.type() == CV_8UC1) && (temp.type() == CV_8UC1));
            //图像 模板尺寸
            int src_width = src.cols;
            int src_height = src.rows;
            int temp_clos = temp.cols;
            int temp_rows = temp.rows;
            int y_end = src_height - temp_rows + 1;
            int x_end = src_width - temp_clos + 1;

            //记录最优位置
            float match_dgree = FLT_MAX;
            int y_match = -1, x_match = -1;
            //扫描
            for (int y = 0; y < y_end; y += xy_stride[1])
            {
                for (int x = 0; x < x_end; x += xy_stride[0])
                {
                    //匹配读计算
                    float match_yx = 0.0f;
                    //对其模板到src，累加模板内像素点差异
                    for (int r = 0; r < temp_rows; r += xy_step[1])
                    {
                        for (int c = 0; c < temp_clos; c += xy_step[0])
                        {
                            uchar src_val = src.ptr<uchar>(y + r)[x + c];
                            uchar temp_val = temp.ptr<uchar>(r)[c];
                            if (match_method == MatchMethod::SQDIFF) // SODIFF
                                match_yx += float(std::abs(src_val - temp_val) * std::abs(src_val - temp_val));
                            if (match_method == MatchMethod::SADIFF) // SADIFF
                                match_yx += float(std::abs(src_val - temp_val));
                        }
                    }
                    //更新
                    if (match_dgree > match_yx)
                    {
                        match_dgree = match_yx;
                        x_match = x;
                        y_match = y;
                    }
                }
            }
            match_location = Rect2i(x_match, y_match, temp_clos, temp_rows);

            return match_dgree;
        }
        float MatchTemplate(const Mat &src, const Mat &temp, Rect2i &match_location,
                            MatchMethod match_method, const vector<Point2d> &sample_points)
        {

            CV_Assert((src.type() == CV_8UC1) && (temp.type() == CV_8UC1));
            //图像 模板尺寸
            int src_width = src.cols;
            int src_height = src.rows;
            int temp_clos = temp.cols;
            int temp_rows = temp.rows;
            int y_end = src_height - temp_rows + 1;
            int x_end = src_width - temp_clos + 1;

            //缩放
            vector<Point2i> Sample_Points(sample_points.size());
#pragma omp parallel for
            for (size_t k = 0; k < sample_points.size(); k++)
            {
                const Point2d &ptd = sample_points[k];
                Point2i &pti = Sample_Points[k];
                pti.x = cvRound(ptd.x * temp_clos);
                pti.y = cvRound(ptd.y * temp_rows);
            }
            //记录最佳匹配
            float match_dgree = FLT_MAX;
            int y_match = -1, x_match = -1;
#pragma omp parallel for
            //扫描
            for (int y = 0; y < y_end; y++)
            {
                for (int x = 0; x < x_end; x++)
                {
                    //匹配读计算
                    float match_yx = 0.0f;
                    //按照采样点数组计算模板与原始图像匹配度
                    for (size_t k = 0; k < sample_points.size(); k++)
                    {
                        Point2i &pt = Sample_Points[k];
                        uchar src_val = src.ptr<uchar>(y + pt.y)[x + pt.x];
                        uchar temp_val = temp.ptr<uchar>(pt.y)[pt.x];
                        if (match_method == MatchMethod::SQDIFF)
                        {
                            match_yx += float(std::abs(src_val - temp_val) * std::abs(src_val - temp_val));
                        }
                        if (match_method == MatchMethod::SADIFF)
                        {
                            match_yx += float(std::abs(src_val - temp_val));
                        }
                    }
                    //更新
                    if (match_dgree > match_yx)
                    {
                        match_dgree = match_yx;
                        x_match = x;
                        y_match = y;
                    }
                }
            }
            match_location = Rect2i(x_match, y_match, temp_clos, temp_rows);
            return match_dgree;
        }
        //估计下一帧范围
        void EstimateSearchArea(const Rect &target_location, Rect &search_area, int expend_x, int expend_y)
        {
            float center_x = target_location.x + 0.5f * target_location.width;
            float center_y = target_location.y + 0.5f * target_location.height;
            search_area.width = target_location.width + expend_x;
            search_area.height = target_location.height + expend_y;
            search_area.x = int(center_x - 0.5f * search_area.width);
            search_area.y = int(center_y - 0.5f * search_area.height);
            search_area &= this->FrameArea;
        }
        //产生截断正态分布点击
        void GenerateRandomSamplePoints(vector<Point2d> &sample_points, int num_points = 1000, Point2d sigma = Point2d(0.3, 0.3))
        {
            RNG rng = theRNG();
            Rect2d sample_area(0.0, 0.0, 1.0, 1.0);
            for (int k = 0; k < num_points;)
            {
                Point2d pt;
                pt.x = sample_area.width / 2 + rng.gaussian(sigma.x);
                pt.y = sample_area.height / 2 + rng.gaussian(sigma.y);
                if (sample_area.contains(pt))
                {
                    sample_points.push_back(pt);
                    k++;
                }
            }
        }

        //产生多尺度模板
        void GenerateMultiScaleTargetTemplates(const Mat &origin_target, vector<Mat> &multiscale_target)
        {
            vector<double> resize_scales = {/* 1.5, 1.4, 1.3, */ 1.2, 1.1, 1.0, 0.9, 0.8 /* , 0.7, 0.6, 0.5 */};
            multiscale_target.resize(resize_scales.size(), Mat());
            for (size_t scidx = 0; scidx < resize_scales.size(); scidx++)
            {
                cv::resize(origin_target, multiscale_target[scidx], Size(),
                           resize_scales[scidx], resize_scales[scidx], InterpolationFlags::INTER_AREA);
            }
            return;
        }
        //显示多尺度模板
        void ShowMultiScaleTemplate(const vector<Mat> &multiscale_target)
        {
            int total_cols = 0, total_rows = 0;
            vector<Rect2i> target_rois(multiscale_target.size());
            for (size_t k = 0; k < multiscale_target.size(); k++)
            {
                target_rois[k] = Rect2i(total_cols, 0, multiscale_target[k].cols, multiscale_target[k].rows);
                total_cols += multiscale_target[k].cols;
                total_rows = max(multiscale_target[k].rows, total_rows);
            }
            Mat targetsImg = Mat::zeros(total_rows, total_cols, CV_8UC1);
            for (size_t k = 0; k < multiscale_target.size(); k++)
            {
                multiscale_target[k].copyTo(targetsImg(target_rois[k]));
            }
            imshow("Targets Image", targetsImg);
            waitKey(100);
        }
        //使用多尺度模板匹配
        float MatchMultiScaleTemplates(const Mat &src, const vector<Mat> &multiscale_templs, Rect2i &best_match_location,
                                       MatchMethod match_method, const vector<Point2d> &sample_points, 
                                       MatchStrategy match_strategy ,Vec2i& xy_step,Vec2i& xy_stride)
        {
            CV_Assert(match_strategy == 0 || match_strategy == 1);

            float bestMatchDgree = FLT_MAX;
            Rect bestMatchLocation;
            Rect matchLocation;
            float matchDgree;

            for (size_t scaleIdx = 0; scaleIdx < multiscale_templs.size(); scaleIdx++)
            {
                const Mat &templ = multiscale_templs[scaleIdx];
                if (match_strategy == MatchStrategy::UNIFORM)
                {

                    matchDgree = this->MatchTemplate(src, templ, matchLocation, match_method, xy_step, xy_stride);
                }
                if (match_strategy == MatchStrategy::NORMAL)
                {
                    matchDgree = this->MatchTemplate(src, templ, matchLocation, match_method, sample_points);
                }
                //记录最佳匹配
                if (matchDgree < bestMatchDgree)
                {
                    bestMatchDgree = matchDgree;
                    bestMatchLocation = matchLocation;
                }
            } // endof scaleIdx
            best_match_location = bestMatchLocation;
            return bestMatchDgree;
        }
        //更新多尺度模板库
        void UpdateMultiScaleTemplates(const Mat& currentTargetPatch){
            for(size_t idx=0;idx<this->MultiScaleTargetTemplates.size();idx++){
                if(this->MultiScaleTargetTemplates[idx].size()==currentTargetPatch.size()){
                    cv::addWeighted(this->MultiScaleTargetTemplates[idx],this->params.alpha,currentTargetPatch,1-this->params.alpha,
                    0.0,this->MultiScaleTargetTemplates[idx]);
                }
            }
        }

    public:
        MultiTemplateTracker(Params _params);
        ~MultiTemplateTracker();

    public:
        bool init(const Mat &initFrame, const Rect &initBoundingBox)
        {
            cout << "Go SingleTemplateTracker::init!!" << endl;
            this->FrameArea = Rect(0, 0, initFrame.cols, initFrame.rows);
            cout << "initBoundingBox" << initBoundingBox.size() << endl;
            //初始帧模板
            Mat TargetTemplate = initFrame(initBoundingBox).clone();
            cout << "TargetTemplate" << TargetTemplate.size() << endl;

            //产生多尺度目标模板
            this->GenerateMultiScaleTargetTemplates(TargetTemplate, this->MultiScaleTargetTemplates);
            this->ShowMultiScaleTemplate(this->MultiScaleTargetTemplates);
            //下一帧搜索范围
            this->EstimateSearchArea(initBoundingBox, this->NextSearchArea, this->params.expandWidth, this->params.expandWidth);
            //初始化随机采样
            this->GenerateRandomSamplePoints(this->SamplePoints, this->params.numPoints, this->params.sigma);

            return false;
        }
        bool track(const Mat &currentFrame, Rect &currentBoundingBox)
        {
            cout << "Go SingleTemplateTracker::track!!" << endl;
            // matching tempale 全帧匹配
            Rect2i match_location(-1, -1,0,0);
            //搜索目标
            /* //单模板匹配
                        if (this->params.matchStrategy == MatchStrategy::UNIFORM)
                        {

                            this->MatchTemplate(currentFrame(this->NextSearchArea), this->TargetTemplate, match_location,
                                                this->params.matchMethod, this->params.xyStep, this->params.xyStride);
                        }
                        if (this->params.matchStrategy == MatchStrategy::NORMAL)
                        {
                            this->MatchTemplate(currentFrame(this->NextSearchArea), this->TargetTemplate,
                                                match_location, this->params.matchMethod, this->SamplePoints);
                        }
             */
            //多模板匹配
            this->MatchMultiScaleTemplates(currentFrame(this->NextSearchArea),this->MultiScaleTargetTemplates,match_location,
            this->params.matchMethod,this->SamplePoints,this->params.matchStrategy,this->params.xyStep,this->params.xyStride);
            //调整匹配点位置
            match_location.x += this->NextSearchArea.x;
            match_location.y += this->NextSearchArea.y;
            //计算当前位置目标框
            this->CurrentBoundingBox = match_location;
            //抓取当前帧的目标图像块
            cout << "CurrentBoundingBox" << CurrentBoundingBox.size() << endl;
            this->CurrentTargetPatch = currentFrame(this->CurrentBoundingBox).clone();

            currentBoundingBox = this->CurrentBoundingBox;
            return false;
        }
        bool update(Rect& searchBox)
        {
            cout << "Go SingleTemplateTracker::update!!" << endl;
            //更新目标表面特征模型 t(k+1)=aT(k)+bT(k-1)

        //    cv::addWeighted(this->TargetTemplate, this->params.alpha, this->CurrentTargetPatch, 1.0 - this->params.alpha, 0.0, this->TargetTemplate);
            this->UpdateMultiScaleTemplates(this->CurrentTargetPatch);
            //更新下一帧上的搜索范围
            this->EstimateSearchArea(this->CurrentBoundingBox, this->NextSearchArea, this->params.expandWidth, this->params.expandWidth);
            //输出局部搜索范围
            searchBox=this->NextSearchArea;
            return false;
        }
    };

    MultiTemplateTracker::MultiTemplateTracker(Params _params)
    {
        this->params = _params;
    }

    MultiTemplateTracker::~MultiTemplateTracker()
    {
    }
    typedef MultiTemplateTracker MTTracker;

}
