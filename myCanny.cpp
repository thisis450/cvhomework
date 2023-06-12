#include <opencv2/opencv.hpp>
#include <iostream>
#include<cmath>
using namespace cv;
using namespace std;

#define JUST_SHOW_RESULT
//#define SHOW_OPENCV_CANNY


#ifndef JUST_SHOW_RESULT
//设置展示中间过程的宏定义
#define SHOW_EDGE
#define SHOW_DIRECTION
#define SHOW_GRAY
#define SHOW_SOBEL
#define SHOW_MAX
#define SHOW_THRESHOLD
#define SHOW_CONNECT

#endif



#ifdef JUST_SHOW_RESULT
#define SHOW_GRAY
#define SHOW_EDGE
#define SHOW_CANNY
#endif


//设置一些参数
const float lowThreshold = 30/255.0;
const float highThreshold = 95/255.0;
const int cvcannyhigh=200;
const int cvcannylow=100;
bool is_gauss=true;

enum{nonMaximumSuppression_interpolation,nonMaximumSuppression_easy};
void onMouse(int event,int x,int y,int flags,void*userdata)
{
    Mat* img = static_cast<Mat*>(userdata);
    if (event == EVENT_LBUTTONDOWN)
    {
        if (img->type() == CV_8UC1)
        {
            uchar pixel = img->at<uchar>(y, x);
            cout << "Pixel value at (" << x << ", " << y << "): " << static_cast<int>(pixel) << endl;
        }
        else if (img->type() == CV_8UC3)
        {
            Vec3b pixel = img->at<Vec3b>(y, x);
            cout << "Pixel value at (" << x << ", " << y << "): " << pixel << endl;
        }
        else if (img->type() == CV_32FC1)
        {
            float pixel = img->at<float>(y, x);
            cout << "Pixel value at (" << x << ", " << y << "): " << pixel << endl;
        }
        else
        {
            cout<<"wrong type,type is"<<img->type()<<endl;

        }
    }
}
void my_sobel(Mat& src, Mat&grandient,Mat&direction)
{   direction=cv::Mat::zeros(src.size(), CV_32F);
    grandient=cv::Mat::zeros(src.size(), CV_32F);
    Mat x_temp=cv::Mat::zeros(src.size(),CV_16S);
    Mat y_temp=cv::Mat::zeros(src.size(),CV_16S);
    Mat kernel_x = (Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    Mat kernel_y = (Mat_<double>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
    filter2D(src, x_temp, CV_16S, kernel_x);
    filter2D(src, y_temp, CV_16S, kernel_y);

    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            float x = (float)x_temp.at<short>(i, j);
            float y = (float)y_temp.at<short>(i, j);
            float grad = sqrt(x*x + y*y);
            grandient.at<float>(i, j) = (float)grad;
            direction.at<float>(i, j) = atan2(-y, x) * 180 / CV_PI;

        }
    }
    grandient=grandient/255;
    convertScaleAbs(x_temp, x_temp);
    convertScaleAbs(y_temp, y_temp);
#ifdef SHOW_SOBEL
    namedWindow("x_temp");
    setMouseCallback("x_temp", onMouse,&x_temp);
    imshow("x_temp", x_temp);
    namedWindow("y_temp");
    setMouseCallback("y_temp", onMouse,&y_temp);
    imshow("y_temp", y_temp);
    namedWindow("grandient");
    setMouseCallback("grandient", onMouse,&grandient);
    imshow("grandient", grandient);

    waitKey(0);
#endif
#ifdef SHOW_DIRECTION
    //把direction转换到彩色,不同方位对应不同颜色
    Mat colorImg=Mat::zeros(grandient.size(),CV_8UC3);
    for(int i=0;i<grandient.rows;i++)
    {
        for(int j=0;j<grandient.cols;j++)
        {
            if(grandient.at<float>(i,j)<lowThreshold)
                continue;

                Vec3b color = colorImg.at<Vec3b>(i,j);
                float t=direction.at<float>(i,j);
                if((t>=-22.5&&t<=22.5)||(t>=157.5||t<=-157.5))
                {//左右为蓝
                    color[0]=255;
                    color[1]=0;
                    color[2]=0;
                }
                else if((t>=22.5&&t<=67.5)||(t>=-157.5&&t<=-112.5))
                {//右上左下为绿
                    color[0]=0;
                    color[1]=255;
                    color[2]=0;
                }
                else if((t>=67.5&&t<=112.5)||(t>=-112.5&&t<=-67.5))
                {//上下为红
                    color[0]=0;
                    color[1]=0;
                    color[2]=255;
                }
                else if((t>=112.5&&t<=157.5)||(t>=-67.5&&t<=-22.5))
                {//左上右下为青蓝色
                    color[0]=255;
                    color[1]=255;
                    color[2]=0;
                }

                colorImg.at<Vec3b>(i,j)=color;



        }
    }
    //点击图片输出像素值
    namedWindow("colorImg");
    setMouseCallback("colorImg", onMouse,&colorImg);
    imshow("colorImg",colorImg);
    waitKey(0);


#endif
}
void my_nonMaximumSuppression_easy(Mat&dst,Mat&grandient,Mat &direction)
{   dst=grandient.clone();
    for (int i = 1; i < dst.rows - 1; i++)
    {
        for (int j = 1; j < dst.cols - 1; j++)
        {
            float dir = direction.at<float>(i, j);
            float p1, p2;
            if ((dir >= -22.5 && dir <= 22.5)||(dir >= 157.5 || dir <= -157.5))
            {
                p1 = grandient.at<float>(i, j - 1);
                p2 = grandient.at<float>(i, j + 1);
            }
            else if((dir >= 22.5 && dir <= 67.5)||(dir >= -157.5 && dir <= -112.5))
            {
                p1 = grandient.at<float>(i - 1, j + 1);
                p2 = grandient.at<float>(i + 1, j - 1);
            }
            else if((dir >= 67.5 && dir <= 112.5)||(dir >= -112.5 && dir <= -67.5))
            {
                p1 = grandient.at<float>(i - 1, j);
                p2 = grandient.at<float>(i + 1, j);
            }
            else if((dir >= 112.5 && dir <= 157.5)||(dir >= -67.5 && dir <= -22.5))
            {
                p1 = grandient.at<float>(i - 1, j - 1);
                p2 = grandient.at<float>(i + 1, j + 1);
            }
            else
            {
                p1 = grandient.at<float>(i, j - 1);
                p2 = grandient.at<float>(i, j + 1);
            }

            float p = grandient.at<float>(i, j);
            if (p < p1 || p < p2)
            {
                dst.at<float>(i, j) = 0;
            }
        }
    }
    #ifdef SHOW_MAX
    namedWindow("nonMaximumSuppression_easy");
    setMouseCallback("nonMaximumSuppression_easy", onMouse,&dst);
    imshow("nonMaximumSuppression_easy", dst);
    waitKey(0);
    #endif

}
void my_nonMaximumSuppression_interpolation(Mat&dst,Mat&grandient,Mat &direction)
{   dst=grandient.clone();
    for(int i=1;i<grandient.rows-1;i++)
    {
        for(int j=1;j<grandient.cols-1;j++)
        {
            float dir=direction.at<float>(i,j);
            double tan_value = tan(dir * CV_PI / 180);
            float w1=grandient.at<float>(i-1,j-1),//左上
            w2=grandient.at<float>(i-1,j+1),//右上
            w3=grandient.at<float>(i+1,j-1),//左下
            w4=grandient.at<float>(i+1,j+1);//右下
            float value1=(w1+w2)/2+(w2-w1)/tan_value/2;
            float value2=(w3+w4)/2+(w3-w4)/tan_value/2;
            if(fabs(dir * CV_PI / 180)<CV_PI/4)
            {
                value1=(w4+w2)/2+(w2-w4)*tan_value/2;
                value2=(w3+w1)/2+(w3-w1)*tan_value/2;
            }
            if(grandient.at<float>(i,j)>value1&&grandient.at<float>(i,j)>value2)
            {
                dst.at<float>(i,j)=grandient.at<float>(i,j);
            }
            else
            {
                dst.at<float>(i,j)=0;
            }



        }
    }
    #ifdef SHOW_MAX
    namedWindow("nonMaximumSuppression_interpolation");
    setMouseCallback("nonMaximumSuppression_interpolation", onMouse,&dst);
    imshow("nonMaximumSuppression_interpolation", dst);
    waitKey(0);
    #endif
}
void my_threshold(Mat&src,Mat&dst,float highThreshold,float lowThreshold)
{
    dst = src.clone();
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            if (src.at<float>(i, j) > highThreshold)
            {
                dst.at<float>(i, j) = 1.0;
            }
            else if (src.at<float>(i, j) < lowThreshold)
            {
                dst.at<float>(i, j) = 0;
            }
        }
    }
#ifdef SHOW_THRESHOLD

    namedWindow("threshold");
    setMouseCallback("threshold", onMouse,&dst);
    imshow("threshold", dst);
    waitKey(0);
#endif
}
void my_connect_bfs(Mat&src,Mat&dst,float lowThreshold)
{
    dst = Mat::zeros(src.size(), CV_8U);
    Mat flags=Mat::zeros(src.size(),CV_8U);

    #ifdef SHOW_EDGE
    Mat edge=Mat::zeros(src.size(),CV_8UC3);
    #endif




    for (int i = 1; i < src.rows - 1; i++)
    {
        for (int j = 1; j < src.cols - 1; j++)
        {
            if (flags.at<uchar>(i, j) == 0 && src.at<float>(i, j) ==1.0)
            {
                queue<Point> q;
                q.push(Point(i, j));
                #ifdef SHOW_EDGE
                //生成一个随机颜色
                int b = rand() % 256;
                int g = rand() % 256;
                int r = rand() % 256;
                #endif
                while (!q.empty())
                {
                    Point p = q.front();
                    q.pop();
                    dst.at<uchar>(p.x, p.y) = 255;
                    flags.at<uchar>(p.x, p.y) = 1;
                    if(p.x<1||p.y<1||p.x>src.rows-2||p.y>src.cols-2)
                        continue;
                    if (src.at<float>(p.x - 1, p.y) >= lowThreshold && flags.at<uchar>(p.x - 1, p.y) == 0)
                    {
                        q.push(Point(p.x - 1, p.y));
                        flags.at<uchar>(p.x - 1, p.y) = 1;
                    }
                    if (src.at<float>(p.x + 1, p.y) >= lowThreshold && flags.at<uchar>(p.x + 1, p.y) == 0)
                    {
                        q.push(Point(p.x + 1, p.y));
                        flags.at<uchar>(p.x + 1, p.y) = 1;
                    }
                    if (src.at<float>(p.x, p.y - 1) >= lowThreshold && flags.at<uchar>(p.x, p.y - 1) == 0)
                    {
                        q.push(Point(p.x, p.y - 1));
                        flags.at<uchar>(p.x, p.y - 1) = 1;
                    }
                    if (src.at<float>(p.x, p.y + 1) >= lowThreshold && flags.at<uchar>(p.x, p.y + 1) == 0)
                    {
                        q.push(Point(p.x, p.y + 1));
                        flags.at<uchar>(p.x, p.y + 1) = 1;
                    }
                    if (src.at<float>(p.x - 1, p.y - 1) >= lowThreshold && flags.at<uchar>(p.x - 1, p.y - 1) == 0)
                    {
                        q.push(Point(p.x - 1, p.y - 1));
                        flags.at<uchar>(p.x - 1, p.y - 1) = 1;
                    }
                    if (src.at<float>(p.x - 1, p.y + 1) >= lowThreshold && flags.at<uchar>(p.x - 1, p.y + 1) == 0)
                    {
                        q.push(Point(p.x - 1, p.y + 1));
                        flags.at<uchar>(p.x - 1, p.y + 1) = 1;
                    }
                    if (src.at<float>(p.x + 1, p.y - 1) >= lowThreshold && flags.at<uchar>(p.x + 1, p.y - 1) == 0)
                    {
                        q.push(Point(p.x + 1, p.y - 1));
                        flags.at<uchar>(p.x + 1, p.y - 1) = 1;
                    }
                    if (src.at<float>(p.x + 1, p.y + 1) >= lowThreshold && flags.at<uchar>(p.x + 1, p.y + 1) == 0)
                    {
                        q.push(Point(p.x + 1, p.y + 1));
                        flags.at<uchar>(p.x + 1, p.y + 1) = 1;
                    }
                    #ifdef SHOW_EDGE
                    edge.at<Vec3b>(p.x,p.y)[0]=b;
                    edge.at<Vec3b>(p.x,p.y)[1]=g;
                    edge.at<Vec3b>(p.x,p.y)[2]=r;
                    #endif




                }
               
        }
        flags.at<uchar>(i,j)=1;
        }
            
    }
#ifdef SHOW_CONNECT
    namedWindow("connect");
    setMouseCallback("connect", onMouse,&dst);
    imshow("connect", dst);
    waitKey(0);
#endif
    #ifdef SHOW_EDGE
    namedWindow("edge");
    setMouseCallback("edge", onMouse,&edge);
    imshow("edge", edge);
    waitKey(0);
    #endif
}

void my_nonMaximumSuppression(Mat&dst,Mat&grandient,Mat& direction,int func)
{
    switch (func)
    {
    case nonMaximumSuppression_easy:
        my_nonMaximumSuppression_easy(dst, grandient, direction);
        break;
    case nonMaximumSuppression_interpolation:
        my_nonMaximumSuppression_interpolation(dst, grandient, direction);
        break;
    default:
        break;
    }
}


void my_canny(
    Mat& src, 
    Mat& dst,
    float highThreshold,
    float lowThreshold,
    int nonMaximumSuppression_func=nonMaximumSuppression_easy,
    bool is_gauss=false,
    int gauss_ksize=3)
{   //灰度图
    Mat tmp;
    cvtColor(src, tmp, COLOR_BGR2GRAY);
    #ifdef SHOW_GRAY
    imshow("gray", tmp);
    waitKey(0);
    #endif
    //高斯滤波
    if(is_gauss)
    {
        GaussianBlur(tmp, tmp, Size(gauss_ksize, gauss_ksize), 0, 0);
    }
    //梯度计算
    Mat direction;
    Mat grandient;
    my_sobel(tmp, grandient,direction);
    //非极大值抑制
    Mat Suppression;
    my_nonMaximumSuppression(Suppression,grandient,direction,nonMaximumSuppression_func);
    //双阈值
    Mat threshold;
    my_threshold(Suppression,threshold,highThreshold,lowThreshold);
    //边缘连接
    Mat connect;
    my_connect_bfs(threshold, connect,lowThreshold);
    dst = connect;
#ifdef SHOW_CANNY
    imshow("canny", dst);
    waitKey(0);
#endif

#ifdef SHOW_OPENCV_CANNY
    Mat dst2;
    Canny(src, dst2, cvcannylow, cvcannyhigh);
    imshow("opencv_canny", dst2);
    waitKey(0);
#endif    

}

int main(int argc, char** argv)
{
    Mat src = imread("images/cones/im2.png");
    Mat dst;
    my_canny(src, dst,highThreshold,lowThreshold,nonMaximumSuppression_easy,is_gauss);


    return 0;
}
