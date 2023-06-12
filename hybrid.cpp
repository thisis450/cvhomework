
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;
int clickCount = 0;
void myfilter2D(Mat& src, Mat& dst, Mat& kernel)
{
    dst.create(src.size(), src.type());
    int rows = src.rows;
    int cols = src.cols;
    int krows = kernel.rows;
    int kcols = kernel.cols;
    int kcenterX = kcols / 2;
    int kcenterY = krows / 2;

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            double sum = 0.0;
            for (int m = 0; m < krows; ++m)
            {
                for (int n = 0; n < kcols; ++n)
                {
                    int ii = i + (m - kcenterY);
                    int jj = j + (n - kcenterX);
                    if (ii >= 0 && ii < rows && jj >= 0 && jj < cols)
                    {
                        sum += src.at<uchar>(ii, jj) * kernel.at<double>(m, n);
                    }
                }
            }
            dst.at<uchar>(i, j) = saturate_cast<uchar>(sum);
        }
    }
}
void filter2DMultiChannel(Mat& src, Mat& dst, Mat& kernel)
{
    vector<Mat> channels;
    split(src, channels);
    vector<Mat> resultChannels(channels.size());
    for (size_t i = 0; i < channels.size(); ++i)
    {
        myfilter2D(channels[i], resultChannels[i], kernel);
    }
    merge(resultChannels, dst);
}
class match_class
{
    
public:
    int i1x1,i1y1,i1x2,i1y2,i2x1,i2y1,i2x2,i2y2;
    match_class(int i1x1,int i1y1,int i1x2,int i1y2,int i2x1,int i2y1,int i2x2,int i2y2)
    {
        this->i1x1=i1x1;
        this->i1y1=i1y1;
        this->i1x2=i1x2;
        this->i1y2=i1y2;
        this->i2x1=i2x1;
        this->i2y1=i2y1;
        this->i2x2=i2x2;
        this->i2y2=i2y2;
    }


};
match_class matchpoint(0,0,0,0,0,0,0,0);
void onMouse(int event, int x, int y, int flags, void* userdata)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        if(clickCount==0)
        {
            matchpoint.i1x1=x;
            matchpoint.i1y1=y;
            clickCount++;
        }
        else if(clickCount==1)
        {
            matchpoint.i1x2=x;
            matchpoint.i1y2=y;
            clickCount++;
            cv::destroyWindow("Image1");
        }
        else if(clickCount==2)
        {
            matchpoint.i2x1=x;
            matchpoint.i2y1=y;
            clickCount++;
        }
        else if(clickCount==3)
        {
            matchpoint.i2x2=x;
            matchpoint.i2y2=y;
            clickCount++;
            cv::destroyWindow("Image2");
        }
    }
}
void set_center(Mat& img,Mat& dst,int x,int y)
{
    int dx = (x - img.cols / 2)*2;
    int dy = (y - img.rows / 2)*2;
    int bottom = dy > 0 ? dy : 0;
    int top = dy < 0 ? -dy : 0;
    int right = dx > 0 ? dx : 0;
    int left = dx < 0 ? -dx : 0;
    cv::copyMakeBorder(img, dst, top, bottom, left, right, cv::BORDER_CONSTANT);
    //cout<<img.size()<<endl;
}
void change_size(Mat& img1,Mat& img2,float distance1,float distance2)
{
    float scale=distance1/distance2;
    int width=img2.cols*scale;
    int height=img2.rows*scale;
    resize(img2,img2,Size(width,height));
    
}
void rotate(Mat& img,Mat& dst,float k1,float k2)
{
    float angle=atan((k2-k1)/(1+k1*k2))*180/3.1415926;
    Point2f center(img.cols/2,img.rows/2);
    Mat rot_mat=getRotationMatrix2D(center,angle,1);
    warpAffine(img,dst,rot_mat,img.size());
}
float calcu_k(int x1,int y1,int x2,int y2)
{
    float k=(float)(y2-y1)/(x2-x1);
    return k;
}
float calcu_distance(int x1,int y1,int x2,int y2)
{
    float distance=sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
    return distance;
}
void cut_img_width(Mat& img1,Mat& img2,Mat& dst1,Mat& dst2)
{
    if(img1.size().width<img2.size().width)
    {
        int width=img1.size().width;
        int dw=img2.size().width-width;
        Rect rect(dw/2,0,img2.size().width-dw,img2.size().height);
        Mat img2_roi=img2(rect);
        img2_roi.copyTo(dst2);
    }
    else if(img1.size().width>=img2.size().width)
    {
        int width=img2.size().width;
        int dw=img1.size().width-width;
        Rect rect(dw/2,0,img1.size().width-dw,img1.size().height);
        Mat img1_roi=img1(rect);
        img1_roi.copyTo(dst1);
    }
    
}
void cut_img_height(Mat& img1,Mat& img2,Mat& dst1,Mat& dst2)
{
    if(img1.size().height<img2.size().height)
    {
        int height=img1.size().height;
        int dh=img2.size().height-height;
        Rect rect(0,dh/2,img2.size().width,img2.size().height-dh);
        Mat img2_roi=img2(rect);
        img2_roi.copyTo(dst2);
    }
    else if(img1.size().height>=img2.size().height)
    {
        int height=img2.size().height;
        int dh=img1.size().height-height;
        Rect rect(0,dh/2,img1.size().width,img1.size().height-dh);
        Mat img1_roi=img1(rect);
        img1_roi.copyTo(dst1);
    }
    
}
void match_size(Mat& img1,Mat& img2)
{
    float k1=calcu_k(matchpoint.i1x1,matchpoint.i1y1,matchpoint.i1x2,matchpoint.i1y2);
    float k2=calcu_k(matchpoint.i2x1,matchpoint.i2y1,matchpoint.i2x2,matchpoint.i2y2);
    float distance1=calcu_distance(matchpoint.i1x1,matchpoint.i1y1,matchpoint.i1x2,matchpoint.i1y2);
    float distance2=calcu_distance(matchpoint.i2x1,matchpoint.i2y1,matchpoint.i2x2,matchpoint.i2y2);
    set_center(img1,img1,matchpoint.i1x1/2+matchpoint.i1x2/2,matchpoint.i1y1/2+matchpoint.i1y2/2);
    set_center(img2,img2,matchpoint.i2x1/2+matchpoint.i2x2/2,matchpoint.i2y1/2+matchpoint.i2y2/2);
    change_size(img1,img2,distance1,distance2);
    rotate(img2,img2,k1,k2);
    cut_img_width(img1,img2,img1,img2);
    cut_img_height(img1,img2,img1,img2);
    cout<<img1.size()<<endl;
    cout<<img2.size()<<endl;
    //resize (img1, img1, img2.size(), 0, 0, INTER_LINEAR);
    
} 
void high_freq_imgarea(string method,Mat src,Mat dst)
{
    if(method=="gauss")
    {
    int ksize=min(src.cols,src.rows)/5;
    if(ksize%2==0)
        ksize=ksize+1;
    Mat keX = getGaussianKernel(ksize, 0);
    Mat keY = getGaussianKernel(ksize, 0);
    Mat kernel=keX*keY.t();
    //filter2DMultiChannel(src,dst,kernel);
    GaussianBlur(src, dst, Size(ksize,ksize), 0, 0);
    dst=src-dst;
    return;
    }
    else if(method=="laplace1")
    {
    Mat laplace1_kernal=(Mat_<double>(3,3)<<-1,-1,-1,-1,8,-1,-1,-1,-1);
    filter2DMultiChannel(src,dst,laplace1_kernal);
    //filter2D ( src, dst, src.depth(), laplace1_kernal);
    return;
    }
    else if (method=="laplace2")
    {
    Mat laplace2_kernal=(Mat_<double>(3,3)<<0,-1,0,-1,4,-1,0,-1,0);
    filter2DMultiChannel(src,dst,laplace2_kernal);
    //filter2D ( src, dst, src.depth(), laplace2_kernal);
    return;
    }
    else if (method=="mean")
    {
     Mat mean_kernal=(Mat_<double>(3,3)<<1.0/9,1.0/9,1.0/9,1.0/9,1.0/9,1.0/9,1.0/9,1.0/9,1.0/9);
     filter2DMultiChannel(src,dst,mean_kernal);
    //filter2D ( src, dst, src.depth(), mean_kernal );
    dst=src-dst;
    return;

    }
    else if(method=="median")
    {
    medianBlur(src, dst, 9);
    dst=src-dst;
    dst=dst*2;
    return;
    }
        else if(method=="laplace3")
    {
    Mat laplace3_kernal=(Mat_<double>(5,5)<<1,1,1,1,1,1,1,1,1,1,1,1,-25,1,1,1,1,1,1,1,1,1,1,1,1);
    filter2DMultiChannel(src,dst,laplace3_kernal);
    //filter2D ( src, dst, src.depth(), laplace1_kernal);
    return;
    }

return;
}
void low_freq_imgarea(string method,Mat src,Mat dst)
{
if (method=="mean")
{
 Mat mean_kernal=(Mat_<double>(3,3)<<1.0/9,1.0/9,1.0/9,1.0/9,1.0/9,1.0/9,1.0/9,1.0/9,1.0/9);
    filter2DMultiChannel(src,dst,mean_kernal);
    //filter2D ( src, dst, src.depth(), mean_kernal );
    return;
}
else if(method=="gauss")
{
    int ksize=min(src.cols,src.rows)/5;
    if(ksize%2==0)
        ksize=ksize+1;
    Mat keX = getGaussianKernel(ksize, 0);
    Mat keY = getGaussianKernel(ksize, 0);
    Mat kernel=keX*keY.t();
    //filter2DMultiChannel(src,dst,kernel);
    GaussianBlur(src, dst, Size(ksize,ksize), 0, 0);
    return;
}
else if(method=="median")
{
    medianBlur(src, dst, 7);
    dst=dst;
    return;
}
    else if(method=="laplace1")
    {
    Mat laplace1_kernal=(Mat_<double>(3,3)<<-1,-1,-1,-1,8,-1,-1,-1,-1);
    filter2DMultiChannel(src,dst,laplace1_kernal);
    //filter2D ( src, dst, src.depth(), laplace1_kernal);
    dst=src-dst;
    return;
    }
    else if (method=="laplace2")
    {
    Mat laplace2_kernal=(Mat_<double>(3,3)<<0,-1,0,-1,4,-1,0,-1,0);
    filter2DMultiChannel(src,dst,laplace2_kernal);
    //filter2D ( src, dst, src.depth(), laplace2_kernal);
    dst=src-dst;
    return;
    }
}
void get_match_class(Mat img1,Mat img2)
{
return;
}
void gui_get_match_class(Mat img1,Mat img2)
{
    clickCount=0;
    cv::namedWindow("Image1");
    cv::setMouseCallback("Image1", onMouse);
    cv::imshow("Image1", img1);
    cv::namedWindow("Image2");
    cv::setMouseCallback("Image2", onMouse);
    cv::imshow("Image2", img2);
    cv::waitKey(0);
    cout<<matchpoint.i1x1<<' '<<matchpoint.i1y1<<' '<<matchpoint.i1x2<<' '<<matchpoint.i1y2<<endl;
    cout<<matchpoint.i2x1<<' '<<matchpoint.i2y1<<' '<<matchpoint.i2x2<<' '<<matchpoint.i2y2<<endl;
return;
}
void hybrid_image(string img_path1,string img_path2)
{
Mat img1 = imread(img_path1,IMREAD_GRAYSCALE);
Mat img2 = imread(img_path2,IMREAD_GRAYSCALE);
string temp;
char model;
cout<<"是否需要调整图像(y/n)"<<endl;
cin>>model;
if(model=='y')
{   
    cout<<"调整方法1为手动2为模板匹配"<<endl;
    cin>>model;
    if(model=='1')
    {
        gui_get_match_class(img1,img2);
        match_size(img1,img2);
    }
    else if(model=='2')
    {
        get_match_class(img1,img2);
        match_size(img1,img2);
    }
    else
    {
        cout<<"错误的调整方法"<<endl;
        return;
    }
    cout<<"是否需要展示调整结果(y/n)"<<endl;
    cin>>model;
    if(model=='y')
    {
        imshow("调整后图像1",img1);
        imshow("调整后图像2",img2);
        //cout<<img1.size()<<endl;
        //cout<<img2.size()<<endl;
        waitKey(0);
    }

}
else
{
    resize (img2, img2, img1.size(), 0, 0, INTER_LINEAR);
}
Mat img1_high=img1.clone();
Mat img2_low=img2.clone();
cout<<"高频图像滤波方法"<<endl;
cin>>temp;
high_freq_imgarea(temp,img1,img1_high);
cout<<"低频图像滤波方法"<<endl;
cin>>temp;
low_freq_imgarea(temp,img2,img2_low);
cout<<"是否展示处理后图像(y/n)"<<endl;
cin>>model;
resize (img1_high, img1_high, img2_low.size(), 0, 0, INTER_LINEAR);
if(model=='y')
{
    imshow("处理后图像1",img1_high);
    imshow("处理后图像2",img2_low);
}
Mat img3=img1_high*0.5+img2_low*0.5;
imshow("混合图像",img3);
Mat far;
resize(img3, far, cv::Size(), 0.5, 0.5);
imshow("远处混合图片",far);
Mat farfar;
resize(far, farfar, cv::Size(), 0.5, 0.5);
imshow("更远处混合图片",farfar);
waitKey(0);
}
int main()
{
    string img_path1,img_path2;
     img_path1="high1.png";
     img_path2="low1.png";
     cout<<"输入图片路径1"<<endl;
     cin>>img_path1;
     cout<<"输入图片路径2"<<endl;
     cin>>img_path2;
    hybrid_image(img_path1,img_path2);
    return 0;

}

