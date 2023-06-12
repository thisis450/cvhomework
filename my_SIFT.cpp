#include <opencv2/opencv.hpp>
#include <iostream>
#include<cmath>
#include <opencv2/core/hal/hal.hpp>

using namespace cv;
using namespace std;
//#define DEBUG

int imgn=0;

void on_mouse(int event,int x,int y,int flags,void* param)
{
    Mat* img=(Mat*)param;
    switch(event)
    {
        case EVENT_LBUTTONDOWN:
        {
            cout<<"("<<x<<","<<y<<")"<<endl;
            //输出像素点的值，Mat类型为CV_32F
            cout<<img->at<float>(y,x)<<endl;
            break;
        }
        case EVENT_MOUSEMOVE:
        {
            break;
        }
        case EVENT_LBUTTONUP:
        {
            break;
        }
    }
}

void gauss_gpyr(Mat& srcimg,vector<vector<Mat>>& gpyrs,double sigma=1.6,int n=3)
{

Mat first_img;
//转换为浮点型
srcimg.convertTo(first_img,CV_32F);
//扩大到一倍
resize(first_img,first_img,Size(first_img.cols*2,first_img.rows*2));
//高斯金字塔组数
int O=log2(min(first_img.cols,first_img.rows))-3;
//归一化到0-1
first_img=first_img/255;

//每一组的分层数
int S=n+3;
//清空gpyrs，并初始化内容
gpyrs.clear();
for(int i=0;i<O;i++)
{
    gpyrs.push_back(vector<Mat>());
}
    // 计算每个高斯图对应的Sigma
    float sigma_now, sigma_pre;
    float sigma0 = sigma;
    float k = pow(2.0f, 1.0f / n);
    std::vector<float> sig(S);
    sigma_pre = 0.5;
    sig[0] = sqrtf(sigma0 * sigma0 - sigma_pre * sigma_pre);
    // 计算Octave组内每层的尺度坐标
    for (int i = 1; i < S; i++) {
        sigma_pre = pow(k, (float)(i - 1)) * sigma0;
        sigma = sigma_pre * k;
        sig[i] = sqrtf(sigma * sigma - sigma_pre * sigma_pre);
    }
    for(int i=0;i<O;i++)
    {
        for(int j=0;j<S;j++)
        {
            Mat temp;
            if(i==0&&j==0)
            {
                GaussianBlur(first_img,temp,Size(0,0),sig[j],sig[j]);
            }
            else if(j==0)
            {
                //上一组的最后一张图
                Mat pre_last;
                resize(gpyrs[i-1][S-3],pre_last,Size(gpyrs[i-1][S-3].cols/2,gpyrs[i-1][S-3].rows/2));
                temp=pre_last;
            }
            else
            {
                //高斯滤波
                GaussianBlur(gpyrs[i][j-1],temp,Size(0,0),sig[j],sig[j]);
            }
            gpyrs[i].push_back(temp);
        }
    }

}

void generate_dog(vector<vector<Mat>>& gpyrs,vector<vector<Mat>>& dog)
{
    //清空dog
    dog.clear();
    //生成dog
    for(int i=0;i<gpyrs.size();i++)
    {
        dog.push_back(vector<Mat>());
        for(int j=0;j<gpyrs[i].size()-1;j++)
        {
            Mat temp;
            subtract(gpyrs[i][j+1],gpyrs[i][j],temp);
            dog[i].push_back(temp);
        }
    }
}
void generate_keypoints_roughly(vector<vector<Mat>>& dog,vector<KeyPoint>& keypoints)
{
    for(int i=0;i<dog.size();i++)
    {
        for(int j=1;j<dog[i].size()-1;j++)
        {

            for(int m=1;m<dog[i][j].cols-1;m++)
            {
                for(int n=1;n<dog[i][j].rows-1;n++)
                {
                    if(abs(dog[i][j].at<float>(n,m))<0.02)
                        continue;
                    //判断是否比周围的26个点大或小
                    bool ismax=true;
                    bool ismin=true;
                    for(int p=-1;p<=1;p++)
                    {
                        for(int q=-1;q<=1;q++)
                        {
                            for(int r=-1;r<=1;r++)
                            {
                                if(p==0&&q==0&&r==0)
                                    continue;
                                if(dog[i][j].at<float>(n,m)<=dog[i][j+p].at<float>(n+q,m+r))
                                    ismax=false;
                                if(dog[i][j].at<float>(n,m)>=dog[i][j+p].at<float>(n+q,m+r))
                                    ismin=false;
                            }
                        }
                    }
                    //如果是极值点，就加入到keypoints中
                    if(ismax||ismin)
                    {
                        KeyPoint kp;
                        kp.pt.x=m;
                        kp.pt.y=n;
                        //用octave和size来表示这个点在哪一组哪一层
                        kp.octave=i;
                        kp.size=j;
                        kp.response=abs(dog[i][j].at<float>(n,m));
                        keypoints.push_back(kp);
                    }



                }
            }
            #ifdef DEBUG
            //把keypoints中octave为i,size为j的特征点画在图像
            Mat temp=dog[i][j].clone();
            temp=temp*255;
            cvtColor(temp,temp,COLOR_GRAY2BGR);
            for(int k=0;k<keypoints.size();k++)
            {
                if(keypoints[k].octave==i&&keypoints[k].size==j)
                {
                    circle(temp,keypoints[k].pt,2,Scalar(0,0,255));
                }
            }
            string name="dog"+(i*10+j);
            imshow(name,temp);
            waitKey(0);
            destroyAllWindows();

            #endif
     
        }
    }
}
class gradient_1
{
public:
    float dx;
    float dy;
    float ds;
};
gradient_1 compute_gradient_at_keypoint(vector<vector<Mat>>& dog,KeyPoint kp)
{
    float dx=(dog[kp.octave][kp.size].at<float>(kp.pt.y,kp.pt.x+1)-dog[kp.octave][kp.size].at<float>(kp.pt.y,kp.pt.x-1))/2;
    float dy=(dog[kp.octave][kp.size].at<float>(kp.pt.y+1,kp.pt.x)-dog[kp.octave][kp.size].at<float>(kp.pt.y-1,kp.pt.x))/2;
    float ds=(dog[kp.octave][kp.size+1].at<float>(kp.pt.y,kp.pt.x)-dog[kp.octave][kp.size-1].at<float>(kp.pt.y,kp.pt.x))/2;
    gradient_1 g;
    g.dx=dx;
    g.dy=dy;
    g.ds=ds;
    return g;
}
class gradient_2
{
public:
    float dxx;
    float dyy;
    float dss;
    float dxy;
    float dxs;
    float dys;
};
gradient_2 compute_hessian_at_keypoint(vector<vector<Mat>>& dog,KeyPoint kp)
{
    float dxx=dog[kp.octave][kp.size].at<float>(kp.pt.y,kp.pt.x+1)+dog[kp.octave][kp.size].at<float>(kp.pt.y,kp.pt.x-1)-2*dog[kp.octave][kp.size].at<float>(kp.pt.y,kp.pt.x);
    float dyy=dog[kp.octave][kp.size].at<float>(kp.pt.y+1,kp.pt.x)+dog[kp.octave][kp.size].at<float>(kp.pt.y-1,kp.pt.x)-2*dog[kp.octave][kp.size].at<float>(kp.pt.y,kp.pt.x);
    float dss=dog[kp.octave][kp.size+1].at<float>(kp.pt.y,kp.pt.x)+dog[kp.octave][kp.size-1].at<float>(kp.pt.y,kp.pt.x)-2*dog[kp.octave][kp.size].at<float>(kp.pt.y,kp.pt.x);
    float dxy=(dog[kp.octave][kp.size].at<float>(kp.pt.y+1,kp.pt.x+1)-dog[kp.octave][kp.size].at<float>(kp.pt.y+1,kp.pt.x-1)-dog[kp.octave][kp.size].at<float>(kp.pt.y-1,kp.pt.x+1)+dog[kp.octave][kp.size].at<float>(kp.pt.y-1,kp.pt.x-1))/4;
    float dxs=(dog[kp.octave][kp.size+1].at<float>(kp.pt.y,kp.pt.x+1)-dog[kp.octave][kp.size+1].at<float>(kp.pt.y,kp.pt.x-1)-dog[kp.octave][kp.size-1].at<float>(kp.pt.y,kp.pt.x+1)+dog[kp.octave][kp.size-1].at<float>(kp.pt.y,kp.pt.x-1))/4;
    float dys=(dog[kp.octave][kp.size+1].at<float>(kp.pt.y+1,kp.pt.x)-dog[kp.octave][kp.size+1].at<float>(kp.pt.y-1,kp.pt.x)-dog[kp.octave][kp.size-1].at<float>(kp.pt.y+1,kp.pt.x)+dog[kp.octave][kp.size-1].at<float>(kp.pt.y-1,kp.pt.x))/4;
    gradient_2 g;
    g.dxx=dxx;
    g.dyy=dyy;
    g.dss=dss;
    g.dxy=dxy;
    g.dxs=dxs;
    g.dys=dys;
    return g;
}
void generate_keypoints_accurately(vector<vector<Mat>>& dog,vector<KeyPoint>&keypoint_rough,vector<KeyPoint>& keypoints_accurate,int n=3)
{
for(int i=0;i<keypoint_rough.size();i++)
{   int x=keypoint_rough[i].pt.x;
    int y=keypoint_rough[i].pt.y;
    int octave=keypoint_rough[i].octave;
    float size=keypoint_rough[i].size;
    float response;
    Mat offset(3,1,CV_32FC1);
    Mat hessian(3,3,CV_32FC1);
    Mat gradient(3,1,CV_32FC1);
    bool flag=true;
    for(int n=0;n<5;n++)
    {
    KeyPoint kp;
    kp.pt.x=x;
    kp.pt.y=y;
    kp.octave=octave;
    kp.size=size;
    if(x<1||x>dog[octave][int(size)].cols-2||y<1||y>dog[octave][int(size)].rows-2||size<1||size>dog[octave].size()-2)
    {   
        break;
    }
    //求一阶导和二阶导
    gradient_1 g1=compute_gradient_at_keypoint(dog,kp);
    gradient_2 g2=compute_hessian_at_keypoint(dog,kp);
    //求hessian矩阵
    hessian.at<float>(0,0)=g2.dxx;
    hessian.at<float>(0,1)=g2.dxy;
    hessian.at<float>(0,2)=g2.dxs;
    hessian.at<float>(1,0)=g2.dxy;
    hessian.at<float>(1,1)=g2.dyy;
    hessian.at<float>(1,2)=g2.dys;
    hessian.at<float>(2,0)=g2.dxs;
    hessian.at<float>(2,1)=g2.dys;
    hessian.at<float>(2,2)=g2.dss;
    
    gradient.at<float>(0,0)=g1.dx;
    gradient.at<float>(1,0)=g1.dy;
    gradient.at<float>(2,0)=g1.ds;
    //求偏移量
    
    offset=-hessian.inv()*gradient;
    if(abs(offset.at<float>(0,0))<0.5&&abs(offset.at<float>(1,0))<0.5&&abs(offset.at<float>(2,0))<0.5)
    {
        response=dog[octave][int(size)].at<float>(y,x);
        x=kp.pt.x;
        y=kp.pt.y;
        size=kp.size;
        octave=kp.octave;
        flag=false;
        break;
    }
    else
    {
        //如果偏移量大于0.5，那么就要重新计算一阶导和二阶导，大于0.5的时候视为1
        if(offset.at<float>(0,0)>=0.5)
        {
            x++;
        }
        else if(offset.at<float>(0,0)<=-0.5)
        {
            x--;
        }
        if(offset.at<float>(1,0)>=0.5)
        {
            y++;
        }
        else if(offset.at<float>(1,0)<=-0.5)
        {
            y--;
        }
        if(offset.at<float>(2,0)>=0.5)
        {
            size++;
        }
        else if(offset.at<float>(2,0)<=-0.5)
        {
            size--;
        }
    
    }
    }
    if(flag)
    {
        continue;
    }
    //计算响应值
    response+=0.5*(gradient.at<float>(0,0)*offset.at<float>(0,0)+gradient.at<float>(1,0)*offset.at<float>(1,0)+gradient.at<float>(2,0)*offset.at<float>(2,0));
    //存入keypoints_accurate
    if(abs(response)*n<0.04)
    {
        continue;
    }
    KeyPoint kp;
    kp.pt.x=x+offset.at<float>(0,0);
    kp.pt.y=y+offset.at<float>(1,0);
    kp.octave=octave;
    kp.size=size+offset.at<float>(2,0);
    kp.response=response;
    keypoints_accurate.push_back(kp);
}
}
void eliminate_edge_response(vector<vector<Mat>>& dog,vector<KeyPoint>& keypoints_accurate,vector<KeyPoint>& keypoints_eer,int r=10)
{
//消除边缘响应
for(int i=0;i<keypoints_accurate.size();i++)
{
    gradient_2 g2=compute_hessian_at_keypoint(dog,keypoints_accurate[i]);
    float trace=g2.dxx+g2.dyy;
    float det=g2.dxx*g2.dyy-g2.dxy*g2.dxy;
    if(trace*trace*r<det*(r+1)*(r+1))
    {
        KeyPoint eerkp;
        eerkp.pt.x=keypoints_accurate[i].pt.x;
        eerkp.pt.y=keypoints_accurate[i].pt.y;
        eerkp.octave=keypoints_accurate[i].octave;
        eerkp.size=keypoints_accurate[i].size;
        eerkp.response=keypoints_accurate[i].response;
        keypoints_eer.push_back(eerkp);
        
    }
}
}

void generate_keypoints_direction(vector<vector<Mat>>& gpyrs,vector<KeyPoint>& keypoints_eer,vector<KeyPoint>& keypoints_direction)
{
    for(int i=0;i<keypoints_eer.size();i++)
    {
        int octave=keypoints_eer[i].octave;
        int size=int(keypoints_eer[i].size);
        float kpt_scale=pow(2,float(size)/3)*1.6;
        float sigma=1.5*kpt_scale;
        int x=keypoints_eer[i].pt.x+0.5;
        int y=keypoints_eer[i].pt.y+0.5;
        int r=int(4.5*kpt_scale);
        int xstart=x-r;
        int xend=x+r;
        int ystart=y-r;
        int yend=y+r;
        float direction_response[36]={0};
        for(int j=ystart;j<=yend;j++)
        {
            for(int k=xstart;k<=xend;k++)
            {   if(j<1||j>gpyrs[octave][size].rows-2||k<1||k>gpyrs[octave][size].cols-2)
                {
                    continue;
                }
                float dx=gpyrs[octave][size].at<float>(j,k+1)-gpyrs[octave][size].at<float>(j,k-1);
                float dy=gpyrs[octave][size].at<float>(j+1,k)-gpyrs[octave][size].at<float>(j-1,k);
                float magnitude=sqrt(dx*dx+dy*dy);
                float angle=atan2(dy,dx);
                //转换到0到360度
                if(angle<0)
                {
                    angle+=2*CV_PI;
                }
                //弧度制转换到角度制
                angle=angle*180/CV_PI;
                int bin=int(angle/10);
                //以x,y为中心点，高斯加权bin
                float weight=exp(-(pow(k-keypoints_eer[i].pt.x,2)+pow(j-keypoints_eer[i].pt.y,2))/(2*sigma*sigma));
                direction_response[bin]+=magnitude*weight;
            }
        }

        //对方向直方图进行平滑
        float final_direction_response[36]={0};
        for(int i=0;i<36;i++)
        {
            int is2=i-2>-1?i-2:i-2+36;
            int ia2=i+2<36?i+2:i+2-36;
            int is1=i-1>-1?i-1:i-1+36;
            int ia1=i+1<36?i+1:i+1-36;
            final_direction_response[i]=direction_response[is2]+4*direction_response[is1]+6*direction_response[i]+4*direction_response[ia1]+direction_response[ia2];
        }
        //找出最大和第二大的方向
        int max_index=0;
        int second_index=0;
        float max_value=0;
        float second_value=0;
        for(int j=0;j<36;j++)
        {
            if(final_direction_response[j]>max_value)
            {
                max_value=final_direction_response[j];
                max_index=j;
            }
        }
        for(int j=0;j<36;j++)
        {
            if(j==max_index)
            {
                continue;
            }
            if(final_direction_response[j]>second_value)
            {
                second_value=final_direction_response[j];
                second_index=j;
            }
        }
        //keypoints_direction中加入两个方向的关键点
        KeyPoint kp1;
        kp1.pt.x=keypoints_eer[i].pt.x;
        kp1.pt.y=keypoints_eer[i].pt.y;
        kp1.octave=keypoints_eer[i].octave;
        kp1.size=keypoints_eer[i].size;
        kp1.response=keypoints_eer[i].response;
        //插值法求出精确主方向
        float accuracy_direction_main=max_index*10+5;
        int fs1=max_index-1>-1?max_index-1:max_index-1+36;
        int fa1=max_index+1<36?max_index+1:max_index+1-36;
        accuracy_direction_main=accuracy_direction_main+0.5*10*(final_direction_response[fs1]-final_direction_response[fa1])/(final_direction_response[fs1]+final_direction_response[fa1]-2*final_direction_response[max_index]);
        kp1.angle=accuracy_direction_main;
        keypoints_direction.push_back(kp1);
        //如果第二大的方向和最大的方向幅值比例大于0.8，那么加入第二大的方向
        if(second_value/max_value>0.8)
        {
            KeyPoint kp2;
            kp2.pt.x=keypoints_eer[i].pt.x;
            kp2.pt.y=keypoints_eer[i].pt.y;
            kp2.octave=keypoints_eer[i].octave;
            kp2.size=keypoints_eer[i].size;
            kp2.response=keypoints_eer[i].response;
            float accuracy_direction_helper=second_index*10+5;
            int fs2=second_index-1>-1?second_index-1:second_index-1+36;
            int fa2=second_index+1<36?second_index+1:second_index+1-36;
            accuracy_direction_helper=accuracy_direction_helper+0.5*10*(final_direction_response[fs2]-final_direction_response[fa2])/(final_direction_response[fs2]+final_direction_response[fa2]-2*final_direction_response[second_index]);
            

            keypoints_direction.push_back(kp2);
        }


    }
}

void generate_descriptors(vector<vector<Mat>>& gpyrs,vector<KeyPoint>& keypoints_direction,Mat& descriptors)
{
    //初始化descriptors存储128位的描述子
    descriptors=Mat::zeros(keypoints_direction.size(),128,CV_32F);
for(int ki=0;ki<keypoints_direction.size();ki++)
{
    
    int octave=keypoints_direction[ki].octave;
    int size=cvRound(keypoints_direction[ki].size);
    float scale=pow(2,float(size)/3)*1.6;
    Mat& gauss_image=gpyrs[octave][size];
    float main_angle=keypoints_direction[ki].angle;
    Point ptxy(cvRound(keypoints_direction[ki].pt.x), 
    cvRound(keypoints_direction[ki].pt.y));					//坐标取整
    float cos_t = cosf(-main_angle * (float)(CV_PI / 180));		//把角度转化为弧度，计算主方向的余弦
	float sin_t = sinf(-main_angle * (float)(CV_PI / 180));		//把角度转化为弧度，计算主方向的正弦
    float exp_scale = -1.f / (4 * 4 * 0.5f);//权重指数部分
    float hist_width = 3.0f *scale;//特征点邻域内子区域边长，子区域的边长
    int radius = cvRound(hist_width * (4 + 1) * sqrt(2) * 0.5f);//特征点邻域半径(d+1)*(d+1)，四舍五入
    int rows = gauss_image.rows, cols = gauss_image.cols;		//当前高斯层行、列信息
    radius = min(radius, (int)sqrt((double)rows * rows + cols * cols));
    cos_t = cos_t / hist_width;
    sin_t = sin_t / hist_width;
    int len = (2 * radius + 1) * (2 * radius + 1);				//邻域内总像素数，为后面动态分配内存使用
    int histlen = (4 + 2) * (4 + 2) * (8 + 2);					//值为 360 
    AutoBuffer<float> buf(6 * len + histlen);
    //X保存水平差分，Y保存竖直差分，Mag保存梯度幅度，Angle保存特征点方向, W保存高斯权重
	float* X = buf, * Y = buf + len, * Mag = Y, * Angle = Y + len, * W = Angle + len;
	float* RBin = W + len, * CBin = RBin + len, * hist = CBin + len;
    	//首先清空直方图数据
	for (int i = 0; i < 4 + 2; ++i)				// i 对应 row
	{
		for (int j = 0; j < 4 + 2; ++j)			// j 对应 col
		{
			for (int k = 0; k < 8 + 2; ++k)
 
				hist[(i * (4 + 2) + j) * (8 + 2) + k] = 0.f;
		}
	}
	//把邻域内的像素分配到相应子区域内，计算子区域内每个像素点的权重(子区域即 d*d 中每一个小方格)
	int k = 0;
    	//实际上是在 4 x 4 的网格中找 16 个种子点，每个种子点都在子网格的正中心，
	//通过三线性插值对不同种子点间的像素点进行加权作用到不同的种子点上绘制方向直方图
 
	for (int i = -radius; i < radius; ++i)						// i 对应 y 行坐标
	{
		for (int j = -radius; j < radius; ++j)					// j 对应 x 列坐标
		{
			float c_rot = j * cos_t - i * sin_t;				//旋转后邻域内采样点的 x 坐标
			float r_rot = j * sin_t + i * cos_t;				//旋转后邻域内采样点的 y 坐标
 
			//旋转后 5 x 5 的网格中的所有像素点被分配到 4 x 4 的网格中
			float cbin = c_rot + 2- 0.5f;					//旋转后的采样点落在子区域的 x 坐标
			float rbin = r_rot + 2- 0.5f;					//旋转后的采样点落在子区域的 y 坐标
 
			int r = ptxy.y + i, c = ptxy.x + j;					//ptxy是高斯金字塔中的坐标
 
			if (rbin > -1 && rbin < 4 && cbin>-1 && cbin < 4 && r>0 && r < rows - 1 && c>0 && c < cols - 1)
			{
				float dx = gauss_image.at<float>(r, c + 1) - gauss_image.at<float>(r, c - 1);
				float dy = gauss_image.at<float>(r + 1, c) - gauss_image.at<float>(r - 1, c);
 
				X[k] = dx;												//邻域内所有像素点的水平差分
				Y[k] = dy;												//邻域内所有像素点的竖直差分
 
				CBin[k] = cbin;											//邻域内所有采样点落在子区域的 x 坐标
				RBin[k] = rbin;											//邻域内所有采样点落在子区域的 y 坐标
 
				W[k] = (c_rot * c_rot + r_rot * r_rot) * exp_scale;		//高斯权值的指数部分
 
				++k;
			}
		}
	}
	//计算采样点落在子区域的像素梯度幅度，梯度角度，和高斯权值
	len = k;
 
	cv::hal::exp(W, W, len);						//邻域内所有采样点落在子区域的像素的高斯权值
	cv::hal::fastAtan2(Y, X, Angle, len, true);		//邻域内所有采样点落在子区域的像素的梯度方向，角度范围是0-360度
	cv::hal::magnitude(X, Y, Mag, len);				//邻域内所有采样点落在子区域的像素的梯度幅度

	//在 4 x 4 的网格中找 16 个种子点，每个种子点都在子网格的正中心，
	//通过三线性插值对不同种子点间的像素点进行加权作用到不同的种子点上绘制方向直方图
 
	//计算每个特征点的特征描述子
	for (k = 0; k < len; ++k)
	{
		float rbin = RBin[k], cbin = CBin[k];		//子区域内像素点坐标，rbin,cbin范围是(-1,d)
 
		//子区域内像素点处理后的方向
		float temp = Angle[k] - main_angle;
        if (temp < 0.f)
            temp += 360.f;

		float obin = temp * 8/360.0f;			//指定方向的数量后，邻域内像素点对应的方向
 
		float mag = Mag[k] * W[k];					//子区域内像素点乘以权值后的梯度幅值
 
		int r0 = cvFloor(rbin);						
		int c0 = cvFloor(cbin);						
		int o0 = cvFloor(obin);
 
		rbin = rbin - r0;							//子区域内像素点坐标的小数部分，用于线性插值，分配像素点的作用
		cbin = cbin - c0;
		obin = obin - o0;							//子区域方向的小数部分
 
		//限制范围为梯度直方图横坐标[0,n)，8 个方向直方图
		if (o0 < 0)
			o0 = o0 + 8;
		if (o0 >= 8)
			o0 = o0 - 8;
 
		//三线性插值用于计算落在两个子区域之间的像素对两个子区域的作用，并把其分配到对应子区域的8个方向上
		//像素对应的信息通过加权分配给其周围的种子点，并把相应方向的梯度值进行累加  
 
		float v_r1 = mag * rbin;					//第二行分配的值
		float v_r0 = mag - v_r1;					//第一行分配的值
 
		float v_rc11 = v_r1 * cbin;					//第二行第二列分配的值，右下角种子点
		float v_rc10 = v_r1 - v_rc11;				//第二行第一列分配的值，左下角种子点
 
		float v_rc01 = v_r0 * cbin;					//第一行第二列分配的值，右上角种子点
		float v_rc00 = v_r0 - v_rc01;				//第一行第一列分配的值，左上角种子点
 
		//一个像素点的方向为每个种子点的两个方向做出贡献
		float v_rco111 = v_rc11 * obin;				//右下角种子点第二个方向上分配的值
		float v_rco110 = v_rc11 - v_rco111;			//右下角种子点第一个方向上分配的值
 
		float v_rco101 = v_rc10 * obin;
		float v_rco100 = v_rc10 - v_rco101;
 
		float v_rco011 = v_rc01 * obin;
		float v_rco010 = v_rc01 - v_rco011;
 
		float v_rco001 = v_rc00 * obin;
		float v_rco000 = v_rc00 - v_rco001;
 
		//该像素所在网格的索引
		int idx = ((r0 + 1) * (4 + 2) + c0 + 1) * (4 + 2) + o0;
 
		hist[idx] += v_rco000;
		hist[idx + 1] += v_rco001;
		hist[idx + 8 + 2] += v_rco010;
		hist[idx + 8 + 3] += v_rco011;
		hist[idx + (4 + 2) * (8 + 2)] += v_rco100;
		hist[idx + (4 + 2) * (8 + 2) + 1] += v_rco101;
		hist[idx + (4 + 3) * (8 + 2)] += v_rco110;
		hist[idx + (4 + 3) * (8 + 2) + 1] += v_rco111;
	}
    	//由于圆周循环的特性，对计算以后幅角小于 0 度或大于 360 度的值重新进行调整，使
	//其在 0～360 度之间
	for (int i = 0; i < 4; ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			//类似于 hist[0][2][3] 第 0 行，第 2 列，种子点直方图中的第 3 个 bin
			int idx = ((i + 1) * (4 + 2) + (j + 1)) * (8 + 2);
			hist[idx] += hist[idx + 8];
			for (k = 0; k < 8; ++k)
				descriptors.at<float>(ki,(i * 4 + j) * 8 + k) = hist[idx + k];
		}
	}
 	//对描述子进行归一化
	int lenght = 4 * 4 * 8;
	float norm = 0;
 
	//计算特征描述向量的模值的平方
	for (int i = 0; i < lenght; ++i)
	{
		norm = norm + descriptors.at<float>(ki,i) * descriptors.at<float>(ki,i);
	}
	norm = sqrt(norm);							//特征描述向量的模值
 
	//此次归一化能去除光照的影响
	for (int i = 0; i < lenght; ++i)
	{
		descriptors.at<float>(ki,i) = descriptors.at<float>(ki,i) / norm;
	}
 
	//阈值截断,去除特征描述向量中大于 0.2 的值，能消除非线性光照的影响(相机饱和度对某些放的梯度影响较大，对方向的影响较小)
	for (int i = 0; i < lenght; ++i)
	{   
		descriptors.at<float>(ki,i) = min(descriptors.at<float>(ki,i),0.2f);
	}
 
	//再次归一化，能够提高特征的独特性
	norm = 0;
	for (int i = 0; i < lenght; ++i)
	{
		norm = norm + descriptors.at<float>(ki,i) * descriptors.at<float>(ki,i);
	}
	norm = sqrt(norm);
	for (int i = 0; i < lenght; ++i)
	{
		 descriptors.at<float>(ki,i) =  descriptors.at<float>(ki,i) / norm;
	}


}

}

void trans_to_original_size(vector<KeyPoint>& keypoints,vector<KeyPoint>& keypoints_original)
{
    for(int i=0;i<keypoints.size();i++)
    {
        float ratio=pow(2,keypoints[i].octave-1);
        KeyPoint kp;
        kp.pt.x=keypoints[i].pt.x*ratio;
        kp.pt.y=keypoints[i].pt.y*ratio;
        kp.octave=keypoints[i].octave;
        kp.size=1.5*3*1.6*pow(2,keypoints[i].size/3.0f);
        kp.response=keypoints[i].response;
        kp.angle=keypoints[i].angle;
        keypoints_original.push_back(kp);

    }
}

void my_sift(Mat&src,vector<KeyPoint>&kp,Mat& descriptors,int k=2,double sigma=1.6,int n=3,int r=10)
{
    //对传入图像处理成灰度图
    Mat gray;
    cvtColor(src,gray,COLOR_BGR2GRAY);
    //生成高斯金字塔
    vector<vector<Mat>> gpyrs;
    gauss_gpyr(gray,gpyrs,sigma,n);
    //生成高斯差分金字塔
    vector<vector<Mat>> dog;
    generate_dog(gpyrs,dog);
    //初步生成像素级特征点
    vector<KeyPoint> keypoints;
    generate_keypoints_roughly(dog,keypoints);
    //生成位置更精确的特征点
    vector<KeyPoint> keypoints_accurate;
    generate_keypoints_accurately(dog,keypoints,keypoints_accurate,n);
    //消除边缘响应
    vector<KeyPoint> keypoints_eer;
    eliminate_edge_response(dog,keypoints_accurate,keypoints_eer,r);
    //为稳定关键点赋予方向信息
    vector<KeyPoint> keypoints_direction;
    generate_keypoints_direction(gpyrs,keypoints_eer,keypoints_direction);
    //生成描述子
    generate_descriptors(gpyrs,keypoints_direction,descriptors);
    //转变到原本尺寸
    trans_to_original_size(keypoints_direction,kp);


}

int main()
{   string path="images/";
    //读取图像
    string temp;
    cout<<"input image1 name"<<endl;
    cin>>temp;
    Mat img1=imread(path+temp);
    vector<KeyPoint> kp1;
    Mat descriptors1;
    my_sift(img1,kp1,descriptors1);
    cout<<"keypoints size:"<<kp1.size()<<endl;

    //画出特征点
    Mat img_keypoints;
    drawKeypoints(img1,kp1,img_keypoints,cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("keypoints",img_keypoints);
    waitKey(0);
    cout<<"input image2 name"<<endl;
    string temp2;
    cin>>temp2;
    Mat img2=imread(path+temp2);
    vector<KeyPoint> kp2;
    Mat descriptors2;
    my_sift(img2,kp2,descriptors2);
    cout<<"keypoints size:"<<kp2.size()<<endl;
    //画出特征点
    Mat img_keypoints2;
    drawKeypoints(img2,kp2,img_keypoints2,cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("keypoints2",img_keypoints2);
    waitKey(0);
    //调用BFMatcher进行匹配
    BFMatcher matcher(NORM_L2,true);
    vector<DMatch> matches;
    matcher.match(descriptors1,descriptors2,matches);
    //取前30个匹配结果
    //sort(matches.begin(),matches.end());
    //matches.erase(matches.begin()+30,matches.end());

    //绘制匹配结果
    Mat img_matches;
    drawMatches(img1,kp1,img2,kp2,matches,img_matches);
    imshow("matches",img_matches);
    waitKey(0);

    //调用opencv的sift生成特征点并展示匹配结果
    Ptr<SIFT> sift=SIFT::create();
    vector<KeyPoint> kp3,kp4;
    Mat descriptors3,descriptors4;
    sift->detectAndCompute(img1,Mat(),kp3,descriptors3);
    sift->detectAndCompute(img2,Mat(),kp4,descriptors4);
    cout<<"keypoints size:"<<kp3.size()<<endl;
    cout<<"keypoints size:"<<kp4.size()<<endl;
    BFMatcher matcher2(NORM_L2,true);
    vector<DMatch> matches2;
    matcher2.match(descriptors3,descriptors4,matches2);
    //sort(matches2.begin(),matches2.end());
    //matches2.erase(matches2.begin()+30,matches2.end());
    Mat img_matches2;
    drawMatches(img1,kp3,img2,kp4,matches2,img_matches2);
    imshow("matches2",img_matches2);
    waitKey(0);
    return 0;
}