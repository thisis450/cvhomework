#include <opencv2/opencv.hpp>
#include <math.h>
using namespace cv;
using namespace std;
const int Y[16]={-3,-3,-2,-1,0,1,2,3,3,3,2,1,0,-1,-2,-3};
const int X[16]={0,1,2,3,3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1};
const int sobel_x[9]={-1,0,1,-2,0,2,-1,0,1};
const int sobel_y[9]={-1,-2,-1,0,0,0,1,2,1};

static int bit_pattern_31_[256*4] =
{
    8,-3, 9,5/*mean (0), correlation (0)*/,
    4,2, 7,-12/*mean (1.12461e-05), correlation (0.0437584)*/,
    -11,9, -8,2/*mean (3.37382e-05), correlation (0.0617409)*/,
    7,-12, 12,-13/*mean (5.62303e-05), correlation (0.0636977)*/,
    2,-13, 2,12/*mean (0.000134953), correlation (0.085099)*/,
    1,-7, 1,6/*mean (0.000528565), correlation (0.0857175)*/,
    -2,-10, -2,-4/*mean (0.0188821), correlation (0.0985774)*/,
    -13,-13, -11,-8/*mean (0.0363135), correlation (0.0899616)*/,
    -13,-3, -12,-9/*mean (0.121806), correlation (0.099849)*/,
    10,4, 11,9/*mean (0.122065), correlation (0.093285)*/,
    -13,-8, -8,-9/*mean (0.162787), correlation (0.0942748)*/,
    -11,7, -9,12/*mean (0.21561), correlation (0.0974438)*/,
    7,7, 12,6/*mean (0.160583), correlation (0.130064)*/,
    -4,-5, -3,0/*mean (0.228171), correlation (0.132998)*/,
    -13,2, -12,-3/*mean (0.00997526), correlation (0.145926)*/,
    -9,0, -7,5/*mean (0.198234), correlation (0.143636)*/,
    12,-6, 12,-1/*mean (0.0676226), correlation (0.16689)*/,
    -3,6, -2,12/*mean (0.166847), correlation (0.171682)*/,
    -6,-13, -4,-8/*mean (0.101215), correlation (0.179716)*/,
    11,-13, 12,-8/*mean (0.200641), correlation (0.192279)*/,
    4,7, 5,1/*mean (0.205106), correlation (0.186848)*/,
    5,-3, 10,-3/*mean (0.234908), correlation (0.192319)*/,
    3,-7, 6,12/*mean (0.0709964), correlation (0.210872)*/,
    -8,-7, -6,-2/*mean (0.0939834), correlation (0.212589)*/,
    -2,11, -1,-10/*mean (0.127778), correlation (0.20866)*/,
    -13,12, -8,10/*mean (0.14783), correlation (0.206356)*/,
    -7,3, -5,-3/*mean (0.182141), correlation (0.198942)*/,
    -4,2, -3,7/*mean (0.188237), correlation (0.21384)*/,
    -10,-12, -6,11/*mean (0.14865), correlation (0.23571)*/,
    5,-12, 6,-7/*mean (0.222312), correlation (0.23324)*/,
    5,-6, 7,-1/*mean (0.229082), correlation (0.23389)*/,
    1,0, 4,-5/*mean (0.241577), correlation (0.215286)*/,
    9,11, 11,-13/*mean (0.00338507), correlation (0.251373)*/,
    4,7, 4,12/*mean (0.131005), correlation (0.257622)*/,
    2,-1, 4,4/*mean (0.152755), correlation (0.255205)*/,
    -4,-12, -2,7/*mean (0.182771), correlation (0.244867)*/,
    -8,-5, -7,-10/*mean (0.186898), correlation (0.23901)*/,
    4,11, 9,12/*mean (0.226226), correlation (0.258255)*/,
    0,-8, 1,-13/*mean (0.0897886), correlation (0.274827)*/,
    -13,-2, -8,2/*mean (0.148774), correlation (0.28065)*/,
    -3,-2, -2,3/*mean (0.153048), correlation (0.283063)*/,
    -6,9, -4,-9/*mean (0.169523), correlation (0.278248)*/,
    8,12, 10,7/*mean (0.225337), correlation (0.282851)*/,
    0,9, 1,3/*mean (0.226687), correlation (0.278734)*/,
    7,-5, 11,-10/*mean (0.00693882), correlation (0.305161)*/,
    -13,-6, -11,0/*mean (0.0227283), correlation (0.300181)*/,
    10,7, 12,1/*mean (0.125517), correlation (0.31089)*/,
    -6,-3, -6,12/*mean (0.131748), correlation (0.312779)*/,
    10,-9, 12,-4/*mean (0.144827), correlation (0.292797)*/,
    -13,8, -8,-12/*mean (0.149202), correlation (0.308918)*/,
    -13,0, -8,-4/*mean (0.160909), correlation (0.310013)*/,
    3,3, 7,8/*mean (0.177755), correlation (0.309394)*/,
    5,7, 10,-7/*mean (0.212337), correlation (0.310315)*/,
    -1,7, 1,-12/*mean (0.214429), correlation (0.311933)*/,
    3,-10, 5,6/*mean (0.235807), correlation (0.313104)*/,
    2,-4, 3,-10/*mean (0.00494827), correlation (0.344948)*/,
    -13,0, -13,5/*mean (0.0549145), correlation (0.344675)*/,
    -13,-7, -12,12/*mean (0.103385), correlation (0.342715)*/,
    -13,3, -11,8/*mean (0.134222), correlation (0.322922)*/,
    -7,12, -4,7/*mean (0.153284), correlation (0.337061)*/,
    6,-10, 12,8/*mean (0.154881), correlation (0.329257)*/,
    -9,-1, -7,-6/*mean (0.200967), correlation (0.33312)*/,
    -2,-5, 0,12/*mean (0.201518), correlation (0.340635)*/,
    -12,5, -7,5/*mean (0.207805), correlation (0.335631)*/,
    3,-10, 8,-13/*mean (0.224438), correlation (0.34504)*/,
    -7,-7, -4,5/*mean (0.239361), correlation (0.338053)*/,
    -3,-2, -1,-7/*mean (0.240744), correlation (0.344322)*/,
    2,9, 5,-11/*mean (0.242949), correlation (0.34145)*/,
    -11,-13, -5,-13/*mean (0.244028), correlation (0.336861)*/,
    -1,6, 0,-1/*mean (0.247571), correlation (0.343684)*/,
    5,-3, 5,2/*mean (0.000697256), correlation (0.357265)*/,
    -4,-13, -4,12/*mean (0.00213675), correlation (0.373827)*/,
    -9,-6, -9,6/*mean (0.0126856), correlation (0.373938)*/,
    -12,-10, -8,-4/*mean (0.0152497), correlation (0.364237)*/,
    10,2, 12,-3/*mean (0.0299933), correlation (0.345292)*/,
    7,12, 12,12/*mean (0.0307242), correlation (0.366299)*/,
    -7,-13, -6,5/*mean (0.0534975), correlation (0.368357)*/,
    -4,9, -3,4/*mean (0.099865), correlation (0.372276)*/,
    7,-1, 12,2/*mean (0.117083), correlation (0.364529)*/,
    -7,6, -5,1/*mean (0.126125), correlation (0.369606)*/,
    -13,11, -12,5/*mean (0.130364), correlation (0.358502)*/,
    -3,7, -2,-6/*mean (0.131691), correlation (0.375531)*/,
    7,-8, 12,-7/*mean (0.160166), correlation (0.379508)*/,
    -13,-7, -11,-12/*mean (0.167848), correlation (0.353343)*/,
    1,-3, 12,12/*mean (0.183378), correlation (0.371916)*/,
    2,-6, 3,0/*mean (0.228711), correlation (0.371761)*/,
    -4,3, -2,-13/*mean (0.247211), correlation (0.364063)*/,
    -1,-13, 1,9/*mean (0.249325), correlation (0.378139)*/,
    7,1, 8,-6/*mean (0.000652272), correlation (0.411682)*/,
    1,-1, 3,12/*mean (0.00248538), correlation (0.392988)*/,
    9,1, 12,6/*mean (0.0206815), correlation (0.386106)*/,
    -1,-9, -1,3/*mean (0.0364485), correlation (0.410752)*/,
    -13,-13, -10,5/*mean (0.0376068), correlation (0.398374)*/,
    7,7, 10,12/*mean (0.0424202), correlation (0.405663)*/,
    12,-5, 12,9/*mean (0.0942645), correlation (0.410422)*/,
    6,3, 7,11/*mean (0.1074), correlation (0.413224)*/,
    5,-13, 6,10/*mean (0.109256), correlation (0.408646)*/,
    2,-12, 2,3/*mean (0.131691), correlation (0.416076)*/,
    3,8, 4,-6/*mean (0.165081), correlation (0.417569)*/,
    2,6, 12,-13/*mean (0.171874), correlation (0.408471)*/,
    9,-12, 10,3/*mean (0.175146), correlation (0.41296)*/,
    -8,4, -7,9/*mean (0.183682), correlation (0.402956)*/,
    -11,12, -4,-6/*mean (0.184672), correlation (0.416125)*/,
    1,12, 2,-8/*mean (0.191487), correlation (0.386696)*/,
    6,-9, 7,-4/*mean (0.192668), correlation (0.394771)*/,
    2,3, 3,-2/*mean (0.200157), correlation (0.408303)*/,
    6,3, 11,0/*mean (0.204588), correlation (0.411762)*/,
    3,-3, 8,-8/*mean (0.205904), correlation (0.416294)*/,
    7,8, 9,3/*mean (0.213237), correlation (0.409306)*/,
    -11,-5, -6,-4/*mean (0.243444), correlation (0.395069)*/,
    -10,11, -5,10/*mean (0.247672), correlation (0.413392)*/,
    -5,-8, -3,12/*mean (0.24774), correlation (0.411416)*/,
    -10,5, -9,0/*mean (0.00213675), correlation (0.454003)*/,
    8,-1, 12,-6/*mean (0.0293635), correlation (0.455368)*/,
    4,-6, 6,-11/*mean (0.0404971), correlation (0.457393)*/,
    -10,12, -8,7/*mean (0.0481107), correlation (0.448364)*/,
    4,-2, 6,7/*mean (0.050641), correlation (0.455019)*/,
    -2,0, -2,12/*mean (0.0525978), correlation (0.44338)*/,
    -5,-8, -5,2/*mean (0.0629667), correlation (0.457096)*/,
    7,-6, 10,12/*mean (0.0653846), correlation (0.445623)*/,
    -9,-13, -8,-8/*mean (0.0858749), correlation (0.449789)*/,
    -5,-13, -5,-2/*mean (0.122402), correlation (0.450201)*/,
    8,-8, 9,-13/*mean (0.125416), correlation (0.453224)*/,
    -9,-11, -9,0/*mean (0.130128), correlation (0.458724)*/,
    1,-8, 1,-2/*mean (0.132467), correlation (0.440133)*/,
    7,-4, 9,1/*mean (0.132692), correlation (0.454)*/,
    -2,1, -1,-4/*mean (0.135695), correlation (0.455739)*/,
    11,-6, 12,-11/*mean (0.142904), correlation (0.446114)*/,
    -12,-9, -6,4/*mean (0.146165), correlation (0.451473)*/,
    3,7, 7,12/*mean (0.147627), correlation (0.456643)*/,
    5,5, 10,8/*mean (0.152901), correlation (0.455036)*/,
    0,-4, 2,8/*mean (0.167083), correlation (0.459315)*/,
    -9,12, -5,-13/*mean (0.173234), correlation (0.454706)*/,
    0,7, 2,12/*mean (0.18312), correlation (0.433855)*/,
    -1,2, 1,7/*mean (0.185504), correlation (0.443838)*/,
    5,11, 7,-9/*mean (0.185706), correlation (0.451123)*/,
    3,5, 6,-8/*mean (0.188968), correlation (0.455808)*/,
    -13,-4, -8,9/*mean (0.191667), correlation (0.459128)*/,
    -5,9, -3,-3/*mean (0.193196), correlation (0.458364)*/,
    -4,-7, -3,-12/*mean (0.196536), correlation (0.455782)*/,
    6,5, 8,0/*mean (0.1972), correlation (0.450481)*/,
    -7,6, -6,12/*mean (0.199438), correlation (0.458156)*/,
    -13,6, -5,-2/*mean (0.211224), correlation (0.449548)*/,
    1,-10, 3,10/*mean (0.211718), correlation (0.440606)*/,
    4,1, 8,-4/*mean (0.213034), correlation (0.443177)*/,
    -2,-2, 2,-13/*mean (0.234334), correlation (0.455304)*/,
    2,-12, 12,12/*mean (0.235684), correlation (0.443436)*/,
    -2,-13, 0,-6/*mean (0.237674), correlation (0.452525)*/,
    4,1, 9,3/*mean (0.23962), correlation (0.444824)*/,
    -6,-10, -3,-5/*mean (0.248459), correlation (0.439621)*/,
    -3,-13, -1,1/*mean (0.249505), correlation (0.456666)*/,
    7,5, 12,-11/*mean (0.00119208), correlation (0.495466)*/,
    4,-2, 5,-7/*mean (0.00372245), correlation (0.484214)*/,
    -13,9, -9,-5/*mean (0.00741116), correlation (0.499854)*/,
    7,1, 8,6/*mean (0.0208952), correlation (0.499773)*/,
    7,-8, 7,6/*mean (0.0220085), correlation (0.501609)*/,
    -7,-4, -7,1/*mean (0.0233806), correlation (0.496568)*/,
    -8,11, -7,-8/*mean (0.0236505), correlation (0.489719)*/,
    -13,6, -12,-8/*mean (0.0268781), correlation (0.503487)*/,
    2,4, 3,9/*mean (0.0323324), correlation (0.501938)*/,
    10,-5, 12,3/*mean (0.0399235), correlation (0.494029)*/,
    -6,-5, -6,7/*mean (0.0420153), correlation (0.486579)*/,
    8,-3, 9,-8/*mean (0.0548021), correlation (0.484237)*/,
    2,-12, 2,8/*mean (0.0616622), correlation (0.496642)*/,
    -11,-2, -10,3/*mean (0.0627755), correlation (0.498563)*/,
    -12,-13, -7,-9/*mean (0.0829622), correlation (0.495491)*/,
    -11,0, -10,-5/*mean (0.0843342), correlation (0.487146)*/,
    5,-3, 11,8/*mean (0.0929937), correlation (0.502315)*/,
    -2,-13, -1,12/*mean (0.113327), correlation (0.48941)*/,
    -1,-8, 0,9/*mean (0.132119), correlation (0.467268)*/,
    -13,-11, -12,-5/*mean (0.136269), correlation (0.498771)*/,
    -10,-2, -10,11/*mean (0.142173), correlation (0.498714)*/,
    -3,9, -2,-13/*mean (0.144141), correlation (0.491973)*/,
    2,-3, 3,2/*mean (0.14892), correlation (0.500782)*/,
    -9,-13, -4,0/*mean (0.150371), correlation (0.498211)*/,
    -4,6, -3,-10/*mean (0.152159), correlation (0.495547)*/,
    -4,12, -2,-7/*mean (0.156152), correlation (0.496925)*/,
    -6,-11, -4,9/*mean (0.15749), correlation (0.499222)*/,
    6,-3, 6,11/*mean (0.159211), correlation (0.503821)*/,
    -13,11, -5,5/*mean (0.162427), correlation (0.501907)*/,
    11,11, 12,6/*mean (0.16652), correlation (0.497632)*/,
    7,-5, 12,-2/*mean (0.169141), correlation (0.484474)*/,
    -1,12, 0,7/*mean (0.169456), correlation (0.495339)*/,
    -4,-8, -3,-2/*mean (0.171457), correlation (0.487251)*/,
    -7,1, -6,7/*mean (0.175), correlation (0.500024)*/,
    -13,-12, -8,-13/*mean (0.175866), correlation (0.497523)*/,
    -7,-2, -6,-8/*mean (0.178273), correlation (0.501854)*/,
    -8,5, -6,-9/*mean (0.181107), correlation (0.494888)*/,
    -5,-1, -4,5/*mean (0.190227), correlation (0.482557)*/,
    -13,7, -8,10/*mean (0.196739), correlation (0.496503)*/,
    1,5, 5,-13/*mean (0.19973), correlation (0.499759)*/,
    1,0, 10,-13/*mean (0.204465), correlation (0.49873)*/,
    9,12, 10,-1/*mean (0.209334), correlation (0.49063)*/,
    5,-8, 10,-9/*mean (0.211134), correlation (0.503011)*/,
    -1,11, 1,-13/*mean (0.212), correlation (0.499414)*/,
    -9,-3, -6,2/*mean (0.212168), correlation (0.480739)*/,
    -1,-10, 1,12/*mean (0.212731), correlation (0.502523)*/,
    -13,1, -8,-10/*mean (0.21327), correlation (0.489786)*/,
    8,-11, 10,-6/*mean (0.214159), correlation (0.488246)*/,
    2,-13, 3,-6/*mean (0.216993), correlation (0.50287)*/,
    7,-13, 12,-9/*mean (0.223639), correlation (0.470502)*/,
    -10,-10, -5,-7/*mean (0.224089), correlation (0.500852)*/,
    -10,-8, -8,-13/*mean (0.228666), correlation (0.502629)*/,
    4,-6, 8,5/*mean (0.22906), correlation (0.498305)*/,
    3,12, 8,-13/*mean (0.233378), correlation (0.503825)*/,
    -4,2, -3,-3/*mean (0.234323), correlation (0.476692)*/,
    5,-13, 10,-12/*mean (0.236392), correlation (0.475462)*/,
    4,-13, 5,-1/*mean (0.236842), correlation (0.504132)*/,
    -9,9, -4,3/*mean (0.236977), correlation (0.497739)*/,
    0,3, 3,-9/*mean (0.24314), correlation (0.499398)*/,
    -12,1, -6,1/*mean (0.243297), correlation (0.489447)*/,
    3,2, 4,-8/*mean (0.00155196), correlation (0.553496)*/,
    -10,-10, -10,9/*mean (0.00239541), correlation (0.54297)*/,
    8,-13, 12,12/*mean (0.0034413), correlation (0.544361)*/,
    -8,-12, -6,-5/*mean (0.003565), correlation (0.551225)*/,
    2,2, 3,7/*mean (0.00835583), correlation (0.55285)*/,
    10,6, 11,-8/*mean (0.00885065), correlation (0.540913)*/,
    6,8, 8,-12/*mean (0.0101552), correlation (0.551085)*/,
    -7,10, -6,5/*mean (0.0102227), correlation (0.533635)*/,
    -3,-9, -3,9/*mean (0.0110211), correlation (0.543121)*/,
    -1,-13, -1,5/*mean (0.0113473), correlation (0.550173)*/,
    -3,-7, -3,4/*mean (0.0140913), correlation (0.554774)*/,
    -8,-2, -8,3/*mean (0.017049), correlation (0.55461)*/,
    4,2, 12,12/*mean (0.01778), correlation (0.546921)*/,
    2,-5, 3,11/*mean (0.0224022), correlation (0.549667)*/,
    6,-9, 11,-13/*mean (0.029161), correlation (0.546295)*/,
    3,-1, 7,12/*mean (0.0303081), correlation (0.548599)*/,
    11,-1, 12,4/*mean (0.0355151), correlation (0.523943)*/,
    -3,0, -3,6/*mean (0.0417904), correlation (0.543395)*/,
    4,-11, 4,12/*mean (0.0487292), correlation (0.542818)*/,
    2,-4, 2,1/*mean (0.0575124), correlation (0.554888)*/,
    -10,-6, -8,1/*mean (0.0594242), correlation (0.544026)*/,
    -13,7, -11,1/*mean (0.0597391), correlation (0.550524)*/,
    -13,12, -11,-13/*mean (0.0608974), correlation (0.55383)*/,
    6,0, 11,-13/*mean (0.065126), correlation (0.552006)*/,
    0,-1, 1,4/*mean (0.074224), correlation (0.546372)*/,
    -13,3, -9,-2/*mean (0.0808592), correlation (0.554875)*/,
    -9,8, -6,-3/*mean (0.0883378), correlation (0.551178)*/,
    -13,-6, -8,-2/*mean (0.0901035), correlation (0.548446)*/,
    5,-9, 8,10/*mean (0.0949843), correlation (0.554694)*/,
    2,7, 3,-9/*mean (0.0994152), correlation (0.550979)*/,
    -1,-6, -1,-1/*mean (0.10045), correlation (0.552714)*/,
    9,5, 11,-2/*mean (0.100686), correlation (0.552594)*/,
    11,-3, 12,-8/*mean (0.101091), correlation (0.532394)*/,
    3,0, 3,5/*mean (0.101147), correlation (0.525576)*/,
    -1,4, 0,10/*mean (0.105263), correlation (0.531498)*/,
    3,-6, 4,5/*mean (0.110785), correlation (0.540491)*/,
    -13,0, -10,5/*mean (0.112798), correlation (0.536582)*/,
    5,8, 12,11/*mean (0.114181), correlation (0.555793)*/,
    8,9, 9,-6/*mean (0.117431), correlation (0.553763)*/,
    7,-4, 8,-12/*mean (0.118522), correlation (0.553452)*/,
    -10,4, -10,9/*mean (0.12094), correlation (0.554785)*/,
    7,3, 12,4/*mean (0.122582), correlation (0.555825)*/,
    9,-7, 10,-2/*mean (0.124978), correlation (0.549846)*/,
    7,0, 12,-2/*mean (0.127002), correlation (0.537452)*/,
    -1,-6, 0,-11/*mean (0.127148), correlation (0.547401)*/
};


void rotateVector(double x, double y, double theta, double &xr, double &yr) {
  xr = x * cos(theta) - y * sin(theta);
  yr = x * sin(theta) + y * cos(theta);
}

float my_Harris(Mat&src,int x,int y)
{
    float Ix=0,Iy=0;
    float Ixx=0,Iyy=0,Ixy=0;
    float det=0,trace=0;
    float k=0.04;
    for(int i=-1;i<2;i++)
    {
        for(int j=-1;j<2;j++)
        {
            Ix+=src.at<uchar>(y+i,x+j)*sobel_x[(i+1)*3+j+1];
            Iy+=src.at<uchar>(y+i,x+j)*sobel_y[(i+1)*3+j+1];
        }
    }
    Ixx=Ix*Ix;
    Iyy=Iy*Iy;
    Ixy=Ix*Iy;
    det=Ixx*Iyy-Ixy*Ixy;
    trace=Ixx+Iyy;
    return det-k*trace*trace;
}
void my_FAST(Mat&src,vector<KeyPoint>&keypoints,int octave)
{   
    Mat nonMaximumSuppression=Mat::zeros(src.rows,src.cols,CV_16S);
    
    for(int i=3;i<src.rows-3;i++)
    {
        for(int j=3;j<src.rows-3;j++)
        {
            //计算以i,j为中心的3*3区域的方差
            // double mean=0;
            // double variance=0;
            // for(int i2=i-1;i2<i+2;i2++)
            // {
            //     for(int j2=j-1;j2<j+2;j2++)
            //     {
            //         mean+=src.at<uchar>(i2,j2);
            //     }
            // }
            // mean/=9;
            // for(int i2=i-1;i2<i+2;i2++)
            // {
            //     for(int j2=j-1;j2<j+2;j2++)
            //     {
            //         variance+=abs((src.at<uchar>(i2,j2)-mean));
            //     }
            // }
            // variance/=9;
            // double threshold=variance*1.2;
            //double threshold=src.at<uchar>(i,j)*0.15;
            int threshold=20;
            int count=0;
            int score=0;
            int temp=0;
            for(int k=0;k<16;k++)
            {
                temp=abs(src.at<uchar>(i+Y[k],j+X[k])-src.at<uchar>(i,j));
                if(
                temp
                >threshold)
                {
                    count++;
                    score+=temp;

                }
            }
            if(count>=9)
            {
                nonMaximumSuppression.at<short>(i,j)=score;
            }
        }
    }

    for(int i=3;i<src.rows-3;i++)
    {
        for(int j=3;j<src.rows-3;j++)
        {   
            if(nonMaximumSuppression.at<short>(i,j)>0)
            {   
                bool flag=true;
                for(int i2=i-2;flag&&i2<i+3;i2++)
                {
                    for(int j2=j-2;j2<j+3;j2++)
                    {
                        if(nonMaximumSuppression.at<short>(i2,j2)>nonMaximumSuppression.at<short>(i,j))
                        {
                            flag=false;
                            break;
                        }
                    }
                }
                if(flag==true){
                KeyPoint keypoint;
                keypoint.pt.x=j;
                keypoint.pt.y=i;
                keypoint.octave=octave;
                keypoint.size=7;
                keypoints.push_back(keypoint);
                }
            }
        }
    }


}
void calcu_angle(vector<Mat>&pyr,vector<KeyPoint>&keypoints,int nfeatures)
{   int num=min(nfeatures,(int)keypoints.size());
    for(int i=0;i<num;i++)
    {
        int x=keypoints[i].pt.x;
        int y=keypoints[i].pt.y;
        int octave=keypoints[i].octave;
        int scale=keypoints[i].size;
        int rows=pyr[octave].rows;
        int cols=pyr[octave].cols;
        if(x>=5&&x<cols-5&&y>=5&&y<rows-5)
        {
            int m01=0,m10=0;
            for(int i2=-5;i2<6;i2++)
            {
                for(int j2=-5;j2<6;j2++)
                {
                    m01+=i2*pyr[octave].at<uchar>(y+i2,x+j2);
                    m10+=j2*pyr[octave].at<uchar>(y+i2,x+j2);
                }
            }
            keypoints[i].angle=atan2(m01,m10);
        }
    }
}
void calcu_descriptors(vector<Mat>&pyr,vector<KeyPoint>&keypoints,Mat&descriptors,int nfeatures)
{   int num=min(nfeatures,(int)keypoints.size());
    descriptors=Mat::zeros(num,32,CV_8U);
    for(int i=0;i<num;i++)
    {
        for(int ii=0;ii<256;ii++)
        {
            double t1,t2,t3,t4;
            rotateVector(bit_pattern_31_[ii*4],bit_pattern_31_[ii*4+1],keypoints[i].angle,t1,t2);
            rotateVector(bit_pattern_31_[ii*4+2],bit_pattern_31_[ii*4+3],keypoints[i].angle,t3,t4);
            int x1=keypoints[i].pt.x+t1;
            int y1=keypoints[i].pt.y+t2;
            int x2=keypoints[i].pt.x+t3;
            int y2=keypoints[i].pt.y+t4;
            if(x1>=0&&x1<pyr[keypoints[i].octave].cols&&y1>=0&&y1<pyr[keypoints[i].octave].rows&&x2>=0&&x2<pyr[keypoints[i].octave].cols&&y2>=0&&y2<pyr[keypoints[i].octave].rows)
            {
                if(pyr[keypoints[i].octave].at<uchar>(y1,x1)>pyr[keypoints[i].octave].at<uchar>(y2,x2))
                {
                    descriptors.at<uchar>(i,ii/8)+=1<<(ii%8);
                }
            }
        }
    }
}

void my_ORB(Mat&src,vector<KeyPoint>&keypoints,Mat& descriptors,int nfeatures=200,int nlevel=8,float scaleFactor=1.2)
{   
//灰度化
Mat gray;
cvtColor(src,gray,COLOR_BGR2GRAY);
//生成图像金字塔
vector<Mat> pyr;
for(int i=0;i<nlevel;i++)
{
    pyr.push_back(gray.clone());
}
for(int i=1;i<nlevel;i++)
{
    int rows=pyr[i-1].rows/scaleFactor;
    int cols=pyr[i-1].cols/scaleFactor;
resize(pyr[i-1],pyr[i],Size(cols,rows));
}
//计算金字塔中每一层的FAST角点

    vector<KeyPoint> keypoints1;
    for(int i=0;i<nlevel;i++)
    {
        my_FAST(pyr[i],keypoints1,i);
    }
    for (int i=0;i<keypoints1.size();i++)
    {
        float score=my_Harris(src,keypoints1[i].pt.x,keypoints1[i].pt.y);
        if(score>0)
        {
            keypoints1[i].response=score;
        }
    }
    sort(keypoints1.begin(),keypoints1.end(),[](KeyPoint a,KeyPoint b){return a.response>b.response;});
    //计算角点的方向
    calcu_angle(pyr,keypoints1,nfeatures);
    //计算描述子
    calcu_descriptors(pyr,keypoints1,descriptors,nfeatures);
    for(int i=0;i<nfeatures&&i<keypoints1.size();i++)
    {
        KeyPoint keypoint;
        keypoint.pt.x=keypoints1[i].pt.x;
        keypoint.pt.y=keypoints1[i].pt.y;
        keypoint.octave=keypoints1[i].octave;
        keypoint.angle=keypoints1[i].angle;
        keypoints.push_back(keypoint);
    }



    

}

void my_BF(Mat&Descriptor1,Mat&Descriptor2,vector<DMatch>&matches)
{
int num=min(Descriptor1.rows,Descriptor2.rows);
matches.clear();
for(int i=0;i<num;i++)
{
    int min=100000;
    int min2=1000001;
    int flag=0;
    for(int j=0;j<Descriptor2.rows;j++)
    {
        //计算汉明距离
        int distance=0;
        for(int k=0;k<Descriptor1.cols;k++)
        {
            for(int offset=0;offset<8;offset++)
            {
                uchar temp1=Descriptor1.at<uchar>(i,k)&(1<<offset);
                uchar temp2=Descriptor2.at<uchar>(j,k)&(1<<offset);
                if(temp1!=temp2)
                {
                    distance++;
                }
            }
        }
        if(distance<min)
        {
            min=distance;
            flag=j;
        }
        else if(distance<min2)
        {
            min2=distance;
        }
    }
    if(min<0.7*min2)
    {
    DMatch match(i,flag,min);
    matches.push_back(match);
    }
}
}
int main()
{
Mat img1=imread("images/dog11.jpg");
vector<KeyPoint> keypoints1;
Mat descriptors1;
my_ORB(img1,keypoints1,descriptors1);
//绘制角点
Mat img1_keypoints;
vector<KeyPoint> fordraw1;
for(int i=0;i<keypoints1.size();i++)
{
    int x=pow(1.2,keypoints1[i].octave)*keypoints1[i].pt.x;
    int y=pow(1.2,keypoints1[i].octave)*keypoints1[i].pt.y;
    KeyPoint keypoint;
    keypoint.pt.x=x;
    keypoint.pt.y=y;
    fordraw1.push_back(keypoint);
}
drawKeypoints(img1,fordraw1,img1_keypoints);
imshow("img1_keypoints",img1_keypoints);
waitKey(0);

Mat img2=imread("images/dog2.jpg");
vector<KeyPoint> keypoints2;
Mat descriptors2;
my_ORB(img2,keypoints2,descriptors2);
//绘制角点
Mat img2_keypoints;
vector<KeyPoint> fordraw2;
for(int i=0;i<keypoints2.size();i++)
{
    int x=pow(1.2,keypoints2[i].octave)*keypoints2[i].pt.x;
    int y=pow(1.2,keypoints2[i].octave)*keypoints2[i].pt.y;
    KeyPoint keypoint;
    keypoint.pt.x=x;
    keypoint.pt.y=y;
    fordraw2.push_back(keypoint);
}
drawKeypoints(img2,fordraw2,img2_keypoints);
imshow("img2_keypoints",img2_keypoints);
waitKey(0);
Mat matchresult;
vector<DMatch> matches;
my_BF(descriptors1,descriptors2,matches);
//my_match(img1,img2,fordraw1,fordraw2,descriptors1,descriptors2,matchresult);
drawMatches(img1, fordraw1, img2, fordraw2, matches, matchresult);
imshow("matchresult",matchresult);
waitKey(0);


}