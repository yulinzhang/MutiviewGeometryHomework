#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat addLineToImage(Mat image, Mat points_on_line, const Scalar& color)
{
    Point2d pt1= Point(points_on_line.at<float>(0,0), points_on_line.at<float>(1,0)); 
    Point2d pt2= Point(points_on_line.at<float>(0,1), points_on_line.at<float>(1,1)); 
    Point2d pt3= Point(points_on_line.at<float>(0,2), points_on_line.at<float>(1,2)); 
    Point2d pt4= Point(points_on_line.at<float>(0,3), points_on_line.at<float>(1,3)); 
    
    Point2d pt5= Point(points_on_line.at<float>(0,4), points_on_line.at<float>(1,4)); 
    Point2d pt6= Point(points_on_line.at<float>(0,5), points_on_line.at<float>(1,5)); 
    Point2d pt7= Point(points_on_line.at<float>(0,6), points_on_line.at<float>(1,6)); 
    Point2d pt8= Point(points_on_line.at<float>(0,7), points_on_line.at<float>(1,7));
 
    line(image, pt1, pt2, color, 4);
    line(image, pt3, pt4, color, 4);
    line(image, pt5, pt6, color, 4);
    line(image, pt7, pt8, color, 4);

    return image;
}

Mat initPointsForAffineInImage1()
{
    //initialize points in first image

    Mat A(3,8,DataType<float>::type);
    
    A.at<float>(0,0)=1827;
    A.at<float>(1,0)=1534;
    A.at<float>(2,0)=1;

    A.at<float>(0,1)=1930;
    A.at<float>(1,1)=1526;
    A.at<float>(2,1)=1;

    A.at<float>(0,2)=2056;
    A.at<float>(1,2)=1574;
    A.at<float>(2,2)=1;

    A.at<float>(0,3)=1930;
    A.at<float>(1,3)=1526;
    A.at<float>(2,3)=1;

    A.at<float>(0,4)=939;
    A.at<float>(1,4)=1735;
    A.at<float>(2,4)=1;

    A.at<float>(0,5)=1204;
    A.at<float>(1,5)=1796;
    A.at<float>(2,5)=1;

    A.at<float>(0,6)=1047;
    A.at<float>(1,6)=1809;
    A.at<float>(2,6)=1;

    A.at<float>(0,7)=1078;
    A.at<float>(1,7)=1722;
    A.at<float>(2,7)=1;
    
    cout<<"A = "<<A<<endl;
    return A;
}

void showImage(String window_name, Mat image, Mat marked_points)
{
    if(countNonZero(marked_points) >= 2){
	const Scalar& red_color=Scalar( 0, 0, 255 );
 	image=addLineToImage(image, marked_points, red_color);
    }
    namedWindow(window_name, WINDOW_NORMAL);
    imshow(window_name, image);
    return;
}

Mat constructOrthogonalLine(Mat A)
{
    //initialize four lines: 1 and 2 are orthogonal, 3 and 4 are orthogonal
    Mat T(3,4,DataType<float>::type);
    A.col(0).cross(A.col(1)).copyTo(T.col(0));//point 1 and point 2
    A.col(2).cross(A.col(3)).copyTo(T.col(1));//point 2 and point 3
    A.col(4).cross(A.col(5)).copyTo(T.col(2));//point 1 and point 3
    A.col(6).cross(A.col(7)).copyTo(T.col(3));//point 2 and point 4
    return T;
}

Mat constructLeftS(Mat T)
{
    Mat l1=T.col(0);
    Mat m1=T.col(1);
    Mat l2=T.col(2);
    Mat m2=T.col(3);
    Mat LeftS(2,3,DataType<float>::type);
    LeftS.at<float>(0,0)=l1.at<float>(0,0)*m1.at<float>(0,0);
    LeftS.at<float>(0,1)=l1.at<float>(0,0)*m1.at<float>(1,0)+l1.at<float>(1,0)*m1.at<float>(0,0);
    LeftS.at<float>(0,2)=l1.at<float>(1,0)*m1.at<float>(1,0);
    LeftS.at<float>(1,0)=l2.at<float>(0,0)*m2.at<float>(0,0);
    LeftS.at<float>(1,1)=l2.at<float>(0,0)*m2.at<float>(1,0)+l2.at<float>(1,0)*m2.at<float>(0,0);
    LeftS.at<float>(1,2)=l2.at<float>(1,0)*m2.at<float>(1,0);
    return LeftS;
}

Mat constructS(Mat S_col)
{ 
    Mat S=Mat::eye(2,2,DataType<float>::type);
    float s11=S_col.at<float>(0,0);
    float s12=S_col.at<float>(1,0);
    float s22=S_col.at<float>(2,0);
    S.at<float>(0,0)=s11;
    S.at<float>(0,1)=s12;
    S.at<float>(1,0)=s12;
    S.at<float>(1,1)=s22;
    return S;
}

Mat constructHomography(Mat K)
{
    Mat H=Mat::eye(3,3,DataType<float>::type);
    H.at<float>(0,0)=K.at<float>(0,0);
    H.at<float>(0,1)=K.at<float>(0,1);
    H.at<float>(1,0)=K.at<float>(1,0);
    H.at<float>(1,1)=K.at<float>(1,1); 
    return H;
}

int main(int argc, char** argv )
{
    Mat image_1,image_2;
    image_1 = imread("./q2_image_1_affined.jpg", 1 );

    cout<<"----------------begin rectification-----------------"<<endl;
    Mat A=initPointsForAffineInImage1();
    const Scalar& color=Scalar( 0, 0, 0 );
    image_1=addLineToImage(image_1, A, color);
    

    Mat T=constructOrthogonalLine(A);
    Mat LeftS=constructLeftS(T);
    Mat D_temp,U_temp,Vt,V;
    SVD::compute(LeftS, D_temp, U_temp, Vt);
    
    transpose(Vt,V);
    cout<<"V = "<<V<<endl;
    Mat S_col=V.col(1);
    cout<<"S_col = "<<S_col<<endl;
    Mat S=constructS(S_col);
    //S=S*-1;
    //S=S/S_col.at<float>(2,0);
    //S.at<float>(0,0)=-1*S.at<float>(0,0);
    cout<<"S = "<<S<<endl;

    Mat D,U,Vt_S; 
    SVD::compute(S,D,U,Vt_S);
    cout<<"U = "<<U<<endl;
    cout<<"Vt_S = "<<Vt_S<<endl;
    cout<<"D = "<<D<<endl;

    Mat D_sqrt;
    sqrt(D,D_sqrt);
    cout<<"D_sqrt = "<<D_sqrt<<endl;
    Mat D_matrix_sqrt=Mat::eye(2,2,DataType<float>::type);
    D_matrix_sqrt.at<float>(0,0)=D_sqrt.at<float>(0,0);
    D_matrix_sqrt.at<float>(1,1)=D_sqrt.at<float>(1,0);
    cout<<"D matrix sqrt = "<<D_matrix_sqrt<<endl;
    Mat U_transpose;
    transpose(U,U_transpose);
    Mat K=U*D_matrix_sqrt*U_transpose;
    cout<<"K = "<<K<<endl;
    Mat K_transpose;
    transpose(K,K_transpose);
    cout<<"K*K = "<<K*K_transpose<<endl;
    Mat H=constructHomography(K);
    cout<<"H = "<<H<<endl;
    Mat dst_img;
    warpPerspective(image_1,dst_img, H, image_1.size());
    
    imwrite( "./q3_image_1_original.jpg", image_1 );
    imwrite( "./q3_image_1_affined.jpg", dst_img );
    showImage("Original Image", image_1, Mat::zeros(1,1,DataType<float>::type));
    showImage("Destination Image", dst_img, Mat::zeros(1,1,DataType<float>::type));

    waitKey(0);
    return 0;
}
