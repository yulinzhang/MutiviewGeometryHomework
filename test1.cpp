#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat initPointsInImage1(int image_name)
{

//initialize points in first image
    Mat A(3,4,DataType<float>::type);

    A.at<float>(0,0)=1568;
    A.at<float>(1,0)=1317;
    A.at<float>(2,0)=1;

    A.at<float>(0,1)=1634;
    A.at<float>(1,1)=1292;
    A.at<float>(2,1)=1;

    A.at<float>(0,2)=1718;
    A.at<float>(1,2)=1316;
    A.at<float>(2,2)=1;

    A.at<float>(0,3)=1654;
    A.at<float>(1,3)=1341;
    A.at<float>(2,3)=1;
    return A;
}

Mat initPointsInImage2()
{

//initialize points in second image
    Mat B(3,4,DataType<float>::type);

    B.at<float>(0,0)=1500;
    B.at<float>(1,0)=1300;
    B.at<float>(2,0)=1;

    B.at<float>(0,1)=1550;
    B.at<float>(1,1)=1300;
    B.at<float>(2,1)=1;
   
    B.at<float>(0,2)=1550;
    B.at<float>(1,2)=1350;
    B.at<float>(2,2)=1;

    B.at<float>(0,3)=1500;
    B.at<float>(1,3)=1350;
    B.at<float>(2,3)=1;
   
 return B;
}

Mat addLineToImage(Mat image, Mat points_on_line, const Scalar& color)
{
    Point2d pt1= Point(points_on_line.at<float>(0,0), points_on_line.at<float>(1,0)); 
    Point2d pt2= Point(points_on_line.at<float>(0,1), points_on_line.at<float>(1,1)); 
    Point2d pt3= Point(points_on_line.at<float>(0,2), points_on_line.at<float>(1,2)); 
    Point2d pt4= Point(points_on_line.at<float>(0,3), points_on_line.at<float>(1,3)); 
    
    line(image, pt1, pt2, color, 2);
    line(image, pt2, pt3, color, 2);
    line(image, pt3, pt4, color, 2);
    line(image, pt1, pt4, color, 2);
    return image;
}


void showImage(String window_name, Mat image, Mat marked_points)
{
    if(countNonZero(marked_points) >= 1){
	const Scalar& red_color=Scalar( 0, 0, 255 );
 	image=addLineToImage(image, marked_points, red_color);
    }
    namedWindow(window_name, WINDOW_NORMAL);
    imshow(window_name, image);
    return;
}

//A contains the points as col vectors in image a, B contains those in image b
//first  construct the matrix T for transformation equation
//second inverse T to to get H
Mat solveProjectiveMapping(Mat A, Mat B)
{
    Mat T=Mat::zeros(8,8,DataType<float>::type);
    Mat P(8,1,DataType<float>::type);

    for(int i=0;i<=7;i=i+2)//i=0,2,4,6
    {
	int j=i/2;//j=0,1,2,3
        float x=A.at<float>(0,j);
	float y=A.at<float>(1,j);
 	float x_prime=B.at<float>(0,j);
	float y_prime=B.at<float>(1,j);
	cout<<"("<<x<<","<<y<<")-----"<<"("<<x_prime<<","<<y_prime<<")"<<endl;

	T.at<float>(i,0)=x;
        T.at<float>(i,1)=y;
	T.at<float>(i,2)=1;
        T.at<float>(i,6)=-1*x_prime*x;
	T.at<float>(i,7)=-1*x_prime*y;

	T.at<float>(i+1,3)=x;
	T.at<float>(i+1,4)=y;
	T.at<float>(i+1,5)=1;
	T.at<float>(i+1,6)=-1*y_prime*x;
	T.at<float>(i+1,7)=-1*y_prime*y;

	P.at<float>(i,0)=x_prime;
	P.at<float>(i+1,0)=y_prime;
    }

    cout<<"T="<<T<<endl;
    cout<<"P="<<P<<endl;
    Mat H_vector=T.inv()*P;//DECOMP_SVD
    Mat H(3,3,DataType<float>::type);
    for(int i=0;i<9;i++)
    {
	H.at<float>(i/3,i%3)=H_vector.at<float>(i,0);
    }
    H.at<float>(2,2)=1;
    return H;
}

Mat rectification(Mat image, Mat A, Mat B)
{
    Mat H=solveProjectiveMapping(A,B);
    cout<<"matrix H="<<H<<endl;

    showImage("Raw Image",image,A);

    Mat dst_img;
    warpPerspective(image,dst_img, H, image.size());	
    return dst_img;
}

int main(int argc, char** argv )
{
    Mat image_1,image_2;
    image_1 = imread("./image1.jpg", 1 );
    image_2 = imread("./image2.jpg",1);

    if ( !image_1.data || !image_2.data )
    {
        printf("No image data \n");
        return -1;
    }

    
    Mat A=initPointsInImage1(1);
    Mat B=initPointsInImage2();
    cout<<"matrix A="<<A<<endl;
    Mat dst_img=rectification(image_1, A, B);
    showImage("Destination Image", dst_img, Mat::zeros(1,1,DataType<float>::type));
    imwrite( "./q1_image_1_original.jpg", image_1 );
    imwrite( "./q1_image_1_rectified.jpg", dst_img );
//    A=initPointsInImage1(2);
//    rectification(image_2, A, B);

    waitKey(0);

    return 0;
}
