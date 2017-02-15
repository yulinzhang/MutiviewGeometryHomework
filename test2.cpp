#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat H_static;

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

Mat initPointsForAffineInImage1()
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
    
    cout<<"A = "<<A<<endl;
    return A;

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

Mat getInfinityPoint(Mat point1, Mat point2, Mat point3, Mat point4)
{
	Mat line1=point1.cross(point2);
        Mat line2=point3.cross(point4);
        Mat infinity_point=line1.cross(line2);
	return infinity_point;
}
//image_1 the Mat structure of image to be recovered
//A contains all four points, and two pair of parallel lines
Mat affinity(Mat image_1, Mat A)
{
    cout<<"----------------------enter into affinity----------------------"<<endl;
    Mat infinity_point_1=getInfinityPoint(A.col(0), A.col(1), A.col(2), A.col(3));
    Mat infinity_point_2=getInfinityPoint(A.col(0), A.col(3), A.col(1), A.col(2));
    cout<<"infinity point1 = "<<infinity_point_1<<endl;
    cout<<"infinity point2 = "<<infinity_point_2<<endl;

    Mat infinity_line=infinity_point_1.cross(infinity_point_2);
    cout<<"infinity line = "<<infinity_line<<endl;
    Mat H=Mat::eye(3, 3, DataType<float>::type);
    
    float divider=infinity_line.at<float>(2,0);
    infinity_line=infinity_line/divider;
    cout<<"infinity line divided = "<<infinity_line<<endl;
    cout<<"Before stack, H="<<H<<endl;
    Mat infinity_line_row_vector;
    transpose(infinity_line, infinity_line_row_vector);
    infinity_line_row_vector.copyTo(H.row(2));
    cout<<"After stack inifity point, H="<<H<<endl;

    Mat dst_img_for_A;
    warpPerspective(image_1,dst_img_for_A, H, image_1.size());//DECOMP_SVD
    H_static=H;
    //showImage("Affinity Image For A", dst_img_for_A, Mat::zeros(1,1,DataType<float>::type));
    return dst_img_for_A;
}

int main(int argc, char** argv )
{
    Mat image_1;
    image_1 = imread("./image1.jpg", 1 );

    cout<<"----------------begin affinity-----------------"<<endl;
    Mat A=initPointsForAffineInImage1();
    showImage("original image", image_1, A);
    Mat dst_img=affinity(image_1, A);
    imwrite( "./q2_image_1_original.jpg", image_1 );
    imwrite( "./q2_image_1_affined.jpg", dst_img );

    /*for use of question 3
    cout<<"-----------------calc the new coordinates of the rectangle-------------------------"<<endl;
    Mat A_prime=H_static*A;
    A_prime.col(0)=A_prime.col(0)/A_prime.at<float>(2,0);
    A_prime.col(1)=A_prime.col(1)/A_prime.at<float>(2,1);
    A_prime.col(2)=A_prime.col(2)/A_prime.at<float>(2,2);
    A_prime.col(3)=A_prime.col(3)/A_prime.at<float>(2,3);
    const Scalar& black_color=Scalar( 0, 0, 0 );
    image_1=addLineToImage(image_1, A_prime, black_color);
    showImage("Black colored", dst_img, Mat::zeros(1,1,DataType<float>::type));
    cout<<"A_prime = "<<A_prime<<endl;
    */
    showImage("Destination Image", dst_img, Mat::zeros(1,1,DataType<float>::type));
    waitKey(0);
    return 0;
}
