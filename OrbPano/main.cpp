#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <vector>
#include<opencv2/legacy/legacy.hpp>

using namespace cv;
using namespace std;

Size size;

int main()
{
	Mat img_1 = imread("C:\\1.png");
	Mat img_2 = imread("C:\\2.png");
	if (!img_1.data || !img_2.data)
	{
		cout << "error reading images " << endl;
		return -1;
	}

	ORB orb;
	vector<KeyPoint> keyPoints_1, keyPoints_2;
	Mat descriptors_1, descriptors_2;

	orb(img_1, Mat(), keyPoints_1, descriptors_1);
	orb(img_2, Mat(), keyPoints_2, descriptors_2);
	
	BruteForceMatcher<HammingLUT> matcher;
	vector<DMatch> matches;
	matcher.match(descriptors_1, descriptors_2, matches);

	double max_dist = 0; double min_dist = 100;
	//-- Quick calculation of max and min distances between keypoints
	for( int i = 0; i < descriptors_1.rows; i++ )
	{ 
		double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}
	printf("-- Max dist : %f \n", max_dist );
	printf("-- Min dist : %f \n", min_dist );
	//-- Draw only "good" matches (i.e. whose distance is less than 0.6*max_dist )
	//-- PS.- radiusMatch can also be used here.
	std::vector< DMatch > good_matches;
	for( int i = 0; i < descriptors_1.rows; i++ )
	{ 
		if( matches[i].distance < 0.6*max_dist )
		{ 
			good_matches.push_back( matches[i]); 
		}
	}
	
	// localize the object
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;
	
	for( int i = 0; i < good_matches.size(); i++ )
	 {
	 //-- Get the keypoints from the good matches
	 obj.push_back( keyPoints_1[ good_matches[i].queryIdx ].pt );
	 scene.push_back( keyPoints_2[ good_matches[i].trainIdx ].pt );
	 }
 
	// Find the Homography Matrix
	 Mat H = findHomography( obj, scene, CV_RANSAC );
	 // Use the Homography Matrix to warp the images
	 cv::Mat result;
	 warpPerspective(img_1,result,H,cv::Size(img_1.cols+img_2.cols,img_1.rows));
	 cv::Mat half(result,cv::Rect(0,0,img_2.cols,img_2.rows));
	 img_2.copyTo(half);
	 imshow( "Result", result );
 
 waitKey(0);
 return 0;
}