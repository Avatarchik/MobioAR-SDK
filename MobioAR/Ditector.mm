#import <opencv2/opencv.hpp>
#import <UIKit/UIKit.h>
#import <opencv2/highgui.hpp>
#import <opencv2/videoio/cap_ios.h>
#import <opencv2/objdetect/objdetect.hpp>
#import <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui.hpp>

#import "Ditector.h"
#import "ARDataSet.h"
#include <map>
#include <vector>
#import "DataSet.h"
#import <SceneKit/SceneKit.h>

#define MIN_KPS_IN_FRAME            300     // need to detect at least these keypoints in reference image
#define MIN_INLIER_COUNT            30      // should have at least these many matches
#define CAM_HEIGHT_FROM_FLOOR       75      // assumed distance between device and floor
#define NN_MATCH_RATIO              0.8f    // Nearest-neighbour matching ratio
#define RANSAC_THRESH               2.5f    // RANSAC inlier threshold for solvePnp

using namespace cv;
using namespace std;

/**
 Simple class to represent an AR Ditector.
 */
@interface Ditector () <CvVideoCameraDelegate>
/**
 Declare a property for Opencv video Camera.
 */
@property (nonatomic, retain) CvVideoCamera *videoCamera;
/**
 Declare a property for train image Array.
 */
@property (strong, nonatomic) NSMutableArray *trainImageArray;
/**
 Declare a property for dataSet.
 */
@property (strong, nonatomic) DataSet *dataSet;
/**
 Declare a property for Checking frame processing completion.
 */
@property BOOL flag;

@end

@implementation Ditector
/**
 Initializie ARView with train images.
 */
- (instancetype)initWithARImage:(UIView *)arView trainModels:(nonnull NSArray *)trainingModels
{
    self = [super init];
    if (self) {
        /**
         Initializie OpenCV Video input Camera.
         */
        self.videoCamera = [[CvVideoCamera alloc] initWithParentView:arView];
        self.videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionBack;
        self.videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPreset352x288;
        self.videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
        self.videoCamera.defaultFPS = 30;
        self.videoCamera.grayscaleMode = NO;
        self.videoCamera.delegate = self;
        self.flag = NO;
        /**
         Initializie DataSet.
         */
        self.dataSet = [DataSet new];
        /**
         Get feature list for train images from DataSet Class.
         */
        self.trainImageArray = [self.dataSet getFeaturesListForTrainingModels:trainingModels];
        
        self.sceneView.scene = nil;
        /*self.sceneView.translatesAutoresizingMaskIntoConstraints = NO;
         
         UITapGestureRecognizer *tapGesture = [[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(handleTap:)];
         NSMutableArray *gestureRecognizers = [NSMutableArray array];
         [gestureRecognizers addObject:tapGesture];
         [gestureRecognizers addObjectsFromArray:self.sceneView.gestureRecognizers];
         self.sceneView.gestureRecognizers = gestureRecognizers;
         
         self.sceneView.backgroundColor = [UIColor redColor];
         [v addSubview:self.sceneView];
         
         //Creating the same constraints using Layout Anchors
         UILayoutGuide *arViewMargin = arView.layoutMarginsGuide;
         
         [self.sceneView.topAnchor constraintEqualToAnchor:arViewMargin.topAnchor constant:-8].active = YES;
         [self.sceneView.bottomAnchor constraintEqualToAnchor:arViewMargin.bottomAnchor constant:8].active = YES;
         [self.sceneView.leadingAnchor constraintEqualToAnchor:arViewMargin.leadingAnchor constant:-8].active = YES;
         [self.sceneView.trailingAnchor constraintEqualToAnchor:arViewMargin.trailingAnchor constant:8].active = YES;*/
    }
    
    return self;
}

/**
 Refine Homography and matches after description matching
 @param queryImage : The Features of camera image
 @param trainImage : The Features of train image
 @param threshold : The threshold that define
 @param matches : The DMatch vector from image descriptor matches
 
 */
-(void) refineMatchesWithHomography:(Features *)queryImage trainImage:(Features *)trainImage thresh:(double)threshold matches:(std::vector<cv::DMatch>&)matches
{
    std::vector<cv::Point2f> points1(matches.size());
    std::vector<cv::Point2f> points2(matches.size());
    
    for (size_t i = 0; i < matches.size(); i++)
    {
        points1[i] = queryImage.keypoints[matches[i].queryIdx].pt;
        points2[i] = trainImage.keypoints[matches[i].trainIdx].pt;
    }
    
    // Find homography matrix and get inliers mask
    Mat inliersMask;
    Mat homography = cv::findHomography(points1, points2, CV_FM_RANSAC, threshold, inliersMask, 2000, 0.995);
    
    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point2f(0, 0);
    obj_corners[1] = Point2f( (float)queryImage.inputImage.cols, 0 );
    obj_corners[2] = Point2f( (float)queryImage.inputImage.cols, (float)queryImage.inputImage.rows );
    obj_corners[3] = Point2f( 0, (float)queryImage.inputImage.rows );
    
    //derive camera intrinsic mx from GL projection-mx
    cv::Mat cameraIntrinsicMat = cv::Mat::zeros(3, 3, CV_32F);
    UIImage *image = [UIImage new];
    float focalLength = image.size.height / 2 / tan((0.84 * M_PI / 180)/2);
    
    cameraIntrinsicMat.at<float>(0, 0) = focalLength;
    cameraIntrinsicMat.at<float>(1, 1) = focalLength;
    
    //principal point = image center
    cameraIntrinsicMat.at<float>(0, 2) = image.size.height / 2;
    cameraIntrinsicMat.at<float>(1, 2) = image.size.height / 2;
    
    cameraIntrinsicMat.at<float>(2, 2) = 1.;
    
    std::vector<cv::DMatch> inliers;
    
    for (size_t i=0; i<matches.size(); i++)
    {
        if ((int)inliersMask.at<uchar>(0,0) != 0)
        {
            inliers.push_back(matches[i]);
        }
    }
    // show 3D model and send the trackable name if inlier matches crossed the minimum boundary.
    if(inliers.size()>= 8 && trainImage.modelPath && _sceneView.scene == nil)
    {
        [self showModelForDetectedImage:trainImage.modelPath];
        [self.delegate getTrackableImage:trainImage.imageName];
        
    } else {
        
       // _sceneView.scene = nil;
        self->_flag = NO;
    }
    
    
}

/**
 Show 3D model for detected image
 @param modelPath : The path for scenkit 3D model
 */
-(void)showModelForDetectedImage:(NSString *)modelPath
{
    
    SCNScene *scene = [SCNScene sceneNamed:modelPath];
    
    dispatch_async(dispatch_get_main_queue(), ^{
        
        self->_sceneView.scene = scene;
        // allows the user to manipulate the camera
        self->_sceneView.allowsCameraControl = YES;
        // show statistics such as fps and timing information
        self->_sceneView.showsStatistics = NO;
    });
}

/**
 Converts image keypoints into image points
 @param keypoints : The image Keypoints as std::vector<cv::KeyPoint>
 @returns point as std::vector<cv::Point2f
 */
-(std::vector<cv::Point2f>) Keypoint2Point:(std::vector<cv::KeyPoint>) keypoints
{
    std::vector<cv::Point2f> vectorOfPoints;
    for(unsigned i = 0; i < keypoints.size(); i++)
    {
        vectorOfPoints.push_back(keypoints[i].pt);
    }
    
    return vectorOfPoints;
}
/**
 Match camera image Descriptor with train images descriptor
 @param queryImage : The Features of camera image
 @param trainImage : The Features of train image
 */
-(void) matchingDescriptor:(Features *)queryImage trainImage:(Features *)trainImage
{
    //-- Step 3: Match the BRIEF descriptors in the two images, using Hamming distance
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    std::vector<DMatch> matches, good_matches;
    //BFMatcher matcher ( NORM_HAMMING );
    matcher->match (queryImage.descriptors, trainImage.descriptors, matches);
    
    //-- Step 4: Matching point pairs
    double min_dist=10000, max_dist = 0, distance_threshold = 25.0;
    int good_match_threshold = 15;
    
    //Find the minimum and maximum distance between all matches, which is the distance between the two most similar and least similar points
    for ( int i = 0; i < matches.size(); i++ )
    {
        double dist = matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }
    
    //When the distance between the descriptors is greater than twice the minimum distance, the match is considered to be incorrect. But sometimes the minimum distance is very small, and an empirical value of 30 is set as the upper limit.
    for ( int i = 0; i < matches.size(); i++ )
    {
        if ( matches[i].distance <= min (1.8*min_dist, distance_threshold)) {//min (1.8*min_dist, distance_threshold)
            good_matches.push_back ( matches[i]);
        }
    }
    
    if (good_matches.size() >= good_match_threshold)
    {
        NSLog(@"good matches : %lu",good_matches.size());
        cout <<"good matches = " << good_matches.size()<< "\n";
        [self refineMatchesWithHomography:queryImage trainImage:trainImage thresh:3 matches:good_matches];
    }
    else
    {
        // _sceneView.scene = nil;
        self->_flag = NO;
    }
}
#ifdef __cplusplus


// Macth Camera Image feature with train image feature by KNN Robust Matcher..

-(void) matchingDescriptorWithKNNMatcher:(Features *)queryImage trainImage:(Features *)trainImage {
    
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    std::vector<std::vector<cv::DMatch> > descriptorMatches;
    std::vector<DMatch> good_matches,better_matches;
    std::vector<cv::KeyPoint> sourceMatches, queryMatches;
    
    
    // knn-match with k = 2
    matcher->knnMatch(trainImage.descriptors, queryImage.descriptors, descriptorMatches, 2);
    
    // save matches within a certain distance threshold
    for (unsigned i = 0; i < descriptorMatches.size(); i++) {
        if (descriptorMatches[i][0].distance < NN_MATCH_RATIO * descriptorMatches[i][1].distance) {
            
            good_matches.push_back(descriptorMatches[i][0]);
            sourceMatches.push_back(trainImage.keypoints[descriptorMatches[i][0].queryIdx]);
            queryMatches.push_back(queryImage.keypoints[descriptorMatches[i][0].trainIdx]);
        }
    }
    cout <<"good matches in robust = " << good_matches.size()<< "\n";
    
    if(good_matches.size() > 70) {
        
        if(self.sceneView.scene == nil) {
            
            [self showModelForDetectedImage:trainImage.modelPath];
        }
        
        [self.delegate getTrackableImage:trainImage.imageName];
        
    } else {
        
        //_sceneView.scene = nil;
        self->_flag = NO;
       // return;
    }
    
}

/**
 Get image Matrix from opencv Camera Capture delegate.
 */
- (void)processImage:(Mat&)cameraImage
{
    if(cameraImage.rows<1 || cameraImage.cols<1)
    {
        return;
    }
    
    if (!_flag)
    {
        _flag = YES;
        
        // Step 2 : Get camera image keypoints and descriptors
        cv::Mat singleFrame = cameraImage;
        Features *queryImage = [self.dataSet getFeaturesForCameraImage:singleFrame];
        
        //ignoring number of frame based on less color varience before extracting keypoints
        // boundary set to 200 for keypoints size..
        if(queryImage.keypoints.size() >= 200 )
        {
            dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
                for (Features *trainImage in self->_trainImageArray)
                {
                    if (trainImage.inputImage.rows>0 && trainImage.inputImage.cols>0)
                    {
                        NSLog(@"getting frame");
                        [self matchingDescriptor:queryImage trainImage:trainImage];
                       
                      //  [self matchingDescriptorWithKNNMatcher:queryImage trainImage:trainImage];
                        
                    }
                    else
                    {
                        
                        self->_flag = NO;
                    }
                }
            });
        }
        else
        {
           
            self->_flag = NO;
        }
    }
}

#endif





/**
 Get touch on Scenekit model.
 */
- (void) handleTap:(UIGestureRecognizer*)gestureRecognize
{
    // check what nodes are tapped
    CGPoint p = [gestureRecognize locationInView:_sceneView];
    NSArray *hitResults = [_sceneView hitTest:p options:nil];
    
    // check that we clicked on at least one object
    if([hitResults count] > 0)
    {
        // retrieved the first clicked object
        SCNHitTestResult *result = [hitResults objectAtIndex:0];
        
        // get its material
        SCNMaterial *material = result.node.geometry.firstMaterial;
        
        // highlight it
        [SCNTransaction begin];
        [SCNTransaction setAnimationDuration:0.5];
        
        // on completion - unhighlight
        [SCNTransaction setCompletionBlock:^{
            [SCNTransaction begin];
            [SCNTransaction setAnimationDuration:0.5];
            
            material.emission.contents = [UIColor blackColor];
            
            [SCNTransaction commit];
        }];
        
        material.emission.contents = [UIColor redColor];
        
        [SCNTransaction commit];
    }
}

- (void)startAR
{
    [self.videoCamera start];
}

- (void)stopAR
{
    self.flag = NO;
    self.sceneView.scene = nil;
    [self.videoCamera stop];
}

- (void)refreshAR
{
    self.flag = NO;
    self.sceneView.scene = nil;
    [self.videoCamera start];
}

@end



