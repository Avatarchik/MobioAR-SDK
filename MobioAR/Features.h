//
//  Features.h
//  MobioAR
//
//  Created by Steve JobsOne on 12/17/18.
//  Copyright © 2018 MobioApp Limited. All rights reserved.
//

#import <opencv2/opencv.hpp>
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

using namespace cv;
using namespace std;

@interface Features : NSObject
/**
 Declare  a property for image keypoints
 */
@property std::vector<KeyPoint> keypoints;
/**
 Declare  a property for image descriptor
 */
@property Mat descriptors;
/**
 Declare  a property for input image Matrix
 */
@property cv::Mat inputImage;
/**
 Declare  a property for image name
 */
@property (strong, nonatomic) NSString *imageName;
/**
 Declare  a property for model path
 */
@property (strong, nonatomic) NSString *modelPath;

- (instancetype)initWithImage:(NSString *)imageName modelPath:(NSString *)modelPath;
- (instancetype)initWithCameraImage:(Mat&)cameraImage;

@end

NS_ASSUME_NONNULL_END


