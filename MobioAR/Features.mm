//
//  Features.m
//  MobioAR
//
//  Created by Steve JobsOne on 12/17/18.
//  Copyright © 2018 MobioApp Limited. All rights reserved.
//

#import "Features.h"

@interface Features () {
    Ptr<FeatureDetector> detector;
    Ptr<DescriptorExtractor> descriptor;
}

@end

@implementation Features
/**
 Initialize ORB Detector & Descriptor for train image
 @param imageName The image Name of train images
 @param modelPath The name of 3D model path
 */
- (instancetype)initWithImage:(NSString *)imageName modelPath:(nonnull NSString *)modelPath
{
    self = [super init];
    if (self) {
        //-- initialization
        self.imageName = imageName;
        self.modelPath = modelPath;
        detector = ORB::create(1000);
        descriptor = ORB::create(1000);
        
        UIImage *image = [UIImage imageNamed:imageName];
        [self processImage:[self cvMatFromUIImage:image]];
    }
    
    return self;
}
/**
 Initialize ORB Detector & Descriptor for Camera Image
 @param cameraImage The image Matrix of camera image
 */
- (instancetype)initWithCameraImage:(Mat &)cameraImage
{
    self = [super init];
    if (self) {
        //-- initialization
        /**
         Initialize ORB Detector
         */
        detector = ORB::create(1000);
        /**
         Initialize ORB Descriptor
         */
        descriptor = ORB::create(1000);
        [self processCameraImage:cameraImage];
    }
    
    return self;
}
/**
 Process detector and descriptor for Input image
 @param inputImage The image Matrix of train image
 */
- (void)processImage:(cv::Mat)inputImage
{
    _inputImage = inputImage;
    cv::cvtColor(_inputImage, _inputImage, CV_BGR2GRAY);
    
    //detector->detectAndCompute(_inputImage, cv::Mat(), keypoints, descriptors);
    //-- Step 1: testing Oriented FAST Corner position
    detector->detect (_inputImage, _keypoints);
    
    //-- Step 2: Calculate the BRIEF description based on the corner position
    descriptor->compute (_inputImage, _keypoints, _descriptors);
}
/**
 Process detector and descriptor for cameraImage
 @param cameraImage The image Matrix of camera captured image
 */
- (void)processCameraImage:(Mat &)cameraImage
{
    _inputImage = cameraImage;
    cv::cvtColor(cameraImage, cameraImage, CV_BGR2GRAY);
    
    //-- Step 1: testing Oriented FAST Corner position
    detector->detect (cameraImage, _keypoints);
    
    //-- Step 2: Calculate the BRIEF description based on the corner position
    descriptor->compute (cameraImage, _keypoints, _descriptors);
    
}
/**
 get Image Matrix from UIImage
 @param image as UIImage
 @returns a matrix as cv::Mat
 */
- (cv::Mat)cvMatFromUIImage:(UIImage *)image
{
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    
    cv::Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to  data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    // CGColorSpaceRelease(colorSpace);
    
    return cvMat;
}

@end

