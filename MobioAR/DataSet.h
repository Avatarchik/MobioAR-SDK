//
//  DataSet.h
//  MobioAR
//
//  Created by Steve JobsOne on 12/17/18.
//  Copyright © 2018 MobioApp Limited. All rights reserved.
//

#import <opencv2/opencv.hpp>
#import <UIKit/UIKit.h>
#import <Foundation/Foundation.h>
#import "Features.h"

NS_ASSUME_NONNULL_BEGIN

@interface DataSet : NSObject

-(Features *)getFeaturesForCameraImage:(Mat&)cameraImage;
-(NSMutableArray *)getFeaturesListForTrainingModels:(NSArray *)trainingModels;

@end

NS_ASSUME_NONNULL_END
