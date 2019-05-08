//
//  DataSet.m
//  MobioAR
//
//  Created by Steve JobsOne on 12/17/18.
//  Copyright © 2018 MobioApp Limited. All rights reserved.
//

#import "DataSet.h"

@interface DataSet ()
/**
 Declare a property for image Features Array
 */
@property (strong, nonatomic) NSMutableArray *featuresArray;

@end

@implementation DataSet

- (instancetype)init
{
    self = [super init];
    if (self) {
        self.featuresArray = [NSMutableArray new];
    }
    return self;
}
/**
 Get feature for camera image
 @param cameraImage The Matrix of camera image
 @returns a features
 */
-(Features *)getFeaturesForCameraImage:(Mat &)cameraImage
{
    return [[Features alloc] initWithCameraImage:cameraImage];
}
/**
 Get feature list for train images
 @param trainingModels The Array of train images
 @returns a features Array as NSMutableArray
 */
-(NSMutableArray *)getFeaturesListForTrainingModels:(NSArray *)trainingModels
{
    for (NSDictionary *trainingModel in trainingModels)
    {
        NSArray *trainingImages = [trainingModel objectForKey:@"trainingImages"];
        for (NSString *image in trainingImages)
        {
            [self.featuresArray addObject:[[Features alloc] initWithImage:image modelPath:[trainingModel objectForKey:@"modelPath"]]];
        }
    }
    
    return self.featuresArray;
}

@end



