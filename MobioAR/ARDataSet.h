//
//  ARDataSet.h
//  MobioAR
//
//  Created by Steve JobsOne on 12/24/18.
//  Copyright © 2018 MobioApp Limited. All rights reserved.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface ARDataSet : NSObject
/**
 Declare  a property for training models
 */
@property (nonatomic, strong) NSMutableArray *trainingModels;
/**
 Declare  a function for adding model to dataset
 */
- (void)addModelToDataSet:(NSString *)modelPath trainImages:(NSArray *)trainingImages;

@end

NS_ASSUME_NONNULL_END
