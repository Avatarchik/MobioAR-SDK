//
//  ARDataSet.m
//  MobioAR
//
//  Created by Steve JobsOne on 12/24/18.
//  Copyright © 2018 MobioApp Limited. All rights reserved.
//

#import "ARDataSet.h"

@implementation ARDataSet

- (instancetype)init
{
    self = [super init];
    if (self) {
        self.trainingModels = [NSMutableArray new];
    }
    
    return self;
}
/**
 Add model to DataSet
 @param modelPath as NSString
 @param trainingImages as NSArray
 */
- (void)addModelToDataSet:(NSString *)modelPath trainImages:(NSArray *)trainingImages
{
    NSMutableDictionary *dict = [[NSMutableDictionary alloc]init];
    [dict setObject:modelPath forKey:@"modelPath"];
    [dict setObject:trainingImages forKey:@"trainingImages"];
    [self.trainingModels addObject:dict];
}

@end
