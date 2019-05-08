//
//  Ditector.h
//  MobioAR
//
//  Created by Steve JobsOne on 12/17/18.
//  Copyright © 2018 MobioApp Limited. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <SceneKit/SceneKit.h>
/**
 Objective C Protocol
 We use this to get the Callback result of AR Detection
 */
@protocol MobioARDelegate;

NS_ASSUME_NONNULL_BEGIN

@interface Ditector : NSObject
/**
 SCNView Property
 Create a SCNView property
 */
@property (strong, nonatomic) SCNView *sceneView;
/**
 MobioARDelegate Property
 Create a MobioARDelegate property to use the Ditector Delegate
 */
@property (nonatomic, weak) IBOutlet id <MobioARDelegate> delegate;

- (instancetype)initWithARImage:(UIView *)arView trainModels:(NSArray *)trainingModels;
/**
 Start AR
 */
- (void)startAR;
/**
 Stop AR
 */
- (void)stopAR;
/**
 Refresh AR
 */
- (void)refreshAR;

@end

#pragma mark -

@protocol MobioARDelegate <NSObject>

@optional
/**
 MobioARDelegate Callback
 Create a function for supporting the MobioARDelegate Callback
 */
- (void)getTrackableImage:(NSString *)trackableImage;

@end

NS_ASSUME_NONNULL_END
