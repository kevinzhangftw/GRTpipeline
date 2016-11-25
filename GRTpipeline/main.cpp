//
//  main.cpp
//  GRTpipeline
//
//  Created by Kevin Zhang on 2016-11-11.
//  Copyright © 2016 Kevin Zhang. All rights reserved.
//

#include <iostream>
#include <GRT/GRT.h>
#include "opencv2/core/version.hpp"

using namespace std;
using namespace GRT;

ClassificationData getInput();
//ANBC
GestureRecognitionPipeline configureANBCPipeline(ClassificationData trainData, ClassificationData testData);
//Adaboost
GestureRecognitionPipeline configureAdaBoostPipeline(ClassificationData trainData, ClassificationData testData);
//BAG
GestureRecognitionPipeline configureBAGPipeline(ClassificationData trainData, ClassificationData testData);
//KNN
GestureRecognitionPipeline configureKNNPipeline(ClassificationData trainData, ClassificationData testData);
//GMM
GestureRecognitionPipeline configureGMMPipeline(ClassificationData trainData, ClassificationData testData);
//
GestureRecognitionPipeline configurePipeline(GestureRecognitionPipeline pipeline, ClassificationData trainData, ClassificationData testData);

int main(int argc, const char * argv[]) {
    //input
    ClassificationData trainData = getInput();
    ClassificationData testData = trainData.split(80);
    
    //Create a new Gesture Recognition Pipeline using an Adaboost
    GestureRecognitionPipeline pipelineAdaBoost = configureAdaBoostPipeline(trainData, testData);
    cout << "AdaBoost Pipeline Test Accuracy: " << pipelineAdaBoost.getTestAccuracy() << endl;
    
    return 0;
}


ClassificationData getInput(){
    ClassificationData csvData;
//    csvData.load( "classification_data.csv" );
    csvData.load( "traindata.csv" );
    csvData.printStats();
    cout << "csv formatted classification data OK\n";
    return csvData;
}

GestureRecognitionPipeline configureANBCPipeline(ClassificationData trainData, ClassificationData testData){
    //Create a new Gesture Recognition Pipeline using an Adaptive Naive Bayes Classifier
    GestureRecognitionPipeline pipelineANBC;
    ANBC anbc = ANBC();
    anbc.enableNullRejection(true);
    pipelineANBC.setClassifier( anbc );
    pipelineANBC = configurePipeline(pipelineANBC, trainData, testData);

    return pipelineANBC;
}

GestureRecognitionPipeline configureAdaBoostPipeline(ClassificationData trainData, ClassificationData testData){
    //Create a new Gesture Recognition Pipeline using an AdaBoost
    GestureRecognitionPipeline pipelineAdaBoost;
    AdaBoost adaboost = AdaBoost();
    adaboost.enableNullRejection(true);
    pipelineAdaBoost.setClassifier( adaboost );
    pipelineAdaBoost = configurePipeline(pipelineAdaBoost, trainData, testData);
    
    return pipelineAdaBoost;
}

//BAG
GestureRecognitionPipeline configureBAGPipeline(ClassificationData trainData, ClassificationData testData){
    //Create a new Gesture Recognition Pipeline using an BAG
    GestureRecognitionPipeline pipelineBAG;
    pipelineBAG.setClassifier( BAG() );
    //TODO: NULL Rejection
    pipelineBAG = configurePipeline(pipelineBAG, trainData, testData);
    
    return pipelineBAG;
}

//KNN
GestureRecognitionPipeline configureKNNPipeline(ClassificationData trainData, ClassificationData testData){
    //Create a new Gesture Recognition Pipeline using an KNN
    GestureRecognitionPipeline pipelineKNN;
    KNN knn = KNN();
    knn.enableNullRejection(true);
    pipelineKNN.setClassifier( knn );
    pipelineKNN = configurePipeline(pipelineKNN, trainData, testData);
    
    return pipelineKNN;
}
//GMM
GestureRecognitionPipeline configureGMMPipeline(ClassificationData trainData, ClassificationData testData){
    //Create a new Gesture Recognition Pipeline using an GMM
    GestureRecognitionPipeline pipelineGMM;
    //TODO: NULL Rejection
    pipelineGMM.setClassifier( GMM());
    pipelineGMM = configurePipeline(pipelineGMM, trainData, testData);
    
    return pipelineGMM;
}

GestureRecognitionPipeline configurePipeline(GestureRecognitionPipeline pipelineInput, ClassificationData trainData, ClassificationData testData){
    GestureRecognitionPipeline pipelineConfigured = pipelineInput;
    //Train the pipeline using the training data
    cout << "Training model..." << endl;
    if( !pipelineConfigured.train( trainData ) ){
        cout << "ERROR: Failed to train the pipeline!\n";
        //return EXIT_FAILURE;
    }
    
    //Save the pipeline to a file
    if( !pipelineConfigured.save( "PipelineToTest" ) ){
        cout << "ERROR: Failed to save the pipeline!\n";
        //return EXIT_FAILURE;
    }
    
    //Load the pipeline from a file
    if( !pipelineConfigured.load( "PipelineToTest" ) ){
        cout << "ERROR: Failed to load the pipeline!\n";
        //return EXIT_FAILURE;
    }
    
    //Test the pipeline using the test data
    cout << "Testing model..." << endl;
    if( !pipelineConfigured.test( testData ) ){
        cout << "ERROR: Failed to test the pipeline!\n";
        //return EXIT_FAILURE;
    }
    
    return pipelineConfigured;
}

