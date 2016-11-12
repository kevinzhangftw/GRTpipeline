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
GestureRecognitionPipeline configureANBCPipeline(ClassificationData trainData, ClassificationData testData);
GestureRecognitionPipeline configureAdaBoostPipeline(ClassificationData trainData, ClassificationData testData);
GestureRecognitionPipeline configurePipeline(GestureRecognitionPipeline pipeline, ClassificationData trainData, ClassificationData testData);

int main(int argc, const char * argv[]) {
    //input
    ClassificationData trainData = getInput();
    
    cout << "Splitting data into training/test split..." << endl;
    ClassificationData testData = trainData.split(80);
    
    //Create a new Gesture Recognition Pipeline using an Adaptive Naive Bayes Classifier
    GestureRecognitionPipeline pipelineANBC = configureANBCPipeline(trainData, testData);
    //Print some stats about the testing
    cout << "ANBC Pipeline Test Accuracy: " << pipelineANBC.getTestAccuracy() << endl;
    
    //Create a new Gesture Recognition Pipeline using an Adaptive Naive Bayes Classifier
    GestureRecognitionPipeline pipelineAdaBoost = configureAdaBoostPipeline(trainData, testData);
    //Print some stats about the testing
    cout << "AdaBoost Pipeline Test Accuracy: " << pipelineAdaBoost.getTestAccuracy() << endl;
    
    return 0;
}


ClassificationData getInput(){
    ClassificationData csvData;
    csvData.load( "classification_data.csv" );
    //csvData.printStats();
    cout << "csv formatted classification data OK\n";
    return csvData;
}

GestureRecognitionPipeline configureANBCPipeline(ClassificationData trainData, ClassificationData testData){
    //Create a new Gesture Recognition Pipeline using an Adaptive Naive Bayes Classifier
    GestureRecognitionPipeline pipelineANBC;
    pipelineANBC.setClassifier( ANBC() );
    pipelineANBC = configurePipeline(pipelineANBC, trainData, testData);

    return pipelineANBC;
}

GestureRecognitionPipeline configureAdaBoostPipeline(ClassificationData trainData, ClassificationData testData){
    //Create a new Gesture Recognition Pipeline using an Adaptive Naive Bayes Classifier
    GestureRecognitionPipeline pipelineAdaBoost;
    pipelineAdaBoost.setClassifier( AdaBoost() );
    pipelineAdaBoost = configurePipeline(pipelineAdaBoost, trainData, testData);
    
    return pipelineAdaBoost;
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
    if( !pipelineConfigured.save( "MockPipeline" ) ){
        cout << "ERROR: Failed to save the pipeline!\n";
        //return EXIT_FAILURE;
    }
    
    //Load the pipeline from a file
    if( !pipelineConfigured.load( "MockPipeline" ) ){
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

