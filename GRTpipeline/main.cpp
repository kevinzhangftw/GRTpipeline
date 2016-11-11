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
GestureRecognitionPipeline configurePipeline(ClassificationData trainData, ClassificationData testData);

int main(int argc, const char * argv[]) {
    //input
    ClassificationData trainData = getInput();
    
    cout << "Splitting data into training/test split..." << endl;
    ClassificationData testData = trainData.split(80);
    
    //Create a new Gesture Recognition Pipeline using an Adaptive Naive Bayes Classifier
    GestureRecognitionPipeline pipeline = configurePipeline(trainData, testData);
    //Print some stats about the testing
    cout << "Test Accuracy: " << pipeline.getTestAccuracy() << endl;
    
    return 0;
}


ClassificationData getInput(){
    ClassificationData csvData;
    csvData.load( "classification_data.csv" );
    //csvData.printStats();
    cout << "csv formatted classification data OK\n";
    return csvData;
}

GestureRecognitionPipeline configurePipeline(ClassificationData trainData, ClassificationData testData){
    //Create a new Gesture Recognition Pipeline using an Adaptive Naive Bayes Classifier
    GestureRecognitionPipeline pipeline;
    pipeline.setClassifier( ANBC() );

    //Train the pipeline using the training data
    cout << "Training model..." << endl;
    if( !pipeline.train( trainData ) ){
        cout << "ERROR: Failed to train the pipeline!\n";
        //return EXIT_FAILURE;
    }
    
    //Save the pipeline to a file
    if( !pipeline.save( "MockPipeline" ) ){
        cout << "ERROR: Failed to save the pipeline!\n";
        //return EXIT_FAILURE;
    }
    
    //Load the pipeline from a file
    if( !pipeline.load( "MockPipeline" ) ){
        cout << "ERROR: Failed to load the pipeline!\n";
        //return EXIT_FAILURE;
    }
    
    //Test the pipeline using the test data
    cout << "Testing model..." << endl;
    if( !pipeline.test( testData ) ){
        cout << "ERROR: Failed to test the pipeline!\n";
        //return EXIT_FAILURE;
    }
    
    return pipeline;
}

