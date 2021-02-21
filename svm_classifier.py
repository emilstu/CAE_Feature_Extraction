import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from utils import util
import random

class SvmClassifier:
    def __init__(self, feature_dir, target_dir, ffr_boundary, test_size):
        self.feature_dir = feature_dir
        self.target_dir = target_dir
        self.ffr_boundary = ffr_boundary
        self.test_size = test_size
        self.clf = svm.SVC(kernel='rbf')
        
        # Create out dirs if it doesn't exists 
        if not os.path.exists('evaluation/classification/svm/'):
            os.makedirs('evaluation/classification/svm/')

        # Create out directory
        self.out_dir = util.get_next_folder_name('evaluation/classification/svm/', pattern='ex')
        os.makedirs(self.out_dir)

     
    def train(self):
        print('\n\nStart training of SVM classifier... \n')
        features = util.load_features(self.feature_dir)    
        target = util.ffr_values_to_target_list(self.target_dir, self.ffr_boundary)
   
        # Split data for training and testing 
        feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=self.test_size,random_state=109) 

        # Train the model using the training sets
        self.clf.fit(feature_train, target_train)

        # Save model
        util.save_svm_model(model=self.clf, model_name='svm', out_dir=self.out_dir)


    def predict(self, model_dir=None):
        print('\n\nStart prediction of SVM classifier... \n')
        features = util.load_features(self.feature_dir)
        target = util.ffr_values_to_target_list(self.target_dir, self.ffr_boundary)

        # Load model if model_dit != None
        if model_dir is not None:
            self.clf = util.load_svm_model(model_name='svm', model_dir=model_dir)

        # Split data for training and testing 
        feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=self.test_size,random_state=109) 

        #Predict the unseen data
        pred = self.clf.predict(feature_test) 

        
        # Model Accuracy: how often is the classifier correct?
        acc = metrics.accuracy_score(target_test, pred)
        
        # Model Precision: what percentage of positive tuples are labeled as such?
        prec = metrics.precision_score(target_test, pred)
        
        # Model Recall: what percentage of positive tuples are labelled as such?
        rec = metrics.recall_score(target_test, pred)

        # Print results
        print("\nAccuracy: ", acc)
        print("Precision: ", prec)
        print("Recall: ", rec, '\n')


        # Save model and results
        util.save_svm_results(accuracy=acc, precision=prec, recall=rec, out_dir=self.out_dir)
    
    