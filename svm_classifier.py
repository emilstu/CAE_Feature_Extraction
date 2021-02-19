import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from utils import util
import random

class SvmClassifier:
    def __init__(self, train_feature_dir, train_target_dir, pred_feature_dir, pred_target_dir, ffr_boundary):
        self.train_feature_dir = train_feature_dir
        self.train_target_dir = train_target_dir
        self.pred_feature_dir = pred_feature_dir
        self.pred_target_dir = pred_target_dir
        self.ffr_boundary = ffr_boundary
        
        self.clf = svm.SVC(kernel='rbf')
        
        # Create out dirs if it doesn't exists 
        if not os.path.exists('evaluation/classification/svm/'):
            os.makedirs('evaluation/classification/svm/')


        # Create out directory
        self.out_dir = util.get_next_folder_name('evaluation/classification/svm/', pattern='ex')
        os.makedirs(self.out_dir)

     

    def train(self):
        print('\n\nStart training of SVM classifier... \n')
        features = util.load_features(self.train_feature_dir)    
        target = util.ffr_values_to_target_list(self.train_target_dir, self.ffr_boundary)
            
        # Train the model using the training sets
        self.clf.fit(features, target)

        # Save model
        util.save_svm_model(model=self.clf, model_name='svm', out_dir=self.out_dir)


    def predict(self, model_dir=None):
        print('\n\nStart prediction of SVM classifier... \n')
        features = util.load_features(self.pred_feature_dir)
        target = util.ffr_values_to_target_list(self.pred_target_dir, self.ffr_boundary)

        # Load model if model_dit != None
        if model_dir is not None:
            self.clf = util.load_svm_model(model_name='svm', model_dir=model_dir)

        #Predict the unseen data
        pred = self.clf.predict(features) 

        
        # Model Accuracy: how often is the classifier correct?
        acc = metrics.accuracy_score(target, pred)
        
        # Model Precision: what percentage of positive tuples are labeled as such?
        prec = metrics.precision_score(target, pred)
        
        # Model Recall: what percentage of positive tuples are labelled as such?
        rec = metrics.recall_score(target, pred)

        # Print results
        print("\nAccuracy:", acc)
        print("Precision:", prec)
        print("Recall\n:", rec)

        # Save model and results
        util.save_svm_results(accuracy=acc, precision=prec, recall=rec, out_dir=self.out_dir)
    
    