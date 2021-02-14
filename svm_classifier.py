import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from utils import util
import random

class SvmClassifier:
    def __init__(self, target, test_size, feature_dir, out_dir):
        """
        num_iters (int): Number of iterations of kmeans
        num_clusters (int): Number of clusters
        seg_dir (string): Direcorty of segmentations to cluster
        out_dir (string): Where to store the results
        
        """
        self.target = np.array(target)
        self.test_size = test_size
        self.features = util.load_features(feature_dir)
        self.out_dir = out_dir

        # Create out directory for CAE if it doesn't exists 
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)


    def run(self):
        print(self.features.shape)
        print(self.target.shape)
        
        # Split dataset into training set and test set
        f_train, f_test, t_train, t_test = train_test_split(self.features, self.target, test_size=self.test_size)
        
        print(f_train.shape, f_test.shape)
        print(t_train.shape, t_test.shape)

        #Create a svm classifier with radial basis kernel
        clf = svm.SVC(kernel='rbf')

        #Train the model using the training sets
        clf.fit(f_train, t_train)

        #Predict the response for test dataset
        t_pred = clf.predict(f_test) 

        # Model Accuracy: how often is the classifier correct?
        print("Accuracy:",metrics.accuracy_score(t_test, t_pred))

        # Model Precision: what percentage of positive tuples are labeled as such?
        print("Precision:",metrics.precision_score(t_test, t_pred))

        # Model Recall: what percentage of positive tuples are labelled as such?
        print("Recall:",metrics.recall_score(t_test, t_pred))
    