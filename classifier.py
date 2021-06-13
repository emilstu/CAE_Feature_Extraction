import numpy as np
import math
import os
import matplotlib.pyplot as plt
from utils import util

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif

class Classifier:
    def __init__(self, feature_dir, ffr_dir, ffr_filename, input_dir, ffr_cut_off, folds, iterations):
        self.feature_dir = feature_dir
        self.ffr_dir = ffr_dir
        self.ffr_filename = ffr_filename
        self.input_dir = input_dir
        self.ffr_cut_off = ffr_cut_off
        self.folds = folds
        self.iterations = iterations
        
        # Create out dirs if it doesn't exists 
        if not os.path.exists('evaluation/classification/'):
            os.makedirs('evaluation/classification/')

        # Create out directory
        self.out_dir = util.get_next_folder_name('evaluation/classification/', pattern='ex')
        os.makedirs(self.out_dir)

    def load_data(self):
        features = util.load_features(self.feature_dir)    
        target = util.ffr_values_to_target_list(self.ffr_dir, self.ffr_filename, self.input_dir, self.ffr_cut_off)
        return features, target


    def train(self, fs_type, plot_fs_score, kbest_chi, kbest_mi):
        base_fpr = np.linspace(0, 1, 100)
        features, target  = self.load_data()
        
        if kbest_chi != 'all': kbest_chi = int(kbest_chi)
        if kbest_mi != 'all': kbest_mi = int(kbest_mi)

        print(kbest_chi)
        print(kbest_mi)

        if plot_fs_score: self.plot_statistics(features, target, kbest_chi, kbest_mi)
        
        scores = self.train_classifier(base_fpr, type=fs_type)
   
        ####### PLOTTING OF GP SCORES #########
        gp_tprs=scores.get('gp_tprs')
        gp_aucs=scores.get('gp_aucs')
        gp_mean_tpr = np.mean(scores.get('gp_tprs'), axis=0)

        gp_mean_auc, gp_std_auc = (np.average(gp_aucs), np.std(gp_aucs))

    
        gp_tpr_std_error = np.std(gp_tprs) / math.sqrt(len(gp_tprs))
        gp_auc_std_error = np.std(gp_aucs) / math.sqrt(len(gp_aucs))
        gp_tpr_ci =  1.96 * gp_tpr_std_error
        gp_auc_ci =  1.96 * gp_auc_std_error
        
        gp_tprs_upper = gp_mean_tpr + gp_tpr_ci
        gp_tprs_lower = gp_mean_tpr - gp_tpr_ci

        plt.figure(figsize=(16, 12))
        plt.plot(base_fpr, gp_mean_tpr, color='blue',
            label=r'GPC Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (gp_mean_auc, gp_std_auc),
            lw=2, alpha=.8)
        plt.fill_between(base_fpr, gp_tprs_lower, gp_tprs_upper, color='blue', alpha=.1)


        ####### PLOTTING OF KN SCORES #########
        kn_tprs=scores.get('kn_tprs')
        kn_aucs=scores.get('kn_aucs')
        kn_mean_tpr = np.mean(scores.get('kn_tprs'), axis=0)

        kn_mean_auc, kn_std_auc = (np.average(kn_aucs), np.std(kn_aucs))
        kn_tpr_std_error = np.std(kn_tprs) / math.sqrt(len(kn_tprs))
        kn_auc_std_error = np.std(kn_aucs) / math.sqrt(len(kn_aucs))
        kn_tpr_ci =  1.96 * kn_tpr_std_error
        kn_auc_ci =  1.96 * kn_auc_std_error
        
        kn_tprs_upper = kn_mean_tpr + kn_tpr_ci
        kn_tprs_lower = kn_mean_tpr - kn_tpr_ci

        plt.plot(base_fpr, kn_mean_tpr, color='red',
            label=r'KNN Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (kn_mean_auc, kn_std_auc),
            lw=2, alpha=.8)
        
        plt.fill_between(base_fpr, kn_tprs_lower, kn_tprs_upper, color='red', alpha=.1)
        
        
        ####### PLOTTING OF KN SCORES #########
        if fs_type == 'entire': plt.title('Feature Selection Entire Dataset')
        if fs_type == 'train': plt.title('Feature Selection Train Set ')
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('1 - Specificity (FPR)')
        plt.ylabel('Sensitivity (TPR)')
        #plt.title('ROC')
        plt.legend(loc="lower right")

        # Print scores
        print()
        print('\n\n-- GP SCORES --')
        print('AUC: %.3f (%.3f)' % (gp_mean_auc, gp_std_auc))
        print('SENSITIVITY: %.3f (%.3f)' % (scores.get('gp_sens')[0], scores.get('gp_sens')[1]))
        print('SPECIFICITY: %.3f (%.3f)' % (scores.get('gp_spec')[0], scores.get('gp_spec')[1]))

        print('\n-- KN SCORES --')
        print('AUC: %.3f (%.3f)' % (kn_mean_auc, kn_std_auc))
        print('SENSITIVITY: %.3f (%.3f)' % (scores.get('kn_sens')[0], scores.get('kn_sens')[1]))
        print('SPECIFICITY: %.3f (%.3f)' % (scores.get('kn_spec')[0], scores.get('kn_spec')[1]))
        plt.savefig(f"{self.out_dir}/roc.png")

    def train_classifier(self, base_fpr, type):
        print('\n\nStart training of classifier... \n')
        features, target = self.load_data()

        print('Num of features', features[1].shape)
          
        sel1 = SelectKBest(mutual_info_classif, k='all')
        sel2 = SelectKBest(chi2, k=16)

        gp_total_rec=[]
        gp_total_spec = []
        gp_tprs = []
        gp_aucs = []

        kn_total_rec=[]
        kn_total_spec = []
        kn_tprs = []
        kn_aucs = []

        if type=='entire':
            sel1.fit(features,target)
            features=sel1.transform(features)
            sel2.fit(features,target)
            features=sel2.transform(features)

        for i in range(self.iterations):
            cv_outer = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=i)
            # Under sampling 
            #sm = RandomUnderSampler(sampling_strategy=0.95, random_state=i)
            #re_features, re_target = sm.fit_resample(features, target)
            re_features, re_target = features, target

            for train, test in cv_outer.split(re_features, re_target):
                X_train, X_test = re_features[train], re_features[test]
                y_train, y_test = re_target[train], re_target[test]
      
                if type=='train': 
                    sel1.fit(X_train, y_train)
                    X_train = sel1.transform(X_train)
                    X_test = sel1.transform(X_test)
                    sel2.fit(X_train, y_train)
                    X_train = sel2.transform(X_train)
                    X_test = sel2.transform(X_test)

                # Upsampling of trian set
                #sm = ADASYN(sampling_strategy='minority', random_state=i, n_neighbors=3)
                #sm = SMOTE(random_state=i, k_neighbors=4, sampling_strategy='minority')
                #sm = RandomOverSampler(sampling_strategy='minority', random_state=i)
                #X_train, y_train = sm.fit_resample(X_train, y_train)

                # Scaling
                #scaler = StandardScaler().fit(X_train)
                scaler = RobustScaler().fit(X_train)
                #scaler =  MinMaxScaler().fit(X_train)
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
         
                # Set kernel 
                kernel = 1*DotProduct()
                #kernel = 1*RBF()
                gp_grid = GaussianProcessClassifier(kernel=kernel)
                kn_grid = KNeighborsClassifier(n_neighbors=3)#, weights='distance')
        
                # ADD GP RESULTS
                gp_result = gp_grid.fit(X_train, y_train)
                gp_pred = gp_grid.predict(X_test)    
                #print(result.best_params_)
                print('GP true: ', y_test)
                print('GP pred: ', gp_pred)
                gp_tn, gp_fp, gp_fn, gp_tp = confusion_matrix(y_test, gp_pred).ravel()
                if (gp_tn + gp_fp) != 0:
                    gp_spec = gp_tn / (gp_tn + gp_fp)
                    gp_total_spec.append(gp_spec)
                else:
                    gp_total_spec.append(0)
                
                gp_total_rec.append(metrics.recall_score(y_test, gp_pred))
                gp_probas_ = gp_grid.predict_proba(X_test)

                gp_fpr, gp_tpr, thresholds = metrics.roc_curve(y_test, gp_probas_[:, 1])
                gp_tprs.append(np.interp(base_fpr, gp_fpr, gp_tpr))
                gp_tprs[-1][0] = 0.0
                gp_tprs[0][0] = 0.0
                gp_aucs.append(metrics.auc(gp_fpr, gp_tpr))
                
                # ADD KN RESULTS 
                kn_result = kn_grid.fit(X_train, y_train)
                kn_pred = kn_grid.predict(X_test)    
                kn_tn, kn_fp, kn_fn, kn_tp = confusion_matrix(y_test, kn_pred).ravel()
                if (kn_tn + kn_fp) != 0:
                    kn_spec = kn_tn / (kn_tn + kn_fp)
                    kn_total_spec.append(kn_spec)
                else:
                    kn_total_spec.append(0)
                
                #print(result.best_params_)
                print('KN true: ', y_test)
                print('KN pred: ', kn_pred)
                        
                kn_total_rec.append(metrics.recall_score(y_test, kn_pred))
                
                # Plotting method 1
                #probas_ = best_model.predict_proba(X_test)
                kn_probas_ = kn_grid.predict_proba(X_test)

                kn_fpr, kn_tpr, thresholds = metrics.roc_curve(y_test, kn_probas_[:, 1])
                kn_tprs.append(np.interp(base_fpr, kn_fpr, kn_tpr))
                kn_tprs[-1][0] = 0.0
                kn_tprs[0][0] = 0.0
                kn_aucs.append(metrics.auc(kn_fpr, kn_tpr))       
        
        scores = dict()
        scores['gp_sens']=(np.mean(gp_total_rec), np.std(gp_total_rec))
        scores['gp_spec']=(np.mean(gp_total_spec), np.std(gp_total_spec))
        scores['gp_tprs']=gp_tprs
        scores['gp_aucs']=gp_aucs

        scores['kn_sens']=(np.mean(kn_total_rec), np.std(kn_total_rec))
        scores['kn_spec']=(np.mean(kn_total_spec), np.std(kn_total_spec))
        scores['kn_tprs']=kn_tprs
        scores['kn_aucs']=kn_aucs

        return scores


    def plot_statistics(self, features, target, kbest_chi, kbest_mi):
        check = features.shape[1]
        if check>501: 
            method='m1'
        else:
            method='m2'
        
        sel1 = SelectKBest(mutual_info_classif, k=kbest_mi)
        sel2 = SelectKBest(chi2, k=kbest_chi) 

        print('chi: ', kbest_chi)
        print('mi: ', kbest_mi)
        if method=='m2':
            cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=100)
            i = 0
            os.makedirs(f"{self.out_dir}m2/test/")
            os.makedirs(f"{self.out_dir}m2/train/")
            for train, test in cv_outer.split(features, target):
                X_train, y_train = features[train], target[train]
                X_test, y_test = features[test], target[test]

                X_train = sel1.fit_transform(X_train, y_train)
                X_train = sel2.fit_transform(X_train, y_train)

                mi_values_from_train = sel1.scores_
                chi_values_from_train = sel2.scores_
                
                mi_idx_to_include = sel1.get_support()
                chi_idx_to_include = sel2.get_support() 

                # Get test scores
                X_test = sel1.fit_transform(X_test, y_test)
                X_test = sel2.fit_transform(X_test, y_test)
                
                mi_values_from_test = sel1.scores_
                chi_values_from_test = sel2.scores_
                
               
                mi_values_from_test_used = mi_values_from_test[mi_idx_to_include]
                chi_values_from_test_used = chi_values_from_test[chi_idx_to_include]

                mi_values_from_train_used = mi_values_from_train[mi_idx_to_include]
                chi_values_from_train_used = chi_values_from_train[chi_idx_to_include]
            
                fontSize = 18
                plt.rc('font', size=fontSize)          # controls default text sizes
                plt.rc('axes', titlesize=20)     # fontsize of the axes title
                plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
                plt.rc('legend', fontsize=fontSize)    # legend fontsize
                plt.rc('figure', titlesize=fontSize)  # fontsize of the figure title

                #plt.title("Feature selection based on train-set")
                plt.figure(figsize=(20, 10))
                plt.suptitle("MI and chi-statistics for the test-set", fontsize=36)
                plt.subplot(1, 2, 1)
                plt.hist(mi_values_from_test, bins=10)
                plt.hist(mi_values_from_test_used, bins=10)
                plt.legend(["dist of MI-values from all features",
                            "dist of MI-values from selected features"])
                plt.yscale('log', nonposy='clip')
                plt.xlabel('MI-score')
                plt.ylabel('Frequency')

                plt.subplot(1, 2, 2)
                plt.hist(chi_values_from_test, bins=10)
                plt.hist(chi_values_from_test_used, bins=10)
                plt.xlabel('chi2-score')
                plt.ylabel('Frequency')
                plt.yscale('log', nonposy='clip')
                plt.legend(["dist of chi2-values from MI-reduced subset",
                            "dist of chi2-values from selected features of MI-reduced subset"])
                plt.tight_layout()

                plt.savefig(f"{self.out_dir}m2/test/m2_test_feature-select_{i}.png")
                                
                plt.figure(figsize=(20, 10))
                plt.suptitle("MI and chi-statistics for the train-set", fontsize=36)
                plt.subplot(1, 2, 1)
                plt.hist(mi_values_from_train, bins=10)
                plt.hist(mi_values_from_train_used, bins=10)
                plt.xlabel('MI-score')
                plt.ylabel('Frequency')
                plt.yscale('log', nonposy='clip')
                plt.legend(["dist of MI-values from all features",
                        "dist of MI-values from selected features"])
                
                plt.subplot(1, 2, 2)
                #plt.figure(figsize=(12, 18))
                plt.hist(chi_values_from_train, bins=10)
                plt.hist(chi_values_from_train_used, bins=10)
                plt.xlabel('chi2-score')
                plt.ylabel('Frequency')
                plt.yscale('log', nonposy='clip')
                plt.legend(["dist of chi2-values from MI-reduced subset",
                        "dist of chi2-values from selected features of MI-reduced subset"])
                plt.tight_layout()
                plt.savefig(f"{self.out_dir}m2/train/m2_train_feature-select_{i}.png")
                
                i+=1
        else:
            
            cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=100)
            i = 0
            os.makedirs(f"{self.out_dir}m1/test/")
            os.makedirs(f"{self.out_dir}m1/train/")
            for train, test in cv_outer.split(features, target):
                X_train, y_train = features[train], target[train]
                X_test, y_test = features[test], target[test]

                X_train = sel2.fit_transform(X_train, y_train)
                chi_idx_to_include = sel2.get_support()
                chi_values_from_train = sel2.scores_

                # Get test scores
                X_test = sel2.fit_transform(X_test, y_test)
                chi_values_from_test = sel2.scores_
                
                chi_values_from_test_used = chi_values_from_test[chi_idx_to_include]
                chi_values_from_train_used = chi_values_from_train[chi_idx_to_include]
  
                fontSize = 18
                plt.rc('font', size=fontSize)           # controls default text sizes
                plt.rc('axes', titlesize=20)            # fontsize of the axes title
                plt.rc('axes', labelsize=20)            # fontsize of the x and y labels
                plt.rc('legend', fontsize=18)           # legend fontsize
                plt.rc('figure', titlesize=fontSize)    # fontsize of the figure title

                plt.figure(figsize=(12, 8))
                plt.suptitle("Chi-statistics for the test-set", fontsize=30)
                plt.hist(chi_values_from_test, bins=10)
                plt.hist(chi_values_from_test_used, bins=10)
                plt.xlabel('chi2-score')
                plt.ylabel('Frequency')
                plt.yscale('log', nonposy='clip')
                plt.legend(["dist of chi2-values from all features",
                            "dist of chi2-values from selected features"])
                plt.tight_layout()

                plt.savefig(f"{self.out_dir}m1/test/m1_test_feature-select_{i}.png")
                
                plt.figure(figsize=(12, 8))
                plt.suptitle("Chi-statistics for the train-set", fontsize=30)
                plt.hist(chi_values_from_train, bins=10)
                plt.hist(chi_values_from_train_used, bins=10)
                plt.xlabel('chi2-score')
                plt.ylabel('Frequency')
                plt.yscale('log', nonposy='clip')
                plt.legend(["dist of chi2-values from all features",
                        "dist of chi2-values from selected features"])
                plt.tight_layout()
                plt.savefig(f"{self.out_dir}/m1/train/m1_train_feature-select_{i}.png")
                
                i+=1
            
            
        
