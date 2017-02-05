#!/usr/bin/python

import sys
import pickle
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.cross_validation import train_test_split
from time import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

#Creating a simple first pass at POI identifier with a few hand picked features.
features_list = ['poi','salary','total_stock_value','exercised_stock_options','expenses'] # You will need to use more features

'''
## All features, part of troubleshooting later classifiers

financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances',\
 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',\
 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', \
 'restricted_stock', 'director_fees']

#email addrsss removed as cannot be parsed with featureFormat
email_features = ['to_messages', 'from_poi_to_this_person', \
'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

features_list = ['poi'] + financial_features + email_features  
'''

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
### Task 2: Remove outliers

#Deleting most obvious outlier  - TOTAL
exclude_persons =  ['TOTAL','LOCKHART EUGENE E', 'THE TRAVEL AGENCY IN THE PARK']

for name in data_dict.keys():
    if name in exclude_persons:
        del data_dict[name]
        print 'deleted: ', name
    else:
        pass


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing

##########Messing with featureFormat
data_all = featureFormat(my_dataset, features_list, sort_keys = True,\
remove_NaN=False, remove_all_zeroes=False, remove_any_zeroes=False)
print 'all data: ', len(data_all)
data_noNaN = featureFormat(my_dataset, features_list, sort_keys = True,\
remove_NaN=True, remove_all_zeroes=False, remove_any_zeroes=False)
print 'No NaN data: ', len(data_noNaN)
data_noZero = featureFormat(my_dataset, features_list, sort_keys = True,\
remove_NaN=False, remove_all_zeroes=True, remove_any_zeroes=False)
print 'no 0s data: ', len(data_noZero)
####################actual featureFormat_used
data_noNaNnoZero = featureFormat(my_dataset, features_list, sort_keys = True,\
remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False)
print 'no 0s or NaN data: ', len(data_noNaNnoZero)


count = 0
for name in data_dict.keys():    
    all_zeros = True
    for item in data_dict[name]:      
        if data_dict[name][item] != 0 and data_dict[name][item] != "NaN":
            all_zeros = False
            count += 1
            break
    if all_zeros:
        print name
        #print data_dict[name]
print count

labels, features = targetFeatureSplit(data_noNaNnoZero)

#how many poi in my_dataset?:
print 'my_dataset size = ', len(my_dataset)
#print 'poi = ' ,  np.sum(my_dataset, axis = 0, dtype=np.int8)[0]
feature_count = np.sum(data_noNaNnoZero, axis = 0, dtype=np.int8)[0]
print 'poi = ' ,  feature_count

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
'''
#For like for like comparison with GuassianNB 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis()


# to check later developmets arent skewing witht the data
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()

from sklearn.svm import SVC
clf = SVC()
'''
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

t0 = time()
clf.fit(features_train, labels_train)
fit_time = round((time() - t0), 4)
print "Classifier Fit Time = ", fit_time
t0 = time()
pred = clf.predict(features_test)
pred_time = round((time() - t0), 4)
print "Classifier Predict Time = ",pred_time

#Custom totally pointless metric
def true_positive_p(test, pred):
    label_tuple = zip(pred, test)
    tn = 0.0
    tp = 0.0
    fp = 0.0
    fn = 0.0
    n = len(label_tuple)
    for i in label_tuple:
        if sum(i) == 0:
            tn += 1
        elif sum(i) == 2:
            tp += 1
        else:
            if i[0] == 1:
                fp += 1
            elif i[1] == 1:
                fn += 1
    precision =  tp / (tp + fp) # testing loop above produces correct result
    t_pos = ((tp + tn)/(tp + fp))*(tp/(tp + tn)) #eh, same as above!? At least I learned to apply Bayes rule
    f_pos = fp / (fn + fp) # eh, same as Recall though not identical - unsure if unique.
    return f_pos
    
#Custom scoring system that penalizes very slow classifiers.
#No real mathematical justificaion for this formula, simply arithmetic mean of
#f score and a bothed together speed function. Expodential not chosen for any
#scientific reason, just wanted to return 1 if time = 0!
def Grand_Prix(test, pred, speed_weight = 0.5):
    f1 =  f1_score(test, pred)
    speed = speed_weight/(np.exp(1000*(fit_time + pred_time)))
    result = (f1 + speed)/2
    return result
    
#Classifier Metrics:
print 'Classifier Accurcacy = ', accuracy_score(labels_test, pred)
print 'Classifier Precision = ', precision_score(labels_test, pred)
print 'Classifier Recall = ', recall_score(labels_test, pred)
#print 'Classifier False Positive Rate = ', true_positive_p(labels_test, pred)
print 'Classifier F1 Score = ', f1_score(labels_test, pred)
print 'Speed Weighted F1 Score = ', Grand_Prix(labels_test, pred)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
my_dataset = data_dict
dump_classifier_and_data(clf, my_dataset, features_list)