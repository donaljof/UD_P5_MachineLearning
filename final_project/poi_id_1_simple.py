#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.cross_validation import train_test_split
from time import time
from sklearn.metrics import accuracy_score


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

#Creating a simple first pass at POI identifier with a few hand picked features.
features_list = ['poi','salary','total_stock_value','exercised_stock_options','expenses'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
### Task 2: Remove outliers

#Deleting most obvious outlier  - TOTAL
exclude_persons = ['TOTAL']

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
data_all = featureFormat(my_dataset, features_list, sort_keys = True,\
remove_NaN=False, remove_all_zeroes=False, remove_any_zeroes=False)
print 'all data: ', len(data_all)

data_noNaN = featureFormat(my_dataset, features_list, sort_keys = True,\
remove_NaN=True, remove_all_zeroes=False, remove_any_zeroes=False)
print 'No NaN data: ', len(data_noNaN)
data_noZero = featureFormat(my_dataset, features_list, sort_keys = True,\
remove_NaN=False, remove_all_zeroes=True, remove_any_zeroes=False)
print 'no 0s data: ', len(data_noZero)
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

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

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
print "Classifier Fit Time =  % secs" % (time() - t0)
t0 = time()
pred = clf.predict(features_test)
print "Classifier Predict Time =  % secs" % (time() - t0)

print 'Classifier Accurcacy = ', accuracy_score(labels_test, pred)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)