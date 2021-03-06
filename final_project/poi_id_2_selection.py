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

financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances',\
 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',\
 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', \
 'restricted_stock', 'director_fees']

#email addrsss removed as cannot be parsed with featureFormat
email_features = ['to_messages', 'from_poi_to_this_person', \
'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']


#Creating a simple first pass at POI identifier with a few hand picked features.
features_list = ['poi'] + financial_features + email_features # You will need to use more features

dud_features = ['loan_advances']
for f in dud_features:
    features_list.remove(f)

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
 
    
### Task 2: Remove outliers

#Deleting most obvious outlier  - TOTAL
#Post exploration below Eugene Lockhart has no usefull data
exclude_persons = ['TOTAL','LOCKHART EUGENE E', 'THE TRAVEL AGENCY IN THE PARK']

#Loop to remove persions specified above
for name in data_dict.keys():
    if name in exclude_persons:
        del data_dict[name]
        print 'deleted: ', name
    else:
        pass

### Task 3: Create new feature(s)

##New Feature creation:
#Completing before featureFormat to avoit dealing with labeless arrays

#exercised_stock_ratio = Fraction of excercised from total unrestricted stock
#stock available to excersise (not fully sure if deferred restircted stock should alse be subtracted)
#to_poi_email_fraction = Fraction of emails sent that were to  poi
#from_poi_email_fraction = Fraction of email recived that were froma poi
#total_bonus_compensation = Combined bonus, fees, expenses and otehr benifits that are
#not part of base salary. Other may belong in her but unsure.
#salary_fraction_total_bonus = above as fraction of salary (this may be > 1 )
#Most likely a far less messy way of doing this but below currently works without issues
new_features = ['exercised_stock_ratio','to_poi_email_fraction','from_poi_email_fraction', \
                'total_bonus_compensation','salary_fraction_total_bonus']
def feature_adder(data_dict):
    for name in data_dict.keys():
        d = data_dict[name]
        #change NaN to 0 (wish I tried arrays now)
        for ft in d.keys():
            if d[ft] == 'NaN':
                d[ft] = 0.0
        #if statment to prevent div 0
        if (d['total_stock_value'] - d['restricted_stock']) == 0:
            d['exercised_stock_ratio'] = 0.0
        else:
            d['exercised_stock_ratio'] = d['exercised_stock_options'] / (d['total_stock_value'] - d['restricted_stock'] )
        if d['from_messages'] == 0:
            d['to_poi_email_fraction'] = 0.0
        else:
            d['to_poi_email_fraction'] = d['from_this_person_to_poi'] / d['from_messages']
        if  d['to_messages'] == 0:
            d['from_poi_email_fraction'] = 0.0
        else:
            d['from_poi_email_fraction'] = d['from_poi_to_this_person'] / d['to_messages']
        d['total_bonus_compensation'] = d['deferral_payments'] + d['bonus'] + d['director_fees'] \
+ d['expenses'] + d['deferred_income'] + d['long_term_incentive']
        if d['salary'] == 0:
            d['salary_fraction_total_bonus'] = 0.0
        else:
            d['salary_fraction_total_bonus'] = d['total_bonus_compensation'] / d['salary']
       
    return data_dict

#update data_dict as per above and add new features to feature list
data_dict = feature_adder(data_dict)
features_list = features_list + new_features

### Store to my_dataset for easy export below.

my_dataset = data_dict

### Extract features and labels from dataset for local testing

data_noNaNnoZero = featureFormat(my_dataset, features_list, sort_keys = True,\
remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False)
print 'no 0s or NaN data: ', len(data_noNaNnoZero)

my_dataset = data_noNaNnoZero

    
# Check each name in the data_dict for all 0 / NaN values and return count of non-zeros
#returns dict of each data point (person) with at least on non=0 feature and value
#of total non zero features 
def key_checker(data_dict):
    valid_dict = {}
    for name in data_dict.keys():
        valid_count = 0
        all_zeros = True
        for item in data_dict[name]:      
            if data_dict[name][item] != 0 and data_dict[name][item] != "NaN":
                all_zeros = False
                valid_count += 1
        valid_dict[name] = valid_count
    
        if all_zeros:
            print 'all zeros: ',name
            #print data_dict[name]
    return valid_dict

#check each of the features number of non zero results
#returns dict of each feature with total non zero features 
def feature_checker(data_dict, features_list):
    feature_dict = {}
    for feature in features_list:
        non_zero_count = 0
        empty_feature = True
        for name in data_dict:
            if data_dict[name][feature] != 0 and data_dict[name][feature] != "NaN":
                empty_feature = False
                non_zero_count += 1
        feature_dict[feature] = non_zero_count
        if empty_feature:
            print 'empty feature: ', feature
    return feature_dict
key_check_dict = key_checker(data_dict)

print key_check_dict # printing key useful feature counts
print feature_checker(data_dict, features_list), len( feature_checker(data_dict, features_list))

#with only 2 valid featues and a decidedly unhuman name,
#THE TRAVEL AGENCY IN THE PARK is likely not a POI
'''
for name in key_check_dict.keys():
    if key_check_dict[name] <= 10:
        print name, key_check_dict[name]
print feature_checker(data_dict, features_list)
'''
#Not clear what other is but it produced 93 valid results, 
#will see if removing later improves classifier
'''
for name in data_dict.keys():
    print data_dict[name]['other']
'''
#Loan advances returns only 3 data points - 1 poi
#Not likely to be useful predictor - scrubbed.
'''
for name in data_dict.keys():
    print data_dict[name]['loan_advances'], name, data_dict[name]['poi']
'''
    
labels, features = targetFeatureSplit(my_dataset)

#how many poi in my_dataset?:
print 'my_dataset size = ', len(my_dataset)
print 'poi = ' ,  np.sum(my_dataset, axis = 0, dtype=np.int8)[0]
print 'No of features = ', len(features_list)

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
fit_time = round((time() - t0), 4)
print "Classifier Fit Time = ", fit_time
t0 = time()
pred = clf.predict(features_test)
pred_time = round((time() - t0), 4)
print "Classifier Predict Time = ",pred_time

#Custom slightly pointless metric 
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
    #precision =  tp / (tp + fp) # testing loop above produces correct result
    #t_pos = ((tp + tn)/(tp + fp))*(tp/(tp + tn)) #eh, same as above!?
    f_pos = fp / (fn + fp) # eh, same as Recall though not identical - unsure if unique.
    final_total =  tn + tp + fp + fn 
    if final_total != n:
        print 'true_positive_p has gone tits up'
    else:
        return f_pos
    
#Custom scoring system that penalizes very slow classifiers.
#No real mathematical justificaion for this formula, simply arithmetic mean of
#f score and a bothed together speed function. Expodential not chosen for any
#scientific reason, just wanted to return 1 if time = 0!

def Grand_Prix(test, pred,fit_time, speed_weight = 0.5):
    f1 =  f1_score(test, pred)
    #speed = 1 / (1 - fit_time *speed_weight)
    speed = (np.exp(-(fit_time*speed_weight)/100))
    result = f1 * speed
    return result
    
#Classifier Metrics:
print 'Classifier Accurcacy = ', accuracy_score(labels_test, pred)
print 'Classifier Precision = ', precision_score(labels_test, pred)
print 'Classifier Recall = ', recall_score(labels_test, pred)
print 'Classifier False Positive Rate = ', true_positive_p(labels_test, pred)
print 'Classifier F1 Score = ', f1_score(labels_test, pred)
print 'Speed Weighted F1 Score = ', Grand_Prix(labels_test, pred, 0)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)