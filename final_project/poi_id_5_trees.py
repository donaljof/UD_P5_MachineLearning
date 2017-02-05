#!/usr/bin/python

import sys
import pickle
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.cross_validation import train_test_split
from time import time
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score,classification_report
from sklearn import decomposition
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
#from sklearn.feature_selection import SelectKBest

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
    
print 'starting size', len(data_dict)
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
            d['exercised_stock_ratio'] = d['exercised_stock_options'] / (d[
            'total_stock_value'] - d['restricted_stock'] )
        if d['from_messages'] == 0:
            d['to_poi_email_fraction'] = 0.0
        else:
            d['to_poi_email_fraction'] = d['from_this_person_to_poi'] / d[
            'from_messages']
        if  d['to_messages'] == 0:
            d['from_poi_email_fraction'] = 0.0
        else:
            d['from_poi_email_fraction'] = d['from_poi_to_this_person'] / d[
            'to_messages']
        d['total_bonus_compensation'] = d['deferral_payments'] + d[
        'bonus'] + d['director_fees'] + d['expenses'] + d['deferred_income'] + d[
        'long_term_incentive']
        if d['salary'] == 0:
            d['salary_fraction_total_bonus'] = 0.0
        else:
            d['salary_fraction_total_bonus'] = d['total_bonus_compensation'] / d[
            'salary']
    
    return data_dict

#update data_dict as per above and add new features to feature list
data_dict = feature_adder(data_dict)
features_list = features_list + new_features

### Store to my_dataset for easy export below.

my_dataset = data_dict
print 'my_dataset size = ', len(my_dataset)
### Extract features and labels from dataset for local testing

data_noNaNnoZero = featureFormat(my_dataset, features_list, sort_keys = True,\
remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False)


labels, features = targetFeatureSplit(data_noNaNnoZero)

#how many poi in my_dataset?:
print 'my_dataset size = ', len(my_dataset)
feature_count = np.sum(data_noNaNnoZero, axis = 0, dtype=np.int8)[0]
print 'poi = ' ,  feature_count

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Example starting point. Try investigating other evaluation techniques!
# Moving aboce PCA to fit.
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3,random_state=4)

#Feature reduction step of pipeline
#Start with example PCA from eigenfaces
pca = decomposition.PCA(random_state=44)



#scaling data using statndardization scaler
skl = MinMaxScaler()

sss = StratifiedShuffleSplit(random_state=44, test_size=0.3)
# Provided to give you a starting point. Try a variety of classifiers.

#using Support Vector Machines
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=44)
'''
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
'''   ##Checking grid PCA etc vs oob##
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# pipe parameters 
n_components = [2,4,6, 10, 13,15,19]

citerion = ["gini", "entropy"]
#randomforestclassifier__criterion=citerion
n_estimators  = [5,10, 15, 20, 30, 40]
#randomforestclassifier__n_estimators=n_estimators
m_ft = [0.01,0.03,0.04,0.08,0.1,0.15,0.23, 0.4]
#randomforestclassifier__max_features=m_ft
m_split = [2,3,4,6,10, 20]
#randomforestclassifier__max_features=m_ft
leaf_size = [1,2,4,6]
#randomforestclassifier__min_samples_leaf=leaf_size
max_depth = [5,10,None]
#randomforestclassifier__max_depth=max_depth

#### Pipeline:
t0 = time()
pipe = make_pipeline(skl,pca,clf)


estimator = GridSearchCV(pipe, 
                         dict(pca__n_components=n_components,
                              randomforestclassifier__n_estimators=n_estimators,
                              randomforestclassifier__min_samples_leaf=leaf_size,
                              randomforestclassifier__max_features=m_ft,
                              randomforestclassifier__max_depth=max_depth,
                              randomforestclassifier__criterion=citerion),
                              cv = sss, scoring = 'f1')

#Fitting the grid search estimator
estimator.fit(features, labels)

print 'Best Est = ', estimator.best_estimator_
#assigning to clf for tester.py

pipe_time = round((time() - t0), 4)
print "\n Total Pipeline Time = ", pipe_time , "\n"

print "Best GridSearch F1 Score = ", estimator.best_score_

#clf = estimator.best_estimator_
clf = RandomForestClassifier(random_state=4)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

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
print 'Classifier False Positive Prob = ', true_positive_p(labels_test, pred)
print 'Classifier F1 Score = ', f1_score(labels_test, pred)
print 'Speed Weighted F1 Score = ', Grand_Prix(labels_test, pred, pipe_time)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)