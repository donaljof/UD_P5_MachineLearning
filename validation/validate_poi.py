#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.tree import DecisionTreeClassifier
from time import time
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score, recall_score


data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = \
train_test_split(features, labels, test_size=0.3, random_state=42)

####classifier init and fit
clf = DecisionTreeClassifier()
t0 = time() 
print 'Fitting CLF'
clf.fit(features_train, labels_train)
print "Fit time:", round(time()-t0, 3), "s"

###test/results
print 'accuracy = ', clf.score(features_test, labels_test)
pred = clf.predict(features_test)
print 'no of POIs identified = ', sum(pred)
print 'total no in test set = ', len(pred)
print 'actual test pois = ', sum(labels_test)
print "precision score = ", precision_score(labels_test, pred)
print "recall score = ", recall_score(labels_test, pred)
#print zip(pred, labels_test)

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
label_tuple = zip(predictions, true_labels)
tp = 0.0
fp = 0.0
tn = 0.0
fn = 0.0
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
print '# True Positives =', tp
print '# False Positives =', fp
print '# True Negatives =', tn
print '# False Negatives =', fn
print 'prescision = ', tp / (tp + fp)
print 'recall = ', tp / (tp + fn)


        


