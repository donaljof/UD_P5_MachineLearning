# UD_P5_MachineLearning
Machine Learning Mini Project and Final Project

---
---

# Introduction
---

In August 23rd 2000 Enron stock hit a record high of $90 a share. 16 months later, on December 2nd 2001, [the company declared Chapter 11 bankruptcy with the loss of over 4000 jobs](http://news.bbc.co.uk/1/hi/business/1759599.stm). 
This classic case of corporate fraud is used as the basis Project 5 in the Udacity Data Science Nanodegree - Intro to Machine learning.

The purpose of the project is to identify persons of interest (POI's) within the Enron email corpus ([May 7th 2015 version used](https://www.cs.cmu.edu/~./enron/) ) combined with publicly avialable pay and share data. 
For each of 145 the manually identified POI's/non-POI's a set of descriptive features is associated with the person giving either financial or email information on the person in question.
Using this financial and email information the aim is to create a classifer capable of correctly identifying persons of interest in this data set that can be applied across the full enron corpus.

In this project a machine learning classifer implemented using the Python package [Scikit-Learn](http://scikit-learn.org/) will be developed and a machine learning classifier using Linear Discriminant Analysis deployed. The use of machine learning in this task allows a classification problem 
such as this to be completed programmatically with far less human input than manual idenification  of POIs and in a fraction of the time. A variety of supervised machine learning classifiers
have been applied including Nieve Bayes, Support Vector Machines, Linear Disciminant Analysis and Ensenmble methods.
Naive Bayes was selected for its simplicity as an initial classifer and to get idea of an out of the box performance.
Support vector machines classifier was used as it is a popular and well understood alghorith that works well with binary classification problems though the high feature count of this data set and low 
number of POIs proved to be an issue. SVM does not work well with heavly skewed classification populations and this proved a problem.
Random Forest classifer was used to take advantage of the high number of possible features in the dataset and to leverage the power of an ensemble method.
Linear Discriminant Analysis classifer was used with shrinkage applied to deal with the low training points relative to possible features.

The data set provided contains 143 manually identified persons, if they are a person of interest (POI) or not (represente by a 1 or 0) and a selection of financial and emsil features shown below.

financial features: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']

email features: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

There are a total of 146 data points of which 18 are identified as POI's and the rest are not.

The main file for this project located in this repositry is poi_id.py  located in the final_project directory, this is the completed version of the classifier. 
Multiple steps in the learning process and alternative classifiers have been saved in the format poi_id_X.py to record attempted classifiers (and revert in event of dead end).


# Data Features and Selection
---

Initially no feature selection was completed and all possible features were included in the dataset to explore which may be usefull or need to be discarded. Running the code on a 
simple classifer generated a ValueError due to emails stored as string not convertable into floats as part of featureFormat. As the email address is unlikely to be valuable in identifying
POI's this feature was excluded for simplicity.

For a simple check of the features provided the a feature_checker function of the poi_id_2_selection.py file was defined to loop through all features and count the number of non zero / non NaN item. 
The feature 'loan_advances' provides 3 useful results with 2 non-poi. Based on this the Loan Advances feature is unlikey to be a useful at identifying POI's and was removed.

A number of new features were created, mostly to represent payments of email counts as fractions of total and therefore scale the features more appropriately. For example two employees
that excercise $100,000 worth of their shares might apprear similar in the 'exercised_stock_options' feature but if one employee had $200,000 total stock and the other had $4,000,000 in stock these 
sales of shares would not seem as equivalent. Creating ratio features for some of key financial features of each data point reduces impact of outlier and introduces a degree of scaling to the featues.

List of added features:
'exercised_stock_ratio' = Fraction of excercised from total unrestricted stock available to excersise (not fully sure if deferred restircted stock should also be subtracted from this as not clear about the differnces between these)
'to_poi_email_fraction' = Fraction of emails sent that were to  POI.
'from_poi_email_fraction' = Fraction of email recived that were from a POI.
'total_bonus_compensation' = Combined bonus, fees, expenses and otehr benifits that are not part of base salary. Other may belong in her but unsure.
'salary_fraction_total_bonus' = Above metric as fraction of salary (this may be > 1 ).

For the final classifer, feature selection was completed using SelectKBest with k = 4. The optimal feature number was determined by trial and error 

By looping through the K best F scores it was seen that the best feature (by this metric) is "exercised_stock_options". 

Seleck K Best Scorec (ANOVA F - values)

|Features|F score|
| -------------------------- | :-------: |
| salary|    19.7903943832|
| deferral_payments          |    1.04233627397 |
| total_payments             |    9.64411359811 |
| bonus                      |    10.6430154732 |
| total_stock_value           |   25.5121582709 |
| expenses                    |   7.72689807011 |
| exercised_stock_options     |   25.9894214301 |
| other                        |  7.94119255974 |
| long_term_incentive         |   17.2253598825 |
| restricted_stock            |   10.103322706 |
| director_fees               |   1.65726925903 |
| to_messages                 |   0.130241166893 |
| from_poi_to_this_person     |   0.0939291393021 |
| from_messages               |   0.0220145818701 | 
| from_this_person_to_poi     |   1.94142663375 |
| shared_receipt_with_poi    |    2.35776785893 |
| exercised_stock_ratio     |     0.0263590242677 |
| to_poi_email_fraction    |      0.135172413793 |
| from_poi_email_fraction      |  nan |
| total_bonus_compensation     |  8.07329192324 |
| salary_fraction_total_bonus |   0.0241660636938 |

Post analysis above 'from_poi_email_fraction' was removed from the features as it is not operating as expected and appears to return the same value every time.

Though unused in the final classifer dimensionality reduction was completed using PCA (Principle Component Analysis). A breakdown of the top components calculated by PCA was printed out
from the data set based on their explained variance:

1. explained var =   0.871364222652 
2. explained var =   0.0957615880701 
3. explained var =   0.0126077469676 
4. explained var =   0.0108225652873
5. explained var =   0.00478163091311 

From this we can see the first component accounts for the vast majority of variance in the dataset post transformation and the number of potentially useful dimensions reduced to more managable number.

Sclaing of features was attempted using both Standardization and MinMaxScaling but there was either no performance improvement of a decrease in performance.

# Outlier Removal
---

From the Feature Selection module of the Udacity Machine Learning course an initial outlier was identified as 'TOTALS' and removed from the dataset. Further investigation using the key_checker function defined in poi_id_2_selection.py 
found a name 'LOCKHART EUGENE E' had no features with a non-0 or non-NaN result and so was scrubbed. 

Looking at the number of non-0/NaN features returned for each data point
showed a 'THE TRAVEL AGENCY IN THE PARK' with only 3 useful features and a name that suggests it is not a person, let a alone a POI. This data point was removed as part of the initial data checking.

This initial outlier scrub improved the F1 score of the an out of the box Guassian Naive Bayes classifer from 0.09 (subset of features used as per poi_id_1_simple) to 0.57 with the majority of
this benifit coming from the removal of 'TOTALS'


# Classifiers
---
### Naive Bayes

As a starting point the out of the box Gaussian Naive Bayes using a manually selected feature set. The results of the tester.py script show a high precision but the classifier is let down by the recall score of 0.28 suggesting Naive Bayes struggling to correctly locate all the POI's in the data set.

```
features_list = ['poi','salary','total_stock_value','exercised_stock_options','expenses']

Classifier Fit Time =  0.0
Classifier Predict Time =  0.0
Classifier Accurcacy =  0.857142857143
Classifier Precision =  0.5
Classifier Recall =  0.666666666667
Classifier False Positive Rate =  0.666666666667
Classifier F1 Score =  0.571428571429
Speed Weighted F1 Score =  0.535714285714

GaussianNB(priors=None)
        Accuracy: 0.85800       Precision: 0.50535      Recall: 0.28350 F1: 0.36323     F2: 0.31079
        Total predictions: 14000        True positives:  567    False positives:  555   False negatives: 1433   True negatives: 11445
```

### Support Vector Machines

Support Vector Machines were selected intially for their usefullness in dealing with high dimensionality data, speed in their linear form and low complexity relative to ensemble methods.
It was hoped the large number of numeric features would suit the plane splitting strategy employed by SVM and yield a useful prediction. 

Linear support vector machines alghorith was applied using a pipline containing the PCA and tuned over the number of components, C values, tolerence values, maximum iterations and intercept scaling.

Achieving a best case F1 score as per the output below proved challenging from a tuning point of view. Large numbers of components or tuning parameters outside a narrow range gave recall values of less than 0.1.

A persistant problem with this classifier was underidentification with many of the attempeted classifers giving false positive probabilites of > 90% (see classifer metrics for details of how this is calcualted).
Even for the best case classifier detailed below the probability that a given POI is incorrectly identified is 78%. This not practical for the stated purpose of correctly identifying POIs though the high level of precision for 
identifying non-POIs would ensure few POI's are omitted from the final output, at the expense of many false positives. 

Multiple attempts were made to create a SVM classifier using the RBF kernal but no working version could be found that did not identify all data points as Non-POIs. Using the poly kernal generated a F1 value of 0.11 with the majority of this poor score due to incorrectly identified POI's. 
The difficulties in tuning this algorithm and the poor results achieved resulted in it being abandoned for futher explorations. Intuition would suggest that removing PCA from the pipeline would reduce the efffectivness of SVM due to the increased number of hyperplanes divisions needed but changing the 
method of feature reduction may be useful. The scaling of the features used may also have been an issue though PCA and the creation of ratio based features should have alleviated this. 

```
Best Est =  Pipeline(steps=[('pca', PCA(copy=True, n_components=3, whiten=False)), ('linearsvc', LinearSVC(C=50000, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=10000,
     multi_class='ovr', penalty='l2', random_state=None, tol=8e-05,
     verbose=0))])

 Total Pipeline Time =  44.46 
 
             precision    recall  f1-score   support

    Non-POI       0.93      0.74      0.82        38
        POI       0.23      0.60      0.33         5

avg / total       0.85      0.72      0.77        43

Classifier Accurcacy =  0.720930232558
Classifier Precision =  0.230769230769
Classifier Recall =  0.6
Classifier False Positive Prob =  0.833333333333
Classifier F1 Score =  0.333333333333
Speed Weighted F1 Score =  0.333333333333
```

### Decision Trees



Out of Box:
```
Classifier Accurcacy =  0.906976744186
Classifier Precision =  1.0
Classifier Recall =  0.2
Classifier False Positive Prob =  0.0
Classifier F1 Score =  0.333333333333
Speed Weighted F1 Score =  0.333333333333
```
With PCA and GridselctCV looping over possible values for 'n_components', 'n_estimators', 'max_features', 'min_samples_leaf' and 'max_depth'. Unlike previous classifers above, the mistake in the
Speed Weighted F1 Score was fixed and it is giving a reduced F1 score for this slower alghorith (subjective measure but sclaled to penalize when the metric creator is getting impatient).

Annoyingly, due to the random nature of the Random Forest Classifier the high score below is not repeatabe. Running tester code on below produced an poor F1 score of 0.22. 
Either the variation in the features selected by random forest created "lucky" classifer or its possible the series of POI's selected by the classifer fitted to correct results by pure

```
Best Est =  Pipeline(steps=[('pca', PCA(copy=True, n_components=15, whiten=False)), ('randomforestclassifier', RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=5, max_features=0.08, max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False))])

 Total Pipeline Time =  95.998 

             precision    recall  f1-score   support

    Non-POI       0.95      1.00      0.97        38
        POI       1.00      0.60      0.75         5

avg / total       0.96      0.95      0.95        43

Classifier Accurcacy =  0.953488372093
Classifier Precision =  1.0
Classifier Recall =  0.6
Classifier False Positive Prob =  0.0
Classifier F1 Score =  0.75
Speed Weighted F1 Score =  0.464092184753
```

After some frustration with the mediocre effectivness of the random forest classifiers (and on suggestion from the forums), the scoring parameter was uptated to 'f1' and 
StratifiedShuffleSplit used as cross validation.

Running an out of box Random Forest Classifier, searching over PCA n_components gives a F1 score of 0.36, an improvement to the origional OOB result of 0.33.
```
Best Est =  Pipeline(steps=[('pca', PCA(copy=True, iterated_power='auto', n_components=10, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('randomforestclassifier', RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_node...stimators=10, n_jobs=1, oob_score=False, random_state=41,
            verbose=0, warm_start=False))])

 Total Pipeline Time =  2.302 

             precision    recall  f1-score   support

    Non-POI       0.89      0.92      0.91        37
        POI       0.40      0.33      0.36         6

avg / total       0.83      0.84      0.83        43

Classifier Accurcacy =  0.837209302326
Classifier Precision =  0.4
Classifier Recall =  0.333333333333
Classifier False Positive Prob =  0.428571428571
Classifier F1 Score =  0.363636363636
Speed Weighted F1 Score =  0.359474904232 
```
To further boost the preformace of the classifier, skaling was introduced using a MinMax scaler with default settings. The reasoning being that some of the financial components were producing
incorrect splitting of the dataset as part of the tree generation and causing extremem overfitting along some of the decision trees used.
A wide search of possible tunign parameters was used in an effort to seek out the optimum parameters for the classifer though this resulted in a huge runtime.
```
citerion = ["gini", "entropy"]
n_estimators  = [5,10, 15, 20, 30, 40]
max_features = [0.01,0.03,0.04,0.08,0.1,0.15,0.23, 0.4]
max_split = [2,3,4,6,10, 20]
min_samples_leaf = [1,2,4,6]
max_depth = [5,10,None]
```
The best estimate classifier produced a F1 score of 0.38 using the stratified shuffle split data in grid search and an even less impressive 0.25 when fitted sperately to a train test split data that was 
used to calculate the custom metric produced.
```
Best Est =  Pipeline(steps=[('minmaxscaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('pca', PCA(copy=True, iterated_power='auto', n_components=19, random_state=44,
  svd_solver='auto', tol=0.0, whiten=False)), ('randomforestclassifier', RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',...stimators=20, n_jobs=1, oob_score=False, random_state=44,
            verbose=0, warm_start=False))])

 Total Pipeline Time =  7131.968 

Best GridSearch F1 Score =  0.376868686869
Classifier Accurcacy =  0.860465116279
Classifier Precision =  0.5
Classifier Recall =  0.166666666667
Classifier False Positive Prob =  0.166666666667
Classifier F1 Score =  0.25
Speed Weighted F1 Score =  8.14832365651e-17
```
Running tester.py resulted in a abismal 0.18 and failled to meet the passing criterion.
```
Accuracy: 0.85747       Precision: 0.38725      Recall: 0.11850 F1: 0.18147     F2: 0.13760
        Total predictions: 15000        True positives:  237    False positives:  375   False negatives: 1763   True negatives: 12625
```
Again a more complex classifier has come nowhere near the results achieved by an OOB Naive Bayes classifer despite extensive grid search, scaling, PCA and more intensive cross validation.
At this point, given there has been no significant increase in classifer performance I decided to abandon Random Forest Classifers as a possible final classifer.

Looking at the results so far, reading the documentation and trying to understand what is going wrong there are several possibilities:

* The relative small size of the training dataset coupled with the use of GridSearchCV is resulting in overfit classifers or classifiers that very poorly optimized for data beyond the test set. If the classifer tuning
was completed manually its possible a more accurate classifer could be found. Against this, the overfitting thoery in particular, is the low F1 scores even with the test set included.

* Too many features are being included upfront and some features are being included that are screwing up all subsequent analysis or one of the custom created features is toxic to all 
subsequent analysis for some reason. Testing the classifiers with a manually chosen subset of the features also has produced poor results but its possible the manually chosen feature set was also bad.

* PCA is a poor choice for this data set due to the high dimension to data point ratio and the generation of reduced dimensionatily data is producing outputs with less predictive power than the origional data.
Its also possible it is being implemented incorrecty (needs tuning) or shouldnt be fed into the types of classifers used so far (usefule elsewhere but not here).

*  Adding a scaler is not compatable with PCA and has messed up the data in some way just as a optimal implementation of Random Forests was being reached. Not likely as the results have
been poor all the way along and my understanding is scaling should help with strongly dispersed features.

* SVM and Random Forests are, in fact, terrible choices for classification of this type of problem and particularly ill-suited to this data set.

* There is some other problem that I have not identified and / or I have made an complete balls of this somehow and some fundemental error in the code is poisoning all results.

### Linear Discriminant Analysis

In the above section, I outlined several possible reasons for poor classifer perfomance. To try to segment the possibility that the base classifier chosen is the problem and not some other data 
transformation or tuning is responsible, all 3 were tested using the same code for everthing but assigning the classifier (clf = ...). The base out of box parameters were unchanged and the tester.py 
script ran on each of the resultant classifers. 

In addition to the classifers attempted already, a new classifer was added - Linear Discriminant Analysis. [This classifier was chosen as result of some searching for classifers that work
well with small datasets and a larger number of features](http://stats.stackexchange.com/questions/63565/good-classifiers-for-small-training-sets). More detailed investigation of the use of Discriminant Analysis and reading the sklearn documentation suggests it works well with
multicollinearity, requires limited tuning and works under simlilar mathematical principles to the best classifer so far, Guassian Naive Bayes (based on Bayes rule and assumes multivariate normal distribution). 
```
GaussianNB(priors=None)
        Accuracy: 0.85800       Precision: 0.50535      Recall: 0.28350 F1: 0.36323     F2: 0.31079
        Total predictions: 14000        True positives:  567    False positives:  555   False negatives: 1433   True negatives: 11445
		
LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
              solver='svd', store_covariance=False, tol=0.0001)
        Accuracy: 0.86550       Precision: 0.57687      Recall: 0.21950 F1: 0.31800     F2: 0.25054
        Total predictions: 14000        True positives:  439    False positives:  322   False negatives: 1561   True negatives: 11678
		
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
            verbose=0, warm_start=False)
        Accuracy: 0.84529       Precision: 0.41354      Recall: 0.19850 F1: 0.26824     F2: 0.22154
        Total predictions: 14000        True positives:  397    False positives:  563   False negatives: 1603   True negatives: 11437
```
Got a divide by zero when trying out: 
```
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
Precision or recall may be undefined due to a lack of true positive predicitons.
```
From the results above the performace of LDA is similar to Naive Bayes with better precision but an overall lower F1 score. 
The Random Forest is not as effective but it can be noted
this is a better result than the 2 hour PCA/GridSearchCV attempt in the last section. This suggests there was issues with feature selection/reduction or tuning of the Random Forests used previously though
the starting point for the classifer is low already.
For the SVC classifer the produced an error message due to it not predicting any POI's. The starting point for this classifer is basically 0 for this problem though it may be possible to create a functioning and powerful classifer 
with the correct tuning. Given the multitude of other options, it is far easier to start with something that gives a decent OOB result to start with and work from there.

The analyis above and the deeper thinking about the classifer choice is some thing that really should have been completed upfront and not halfway through the project. A key learnign from
this would be to start simple and work towards increased complexity / tuning for each classifer. In the rush to understand and apply different techniques such as PCA, scaling, pipelines and GridSearchCV too much
was applied too quickly. Each new layer needs to be applied with a keen eye on what it does to the classifier performance and some deep thinking as to the impact on what is happening to the dataset.
The approach with LDA will be to apply new features and more advanced techniques incrementally, with careful anaysis of the impact at each step and consideration of the background impact.

Using a manully selected features set and performing no pre processing of data the starting point for the untuned classifier can be obtained. The plan will beto incrementally add 
features, select tuning and layers of preprocessing with analysis of its impact at each step.

features_list = ['poi','salary','total_stock_value','exercised_stock_options','expenses']

 * Starting point                         - Accuracy: 0.86550       Precision: 0.57687      Recall: 0.21950 F1: 0.31800     F2: 0.25054


Fist step in tuning this alghorithm is to choose a solver type of the three available, singalar value decomposition being the default.

 * Least Squares                          - Accuracy: 0.86493       Precision: 0.57087      Recall: 0.21950 F1: 0.31708     F2: 0.25031

 * Eigenvalue Decomposition               - Accuracy: 0.78743       Precision: 0.30354      Recall: 0.37700 F1: 0.33631     F2: 0.35960

From above it can be seen there is a sigificant gain in recall when using the eigen solver. In addition the least squares and eigenvalue solvers support the shrinkage 
option (automatic shinkage applied in both cases):

 * Least Square with Shrinkage            - Accuracy: 0.86636       Precision: 0.58238      Recall: 0.22800 F1: 0.32770     F2: 0.25959

 * Eigenvalue Decomposition with shinkage - Accuracy: 0.82271       Precision: 0.36491      Recall: 0.32550 F1: 0.34408     F2: 0.33269

From above result a small improvement in F1 score is gained with the inclusion of shrinkage. As the purpose of shrinkage is to deal with high dimesionality its impact might best be oberved with a 
much larger set of features.

For the following runs all of available features and all custom features were included. Some features with negaive numbers were removed as they resulted in errors with the solver 
set to Eigenvalue Decomposition, its not clear why the classifer would be inable to handle negative values or why eigen solver choke on these features when other classifiers are able
to handle them. The answer to this likely in the linear algebra used to estimate the maximum conditional probability but this is beyond the scope of this project.

 * Least Squares                          - Accuracy: 0.83427       Precision: 0.29372      Recall: 0.17300 F1: 0.21775     F2: 0.18849
 * Least Square with Shrinkage            - Accuracy: 0.85233       Precision: 0.40000      Recall: 0.21500 F1: 0.27967     F2: 0.23691

 * Eigenvalue Decomposition               - Accuracy: 0.77527       Precision: 0.24621      Recall: 0.33250 F1: 0.28292     F2: 0.31072
 * Eigenvalue Decomposition with Shinkage - Accuracy: 0.78160       Precision: 0.27614      Recall: 0.39350 F1: 0.32454     F2: 0.36267

 * Singular Value Decomposition           - Accuracy: 0.83500       Precision: 0.29649      Recall: 0.17300 F1: 0.21850     F2: 0.18872
 
The LDA classifer performance drops when an overabundance of features are included but the impact of shrinkage is much more significant, especially for the leat squares solver. The
Its clear from this test that inclusion of too many features, even with ways of dealing with high dimenstionality. The standard SVD solver shows a similar reduction in performance with large numbers
of features included but it lacks a shinkage option.

Finally changing to to the quadratic version of LDA is examined as sklearn documentation suggests it should perform better in data sets with varying degreees of feature covariance. Given the wide
selection of data and data types in the email corpus data there may be an additional perfomance boost by using this form.

 * QDA (manually selected features)       - Accuracy: 0.84457       Precision: 0.41635      Recall: 0.21900 F1: 0.28702     F2: 0.24194
 * QDA (all features)                     - Accuracy: 0.83627       Precision: 0.20466      Recall: 0.07900 F1: 0.11400     F2: 0.09006
 
Quadratic linear analyisis does not perform as well os LDA on the reduced feature set and is falls down compeltly when the full set is used. This may be worth revisiting with correct 
preprocessing in place but for now this route appears a dead end.

Next step in the evolution of this classifier is increasing the number of features and by extension selecting them. Before looking at transforming the data through PCA and since the LDA classifer
is already has build in ways to deal with high dimensionality, the select K means method is used as a starting point.
Using LDA with the Eigenvalue solver, shrinkage and the manually selected feature set (miniset) or the full set (full) multiple k values were supplied.

 * LDA(eigen, auto shink, miniset, kbest = 1) - Accuracy: 0.79621       Precision: 0.28514      Recall: 0.28300 F1: 0.28407     F2: 0.28343
 * LDA(eigen, auto shink, miniset, kbest = 4) - Accuracy: 0.82271       Precision: 0.36491      Recall: 0.32550 F1: 0.34408     F2: 0.33269
 
With the manually selected feature set and select k best applied there is a small dissimprovement when selecting only 1 feature suggesting all 4 of the manaually selected features are meaningful (no one feature is dominating compeltly).
With all 4 selected the resultant F1 score is the same, which makes sense.
 
 * LDA(eigen, auto, full, kbest = 1)          - Accuracy: 0.80907       Precision: 0.28927      Recall: 0.29650 F1: 0.29284     F2: 0.29502
 * LDA(eigen, auto, full, kbest = 4)          - Accuracy: 0.83367       Precision: 0.38265      Recall: 0.40350 F1: 0.39280     F2: 0.39915
 * LDA(eigen, auto, full, kbest = 5)          - Accuracy: 0.81767       Precision: 0.33409      Recall: 0.37000 F1: 0.35113     F2: 0.36221
 * LDA(eigen, auto, full, kbest = 10)         - Accuracy: 0.74933       Precision: 0.25824      Recall: 0.47000 F1: 0.33333     F2: 0.40378
 * LDA(eigen, auto, full, kbest = 15)         - Accuracy: 0.77173       Precision: 0.27136      Recall: 0.42250 F1: 0.33047     F2: 0.38015
 * LDA(eigen, auto, full, kbest = 20)         - Accuracy: 0.73340       Precision: 0.25303      Recall: 0.51200 F1: 0.33868     F2: 0.42500
 
By trying multiple k numbers the impact of increased features can be seen with a initial increase in F1 score followed by a reduction as a result of reduced precision. Its possible as that more features are added to the classifer the decision boundries 
become more complex and tend towards overfitting to the data resulting in an increase in false positives. The best possible K value was found by running with multiple k values and slecting the best result (k = 4).

As well as feature selection, scaling the features is another preprocessing step to be considered. This is especially important for classifers such as LDA taht are expectign normally distributed
data as an imput. For this reason a standard scaler was introduced to give a more normal traning dataset for the classifer to fit on.

 * LDA(singular, manual features)          - Accuracy: 0.86550       Precision: 0.57687      Recall: 0.21950 F1: 0.31800     F2: 0.25054
 * LDA(lsqr, auto, manual features)        - Accuracy: 0.86636       Precision: 0.58238      Recall: 0.22800 F1: 0.32770     F2: 0.25959
 * LDA(eigen, auto, manual features)       - Accuracy: 0.86643       Precision: 0.58145      Recall: 0.23200 F1: 0.33167     F2: 0.26370

 With manually selected features and with a eigen value / auto shinkage LDA classifer there is actually a small decrease in the F1 score for the eigen and least squares solvers and no change for the default 
singular value decompostion. Its possible the shinkage feature is not compatable with standardizationa dn too much transformation of the data causes information loss tha hurts classifer performance

 * LDA(lsqr, no shink, manual features)        - Accuracy: 0.86493       Precision: 0.57087      Recall: 0.21950 F1: 0.31708     F2: 0.25031
 * LDA(eigen, no shink, manual features)       - Accuracy: 0.87214       Precision: 0.65957      Recall: 0.21700 F1: 0.32656     F2: 0.25064
 
 Removing shinkage doesn't help and reduces the F1 score further. Its possible standardization is not the ideal scalar for this, another simple option is a MinMax scaler.
 
 * LDA(lsqr, auto, manual features)        - Accuracy: 0.86636       Precision: 0.58238      Recall: 0.22800 F1: 0.32770     F2: 0.25959
 
 For the eigen solver, using a MinMaxScaler threw an error for predicting no POI's and for least squres it returns the same values as standardization. This does not look like a preprocessing
 route worth pursuing.
 
Before abandoning standardization, the full data set was used with K best selection = 4 (tried several otehr but 4 is still optimum), eigen solver, shinkage and standardization.

 * LDA(eigen, auto, full, kbest = 4)       -Accuracy: 0.85173       Precision: 0.39910      Recall: 0.22150 F1: 0.28489     F2: 0.24314 
 
 The results were below the best F1 score obtained without standardization (0.39280) and allow the conclusion that standardization is not a benifit for this classifier.
 
Another attempeted preprocessing was principle component anlaysis, again using the full feature set and eigen/shinkage LDA. No other feature selection was completed but the number of 
transformed components was varied.
 * LDA(eigen, auto, full, PCA components = 1)  - Accuracy: 0.83547       Precision: 0.35375      Recall: 0.28300 F1: 0.31444     F2: 0.29479
 * LDA(eigen, auto, full, PCA components = 4)  - Accuracy: 0.84553       Precision: 0.38924      Recall: 0.27850 F1: 0.32469     F2: 0.29530
 * LDA(eigen, auto, full, PCA components = 10) - Accuracy: 0.83153       Precision: 0.34193      Recall: 0.28500 F1: 0.31088     F2: 0.29482
 * LDA(eigen, auto, full, PCA components = 20) - Accuracy: 0.82473       Precision: 0.31467      Recall: 0.26700 F1: 0.28888     F2: 0.27534

Again there is no improvement on select K best (k = 4), eigen solver and auto shinkage.

Finally, in attempt to squeeze more perfomance out the best classifer achieved so far, the LDA classifer was fed as the base estimator for an ADA Boost ensemble classifer. This
produced an error message as LDA is unable to support sample weighting required for ADA Bosst. Istead another ensembe method, bagging, was used with the optimal LDA classifer so far
used as the base estimator.

 * Bagging(base = LDA, skb = 4)                - Accuracy: 0.83300       Precision: 0.36605      Recall: 0.34500 F1: 0.35521     F2: 0.34901
 * Bagging(base = LDA, no skb)                 - Accuracy: 0.79880       Precision: 0.29542      Recall: 0.36750 F1: 0.32754     F2: 0.35040

This produced a respectable F1 score but it still lags behind the origional LDA classifier. Removing select K best further degrades the accuracy
 
 
# Classifier Tuning
---

Though many of the machine learning classifiers featured have some predictive value out of the box, maximizing their effectiveness can be achieved by tuning the key parameters used. In the final classifer used (Linear Discriminant Analysis), limited tuning could be completed beyond changing the solver used or the number of features used. This impact of this tuning is detailed above.

For tuning of the other classifers (not used in final classifer), GridselctCV was used to try a range of possible tuning parameters and select the best possible classifier. For support vector machines the tuning parameters used were C, gamma, stopping critereon tolerence and maximum iterations. For Random Forest Classifers the parameters tuned were Number of Estimators, Min samples per leaf, maximum features used per split, the max tree depth and the split critereon.

Stratified Shuffle Split was not used in the final classifer as the simplicity of of LDA and the lack of use of a parameter search did not require a more complex cross validation than a simple train test split. 
For the classifiers using GridSelectCV, cross validation was included as part of the GridSelectCV, initially using a varying number of folds but later switching to Stratified Shuffle Split. Cross validation is essential as part of a pipeline including a Grid Select parameter search to ensure the combinations tested can be tested for the optimal combination on a subset of the data used in the pipeline. Cross valaidation is required to ensure seperation between the data used to train the classifier with each parameter combination and the data used to test the effectiveness of the parameter set. Without cross validation the same data would be used to train and test each parameter set would be the same resulting with potential overfitting using the parameters selected.
For the final classifer, train test split was used for cross validation with a 30% of the data used for testing. Seperating the test adn train data out is essential to ensuring the classifier metrics produced are reliable.


# Classifier Metrics
---

The core metrics used to quantify the quality of the classifiers investigated in this project were the accuracy, precision and recall.

Precision is the ratio of true positives to total positively identified by the classifer and reflects the ability of the classifer to not incorrectly lablel a negative sample as positive. Recall is the ratio of true positives to the total true positives and false positives (total correctly classifierd points) and reflects the ability of the classifer to locate all of the positive samples.

In addition to these the harmonic mean of precision and recall, the F1 score, was used to look at the combined ability of the classifers to identify POI's (recall) and to not identify false positives (precision).
Custom funtions were defined to further explore the classifiers effectiveness. Using Bayes rule a calculation for the probability of a Positive result being a valid True POI and the probability
of a Positive result being not being an actual POI were worked out:
tp, tn = True Negative, Positive
fp, fn = False Negative, Positive
N = total test events
p(T) = Probability of True POI = (tp + tn) / N
p(P) = Probability of Positive Classification = (tp + fp) / N
p()
p(T|P) = Probability of True POI given Positive Classification = p(P)*p(P|T) / p(P)
=> p(T|P) = ((tp + tn) / (tp + fp ) ) * p(P|T)
where p(P|T) = probability of positive result given actual POI = tp / (tp + tn)
=> p(T|P) = ((tp + tn) / (tp + fp ) ) * (tp / (tp + tn))
=> p(T|P) = tp / (tp + fp)
Unfortunatly this turns out to be the exact definition of precision, just expressed in a slightly differently way. The same math can be used to create a measure for the probability of a
positive result not being a valid True POI (false positives):
P(P|F) = fp / (fn + fp)
As this is not a standard metric in the sklean package this was retained and used to measure the tested classifiers.

Finally a simple attempt was made to create a metric that can account for the fitting and prediction time of the classifer and penalize the slowest. In the event that optimization or programatic feature analyisis was 
completed this was also included in the time penalty.
The resulting metric is the arithmetic mean of the F1 score used above and a speed score shown below:
speed score = speed_weight / e^(1000*(fit_time + pred_time))
Where speed_weight is parameter to adjust the relative weight of speed to f1 score, default being 0.5 giving 2/3 of final score to f1 score.
Expodential chosen not for any sound mathematical reasons but to ensure a 0 value for speed returns a speed score of 1 * speed_weight. Exponent of 1000 calculated from some testing with
artificially long running classifiers (user defined fit times) to give a metric that will only penalize noticably glacial classifiers.



# Results and Testing
--- 

The result for the final classifer chosen (Linear Discriminant Analysis using the eigen solver and select K best) is shown below. The F1 score for this classifer when running tester.py is 0.39 with a precision of 0.38 and a recall of 0.40. 
```
starting size 146
deleted:  LOCKHART EUGENE E
deleted:  THE TRAVEL AGENCY IN THE PARK
deleted:  TOTAL
deleted:  restricted_stock_deferred
deleted:  deferred_income
No of features uses =  22
my_dataset size =  143
poi =  18
Total Fit/Pred Time =  0.0
Classifier Accurcacy =  0.813953488372
Classifier Precision =  0.25
Classifier Recall =  0.166666666667
Classifier False Positive Prob =  0.375
Classifier F1 Score =  0.2
Speed Weighted F1 Score =  0.2

Pipeline(steps=[('selectkbest', SelectKBest(k=4, score_func=<function f_classif at 0x000000000D11A978>)), ('lineardiscriminantanalysis', LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage='auto',
              solver='eigen', store_covariance=False, tol=0.0001))])
        Accuracy: 0.83367       Precision: 0.38265      Recall: 0.40350 F1: 0.39280     F2: 0.39915
        Total predictions: 15000        True positives:  807    False positives: 1302   False negatives: 1193   True negatives: 11698
```
The final classifer results in a high accuracy of 83% though given the small number of POI's relative to total points in the dataset this is not a great reflection of the classifier performance. The overall F1 score of 0.39 is based on equal parts precision and recall with only a small difference  between these metrics. This shows the classifer is not stongly weighted towards locating all true positives of avoiding incorrect classification. This balance doesn't point towards any weakness of the classifier though the overall F1 score is not particularly good. This may be as a result of the a difficult and small dataset with a low number of possible true positives. Its also possible that a stonger classifer exists that was simply not found and explored by this porject.

# Conclusion
---

Overall the best classifer found for this data set was Linear Discriminant Analysis with Eigenvalue decomposition as the solver and automatic shrinkage applied. Though not critical for an LDA classifer, classifer tuning is important for optimizing the performance of the classifer and ensuring the best response for the dataset in question. This is especially important for decision tree based classifiers such as random forest classifer to prevent overfitting.

Applying more advanced preprocessing such as PCA was not effective at increasing the performance of the classifier and neither was attempting to boost the classifier using ensemble methods. Only a small subset of the features available (4 of the potential 22 including created features) are used in the final classifer with further features decreasing the effectivness of the classifer. The features used were exercised_stock_options, total_stock_value, long_term_incentive and bonus. Similarly applying feature scaling to the data did not improve the classifier performance though this was in contrast to the expected result of addign scaling.

In conclusion  it was found that more complex classifers do not always produce the best results. Much time was spent tuning and modifying the Support Vector Machine and Random Forest classifers with limited result. A key learning for nay future machine learning project would be to start simple and apply each new pre-processing step and feature carefully adn one at a time. Applying to many techniques all at one without carefully analysing the impact makes it difficult to understand where the loss of classifier performance is coming.



