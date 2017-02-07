# UD_P5_MachineLearning
Machine Learning Mini Project and Final Project

---
---

# Introduction
---

In August 23rd 2000 Enron stock hit a record high of $90 a share. 16 months later, on December 2nd 2001, the company declared Chapter 11 bankruptcy with the loss of over 4000 jobs[2]. 
This classic case of corperate fraud is used as the basis Project 5 in the Udacity Data Science Nanodegree - Intro to Machine learning.

The purpose of the project is to identify persons of interest (POI's) within the Enron email corpus (May 7th 2015 version used [2]) combined with publicly avialable pay and share data. 
For each of 145 the manually identified POI's/non-POI's a set of descriptive features is associated with the person giving either financial or email information on the person in question.
Using this financial and email information the aim is to create a classifer capable of correctly identifying persons of interest in this data set that can be applied across the full enron corpus.

In this project a machine learning classifer implemented using the Python package Scikit-Learn will be developed. The use of machine learning in this task allows a classification problem 
such as this to be completed programmatically with far less human input than manual idenification  of POIs and in a fraction of the time. A variety of supervised machine learning classifiers
have been applied including Nieve Bayes, Support Vector Machines and Ensenmble methods.
Naive Bayes was selected for its simplicity as an initial classifer and to get idea of an out of the box performance.
Support vector machines classifier was used as it is a popular and well understood alghorith that works well with binary classification problems though the high feature count of this data set and low 
number of POIs proved to be an issue. SVM does not work well with heavly skewed classification populations and this proved a problem.
Random Forest classifer was used to take advantage of the high number of possible features in the dataset and to leverage the power of an ensemble method.
Linear Discriminant Analysis classifer was used with shrinkage applied to deal with the low training points relative to possible features. Based on the success of Naive Bayes and the assumed
covariance of the data features (especially in the financial data). 

The data set provided contains 143 manually identified persons, if they are a person of interest (POI) or not (represente by a 1 or 0) and a selection of financial and emsil features shown below.

financial features: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (all units are in US dollars)

email features: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)

There are a total of 146 data points of which 18 are identified as POI's and the rest are not.

The main file for this project located in this repositry is poi_id.py  located in the final_project directory, this is the completed version of the classifier. 
Multiple steps in the learning process and alternative classifiers have been saved in the format poi_id_X.py to record attempted classifiers (and revert in event of dead end).

[1] http://news.bbc.co.uk/1/hi/business/1759599.stm
[2] https://www.cs.cmu.edu/~./enron/

# Data Features and Selection
---

Initially no feature selection was completed and all possible features were included in the dataset to explore which may be usefull or need to be discarded. Running the code on a 
simple classifer generated a ValueError due to emails stored as string not convertable into floats as part of featureFormat. As the email address is unlikely to be valuable in identifying
POI's this feature was excluded for simplicity.

For a simple check of the features provided the a feature_checker function of the poi_id_2_selection.py file was defined to loop through all features and count the number of non zero / non NaN item. 
The feature 'loan_advances' provides 3 useful results with 2 non-poi. Based on this the Loan Advances feature is unlikey to be a useful at identifying POI's and was removed.

A number of new features were created, mostly to represent payments of email counts as fractions of total and therefore scale the features more appropriatly. For example two employees
that excercise $100,000 worth of their shares might apprear similar in the 'exercised_stock_options' feature but if one employee had $200,000 total stock and the other had $4,000,000 in stock these 
sales of shares would not seem as equivelent. Creating reatio features for some of key financial features of each data point reduces the 

List of added features:
'exercised_stock_ratio' = Fraction of excercised from total unrestricted stock available to excersise (not fully sure if deferred restircted stock should also be subtracted from this as not clear about the differnces between these)
'to_poi_email_fraction' = Fraction of emails sent that were to  POI.
'from_poi_email_fraction' = Fraction of email recived that were from a POI.
'total_bonus_compensation' = Combined bonus, fees, expenses and otehr benifits that are not part of base salary. Other may belong in her but unsure.
'salary_fraction_total_bonus' = Above metric as fraction of salary (this may be > 1 ).

Further reduction of the dataset dimensionality was completed using PCA (Principle Component Analysis), employed as part of a pipeline. A breakdown of the top components calculated by PCA was printed out
from the data set based on their explained variance:
1  explained var =   0.871364222652
2  explained var =   0.0957615880701
3  explained var =   0.0126077469676
4  explained var =   0.0108225652873
5  explained var =   0.00478163091311

From this we can see the first component accounts for the vast majority of variance in the dataset post transformation and the number of potentially useful dimensions reduced to more managable number.

Sclaing of features was added at a later stage in the project process and included as the first step in the pipleine but no noticible improvement in performance was see, possibly due to the 
inclusion of ratio based features added to the feature set that were then included in the PCA.

# Outlier Removal
---

From the Feature Selection module of the MD course an initial outlier was identified as 'TOTALS' and removed from the dataset. Further investigation using the key_checker function defined in poi_id_2_selection.py 
found a name 'LOCKHART EUGENE E' had no features with a non-0 or non-NaN result and so was scrubbed. Looking at the number of non-0/NaN features returned for each data point
showed a 'THE TRAVEL AGENCY IN THE PARK' with only 3 useful features and a name that suggests it is not a person, let a alone a POI. This data point was removed as part of the initial data checking.
This initial outlier scrub improved the F1 score of the an out of the box Guassian Naive Bayes classifer from 0.09 (subset of features used as per poi_id_1_simple) to 0.57 with the majority of
this benifit coming from the removal of 'TOTALS'


# Classifiers
---
### Nieve Bayes

OOB Nieve Bayes
Classifier Fit Time =  0.0
Classifier Predict Time =  0.0
Classifier Accurcacy =  0.857142857143
Classifier Precision =  0.5
Classifier Recall =  0.666666666667
Classifier False Positive Rate =  0.666666666667
Classifier F1 Score =  0.571428571429
Speed Weighted F1 Score =  0.535714285714

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
 

### Decision Trees



Out of Box:
Classifier Accurcacy =  0.906976744186
Classifier Precision =  1.0
Classifier Recall =  0.2
Classifier False Positive Prob =  0.0
Classifier F1 Score =  0.333333333333
Speed Weighted F1 Score =  0.333333333333

With PCA and GridselctCV looping over possible values for 'n_components', 'n_estimators', 'max_features', 'min_samples_leaf' and 'max_depth'. Unlike previous classifers above, the mistake in the
Speed Weighted F1 Score was fixed and it is giving a reduced F1 score for this slower alghorith (subjective measure but sclaled to penalize when the metric creator is getting impatient).

Annoyingly, due to the random nature of the Random Forest Classifier the high score below is not repeatabe. Running tester code on below produced an poor F1 score of 0.22. 
Either the variation in the features selected by random forest created "lucky" classifer or its possible the series of POI's selected by the classifer fitted to correct results by pure


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

[ 0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.
  1.  0.  0.  0.  0.  0.  0.]
Classifier Accurcacy =  0.953488372093
Classifier Precision =  1.0
Classifier Recall =  0.6
Classifier False Positive Prob =  0.0
Classifier F1 Score =  0.75
Speed Weighted F1 Score =  0.464092184753


After some frustration with the mediocre effectivness of the random forest classifiers (and on suggestion from the forums), the scoring parameter was uptated to 'f1' and 
StratifiedShuffleSplit used as cross validation.

Running an out of box Random Forest Classifier, searching over PCA n_components gives a F1 score of 0.36, an improvement to the origional OOB result of 0.33.

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

To further boost the preformace of the classifier, skaling was introduced using a MinMax scaler with default settings. The reasoning being that some of the financial components were producing
incorrect splitting of the dataset as part of the tree generation and causing extremem overfitting along some of the decision trees used.
A wide search of possible tunign parameters was used in an effort to seek out the optimum parameters for the classifer though this resulted in a huge runtime.

citerion = ["gini", "entropy"]
n_estimators  = [5,10, 15, 20, 30, 40]
max_features = [0.01,0.03,0.04,0.08,0.1,0.15,0.23, 0.4]
max_split = [2,3,4,6,10, 20]
min_samples_leaf = [1,2,4,6]
max_depth = [5,10,None]

The best estimate classifier produced a F1 score of 0.38 using the stratified shuffle split data in grid search and an even less impressive 0.25 when fitted sperately to a train test split data that was 
used to calculate the custom metric produced.

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

Running tester.py resulted in a abismal 0.18 and failled to meet the passing criterion.

Accuracy: 0.85747       Precision: 0.38725      Recall: 0.11850 F1: 0.18147     F2: 0.13760
        Total predictions: 15000        True positives:  237    False positives:  375   False negatives: 1763   True negatives: 12625

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

In addition to the classifers attempted already, a new classifer was added - Linear Discriminant Analysis. This classifier was chosen as result of some searching for classifers that work
well with small datasets and a larger number of features. More detailed investigation of the use of Discriminant Analysis and reading the sklearn documentation suggests it works well with
multicollinearity, requires limited tuning and works under simlilar mathematical principles to the best classifer so far, Guassian Naive Bayes (based on Bayes rule and assumes multivariate normal distribution). 


http://stats.stackexchange.com/questions/63565/good-classifiers-for-small-training-sets
		
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

Got a divide by zero when trying out: SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
Precision or recall may be undefined due to a lack of true positive predicitons.

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

 * LDA(eigen, auto shink, miniset, kbest = 1) - Accuracy: 0.82271       Precision: 0.36491      Recall: 0.32550 F1: 0.34408     F2: 0.33269
 * LDA(eigen, auto shink, miniset, kbest = 4) - Accuracy: 0.82271       Precision: 0.36491      Recall: 0.32550 F1: 0.34408     F2: 0.33269
 
No improvement with the already small manulally selected feature set though there is suprisingly no 

 * LDA(eigen, auto, full, kbest = 1)          - Accuracy: 0.78160       Precision: 0.27614      Recall: 0.39350 F1: 0.32454     F2: 0.36267
 * LDA(eigen, auto, full, kbest = 10)         - Accuracy: 0.78160       Precision: 0.27614      Recall: 0.39350 F1: 0.32454     F2: 0.36267
 
 !!!!!!!!!all F1 scores are the same - need to build pipe for tester to comprehend preprocessing



# Classifier Tuning
---

Though many of the machine learning classifiers featured have some predictive value out of the box, maximizing their effectiveness can be achieved by tuning the key parameters used.

For tuning of the classifers, GridselctCV was used to try a range of possible tuning parameters and select the best possible classifier.

Cross validation was also included as part of the GridSelectCV 

SVM grid:
n_components = [1,2,3,4]
C_values = [40000, 50000,60000,70000]
tol_values = [0.0001,0.00008, 0.00005,0.00003]
iter_values = [10,100,1000,10000]
ic_scl= [1,2,3,4,5]


# Classifier Metrics
---

The core metrics used to quantify the quality of the classifiers investigated in this project were the accuracy, precision and recall. 

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

# Conclusion
---

