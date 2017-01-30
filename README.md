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
Using this financial and email information the aim is to create a classifer capable of correctly identifying persons of interest in this data set or a data set with the same features. 

financial features: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (all units are in US dollars)

email features: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)

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


### Support Vector Machines
Linear support vector machines was applied using a pipline containing the PCA and tuned over the number of components, C values, tolerence values, maximum iterations and intercept scaling.

Achieving a best case F1 score as per the output below proved challenging from a tuning point of view. Large numbers of components or tuning parameters outside a narrow range gave recall values of less than 0.1.

A persistant problem with this classifier was underidentification with many of the attempeted classifers giving false positive probabilites of > 90% (see classifer metrics for details of how this is calcualted).
Even for the best case classifier detailed below the probability that a given POI is incorrectly identified is 78%. This not practical for the stated purpose of correctly identifying POIs though the high level of precision for 
identifying non-POIs would ensure few POI's are omitted from the final output, at the expense of many false positives. 

Multiple attempts were made to create a SVM classifier using the RBF kernal but no working version could be found that did not identify all data points as Non-POIs. Using the poly kernal generated a F1 value of 0.11 with the majority of this poor score due to incorrectly identified POI's. 

Best Est =  Pipeline(steps=[('pca', PCA(copy=True, n_components=3, whiten=False)), ('linearsvc', LinearSVC(C=60000, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=3, loss='squared_hinge', max_iter=10000,
     multi_class='ovr', penalty='l2', random_state=None, tol=1e-05,
     verbose=0))])

 Total Pipeline Time =  67.779 

             precision    recall  f1-score   support

    Non-POI       0.94      0.82      0.87        38
        POI       0.30      0.60      0.40         5

avg / total       0.87      0.79      0.82        43

Classifier Accurcacy =  0.790697674419
Classifier Precision =  0.3
Classifier Recall =  0.6
Classifier False Positive Prob =  0.777777777778
Classifier F1 Score =  0.4
Speed Weighted F1 Score =  0.4
avg / total       0.80      0.63      0.69        43

Initial Classifier 

# Classifier Tuning
---

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

