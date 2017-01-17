# UD_P5_MachineLearning
Machine Learning Mini Project and Final Project

---
---

# Introduction
---

In August 23rd 2000 Enron stock hit a record high of $90 a share. 16 months later, on December 2nd 2001, the company declared Chapter 11 bankruptcy with the loss of over 4000 jobs[2]. 
This classic case of corperate fraud is used as the basis Project 5 in the Udacity Data Science Nanodegree - Intro to Machine learning.

The purpose of the porject is to identify persons of interest (POI's) within the Enron email corpus (May 7th 2015 version used [2]) combined with publicly avialable pay and share data.

Using a prepared dataset of 145 manually identified POI/non-POI's a machine learning classifier has been created in Python using the numpy and scikit learn packages.

The main file for this project located in this repositry is poi_id.py  located in the final_project directory, this is the completed version of the classifier. 
Multiple steps in the learning process and alternative classifiers have been saved in the format poi_id_X.py to record attempted classifiers (and revert in event of dead end).

[1] http://news.bbc.co.uk/1/hi/business/1759599.stm
[2] https://www.cs.cmu.edu/~./enron/

# Data Features and Selection
---

Initially no feature selection was completed and all possible features were included in the dataset to explore which may be usefull or need to be discarded. Running the code on a 
simple classifer generated a ValueError due to emails stored as string not convertable into floats as part of featureFormat. As the email address is unlikely to be valuable in identifying
POI's this feature was excluded for simplicity.

For a simple check of the features provided the a feature_checker function was defined to loop through all features and count the number of non zero / non NaN item. 
The feature 'loan_advances' provides 3 useful results with 2 non-poi. Based on this the Loan Advances feature is unlikey to be a useful at identifying POI's and was removed.


# Outlier Removal
---



# Classifiers
---

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

