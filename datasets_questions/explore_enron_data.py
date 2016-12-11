#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import sys

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print 'name lenght',  len(enron_data)


poi = 0
for i in enron_data:
    if enron_data[i]['poi'] == True:
        poi += 1
print 'poi ni:', poi

sys.path.append("../final_project/")
from poi_email_addresses import poiEmails
email_list = poiEmails()

poi_names = 0
with open('../final_project/poi_names.txt', 'r') as names_txt:
    for line in names_txt:
        try:
            if str(line)[1] in ['y','n']:
                poi_names += 1
        except IndexError:
            pass

print 'full email lenght: ', len(email_list)
print 'poi names number', poi_names
#Jeffrey K Skilling
print enron_data['SKILLING JEFFREY K']['total_payments']
print enron_data['LAY KENNETH L']['total_payments']
print enron_data['FASTOW ANDREW S']['total_payments']

salaries = []
for name in enron_data:
    if enron_data[name]['total_payments'] == 'NaN':
            salaries.append(enron_data[name]['total_payments'])
print len(salaries), len(enron_data)
        