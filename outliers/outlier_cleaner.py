#!/usr/bin/python

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    #creating a series of tuples for the prediction-output pairs
    predictions = zip(ages, net_worths, predictions)
    data = []
    for i in predictions:
        error = i[1] - i[2]
        tup = (i[0], i[1], error)
        data.append(tup)
        
    sorted_data = sorted(data, key=lambda point: point[2])
    cleaned_data = sorted_data[ len(data)/10:]
    

    return cleaned_data

