# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:27:37 2020

@author: Tati
"""


def mess_accuracy(features_test, labels_test, model):
    from sklearn.metrics import accuracy_score
    
    ### predict labels for the test set 
    pred = model.predict(features_test)
    
    ### return accuracy of the model
    return accuracy_score(labels_test, pred)