# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:15:47 2020

@author: Tati
"""


def classify(features_train, labels_train):       
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    return clf
    