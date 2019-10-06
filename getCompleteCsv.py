# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 11:41:14 2019

@author: juegos
"""
import pandas as pd
import numpy as np
import csv

def uniteCSV(csvName1, csvName2, csvName3, targetNameTrain, targetNameTest):
    import random
    finalCSVTrain = csv.writer(open(targetNameTrain, 'a+'))
    finalCSVTest = csv.writer(open(targetNameTest, 'a+'))
    unitedCsv = list()
    csv1, csv2, csv3 = np.array(pd.read_csv(csvName1, encoding="latin-1")), np.array(pd.read_csv(csvName2, encoding="latin-1")), np.array(pd.read_csv(csvName3, encoding="latin-1"))
    unitedCsv = [i for i in csv1]
    unitedCsv2 = [j for j in csv2]
    unitedCsv3 = [z for z in csv3]
    unitedCsv.extend(unitedCsv2)
    unitedCsv.extend(unitedCsv3)
    random.shuffle(unitedCsv)
    for i in range(int(len(unitedCsv)*0.95)):
        finalCSVTrain.writerow(unitedCsv[i])
    for j in range(int(len(unitedCsv)*0.95), len(unitedCsv)):
        finalCSVTest.writerow(unitedCsv[j])