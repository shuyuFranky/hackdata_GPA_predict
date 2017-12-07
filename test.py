# -*- coding: utf8 -*-
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVR

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import cPickle as cpk

if __name__ == '__main__':
    data_train = pd.read_excel("./static/data_train.xlsx")
    data_test = pd.read_excel("./static/data_test.xlsx")
    #fh = open("./static/data.cpk", "r")
    #data = cpk.load(fh)
    #fh.close()
    data_test = data_test.fillna(0)
    GPA_y = data_train[u'综合GPA']
    GPA_x = data_train
    GPA_x.pop(u'综合GPA')
 
    lm = LinearRegression()
    # lm = LinearSVR()

    lm.fit(GPA_x, GPA_y)

    pred = lm.predict(data_test)

    raw_test = pd.read_excel("./data/TestData.xlsx")
    res = pd.DataFrame({
        '学生ID' : raw_test[u'学生ID'],
        '综合GPA' : pred
    })
    res.drop([0])
    res.to_csv('./static/answer.csv', index=False)
    print ("Test Done.")
