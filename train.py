# -*- coding: utf8 -*-
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVR

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import cPickle as cpk

if __name__ == '__main__':
    data = pd.read_excel("./static/data_train.xlsx")
    #fh = open("./static/data.cpk", "r")
    #data = cpk.load(fh)
    #fh.close()
    # data = data.fillna(0)
    GPA_y = data[u'综合GPA']
    GPA_x = data
    GPA_x.pop(u'综合GPA')
    # lm = LinearRegression()
    # lm.fit(GPA_x, GPA_y)
    X_train, X_test, Y_train, Y_test = train_test_split(GPA_x, GPA_y,
    test_size=0.33, random_state=5)


    lm = LinearRegression()
    # lm = LinearSVR()
    
    
    lm.fit(X_train, Y_train)
    pred_train = lm.predict(X_train)
    pred_test = lm.predict(X_test)
    print ("训练误差为：%.6f" % np.mean((Y_train - lm.predict(X_train)) ** 2))
    print ("测试误差为：%.6f" % np.mean((Y_test - lm.predict(X_test)) ** 2))

    # test
    # test_data = pd.read_excel("./static/data_test.xlsx")
    # predict = lm.predict(test_data)
