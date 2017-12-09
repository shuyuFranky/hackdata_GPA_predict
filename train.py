# -*- coding: utf8 -*-
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVR
from sklearn.ensemble import GradientBoostingRegressor

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


    # lm = LinearRegression()
    # lm = LinearSVR()
    lm = GradientBoostingRegressor(
        loss='ls', 
        learning_rate=0.1, 
        n_estimators=100, 
        subsample=1.0, 
        criterion='friedman_mse', 
        min_samples_split=2, 
        min_samples_leaf=1, 
        min_weight_fraction_leaf=0.0, 
        max_depth=3, 
        min_impurity_decrease=0.0, 
        min_impurity_split=None, 
        init=None, 
        random_state=None, 
        max_features=None, 
        alpha=0.9, 
        verbose=0, 
        max_leaf_nodes=None, 
        warm_start=False, 
        presort='auto') 
    
    lm.fit(X_train, Y_train)
    pred_train = lm.predict(X_train)
    pred_test = lm.predict(X_test)
    print ("训练误差为：%.6f" % np.mean((Y_train - lm.predict(X_train)) ** 2))
    print ("测试误差为：%.6f" % np.mean((Y_test - lm.predict(X_test)) ** 2))

    # test
    # test_data = pd.read_excel("./static/data_test.xlsx")
    # predict = lm.predict(test_data)
