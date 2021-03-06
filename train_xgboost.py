# -8- coding: utf8 -*-
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
import pandas as pd

data = pd.read_excel("./static/data_train.xlsx")
data_ = data.fillna(0)
GPA_y = data_[u'综合GPA']
GPA_x = data_
GPA_x.pop(u'综合GPA')

X_train, X_test, Y_train, Y_test = train_test_split(GPA_x, GPA_y,
test_size=0.33, random_state=5)


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=1729)
#print(X_train.shape, X_test.shape)

#模型参数设置
xlf = xgb.XGBRegressor(max_depth=10,
                        learning_rate=0.1,
                        n_estimators=250,
                        silent=True,
                        objective='reg:linear',
                        nthread=-1,
                        gamma=2.0,
                        min_child_weight=1,
                        max_delta_step=0,
                        subsample=0.85,
                        colsample_bytree=0.7,
                        colsample_bylevel=1,
                        reg_alpha=0,
                        reg_lambda=1,
                        scale_pos_weight=1,
                        seed=1440,
                        missing=None)

xlf.fit(X_train, Y_train, eval_metric='rmse', verbose = True, eval_set = [(X_test, Y_test)], early_stopping_rounds=1000)

pred_train = xlf.predict(X_train)
pred_test = xlf.predict(X_test)
print ("训练误差为：%.6f" % np.mean((Y_train - xlf.predict(X_train)) ** 2))
print ("测试误差为：%.6f" % np.mean((Y_test - xlf.predict(X_test)) ** 2))

# 计算 auc 分数、预测

data_test = pd.read_excel("./static/data_test.xlsx")
data_test = data_test.fillna(0)
preds = xlf.predict(data_test)

raw_test = pd.read_excel("./data/TestData.xlsx")
res = pd.DataFrame({
    '学生ID' : raw_test[u'学生ID'],
    '综合GPA' : preds
})
res.drop([0])
res.to_csv('./static/answer.csv', index=False)
