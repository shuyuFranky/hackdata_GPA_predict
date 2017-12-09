# -8- coding: utf8 -*-
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
import pandas as pd

data_train = pd.read_excel("./static/data_train.xlsx")
data_test = pd.read_excel("./static/data_test.xlsx")
#fh = open("./static/data.cpk", "r")
#data = cpk.load(fh)
#fh.close()
data_test = data_test.fillna(0)
GPA_y = data_train[u'综合GPA']
GPA_x = data_train
GPA_x.pop(u'综合GPA')

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

xlf.fit(GPA_x, GPA_y, eval_metric='rmse')

# 计算 auc 分数、预测
preds = xlf.predict(data_test)

raw_test = pd.read_excel("./data/TestData.xlsx")
res = pd.DataFrame({
    '学生ID' : raw_test[u'学生ID'],
    '综合GPA' : preds
})
# res.drop([0])
res.to_csv('./static/answer.csv', index=False)
print("Test Done")
