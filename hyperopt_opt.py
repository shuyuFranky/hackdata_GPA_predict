# -*- coding: utf8 -*-
# Run an XGBoost model with hyperparmaters that are optimized using hyperopt
# The output of the script are the best hyperparmaters
# The optimization part using hyperopt is partly inspired from the following script:
# https://github.com/bamine/Kaggle-stuff/blob/master/otto/hyperopt_xgboost.py


# Data wrangling

import pandas as pd

# Scientific

import numpy as np


# Machine learning

import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

# Hyperparameters tuning

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

# Some constants

SEED = 314159265
VALID_SIZE = 0.2
TARGET = 'outcome'

#-------------------------------------------------#

# Utility functions

def intersect(l_1, l_2):
    return list(set(l_1) & set(l_2))


#-------------------------------------------------#

# Scoring and optimization functions
def score(params):
    print("Training with params: ")
    print(params)

    xlf = xgb.XGBRegressor(max_depth=params['max_depth'],
                    learning_rate=0.1,
                    n_estimators=params['n_estimators'],
                    silent=True,
                    objective='reg:linear',
                    eval_metric='rmse',
                    nthread=4,
                    gamma=params['gamma'],
                    min_child_weight=params['min_child_weight'],
                    max_delta_step=0,
                    subsample=params['subsample'],
                    colsample_bytree=params['colsample_bytree'],
                    colsample_bylevel=1,
                    reg_alpha=0,
                    reg_lambda=1,
                    scale_pos_weight=1,
                    seed=random_state,
                    missing=None)

    xlf.fit(train_features, y_train, eval_metric='rmse', verbose = True, eval_set = [(valid_features, y_valid)], early_stopping_rounds=1000)

    predictions = xlf.predict(dvalid, ntree_limit=xlf.best_iteration + 1)
    loss = np.mean((y_valid - predictions) ** 2)
    print ("测试误差为：%.6f" % loss)
    return loss


def optimize(
             trials,
             random_state=SEED):
    """
    This is the optimization function that given a space (space here) of
    hyperparameters and a scoring function (score here), finds the best hyperparameters.
    """
    # To learn more about XGBoost parameters, head to this page:
    # https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
    space = {
        'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
        'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
        'max_depth':  hp.choice('max_depth', np.arange(1, 14, dtype=int)),
        'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
        'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
        'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
        'eval_metric': 'rmse',
        'objective': 'reg:linear',
        'nthread': 4,
        'booster': 'gbtree',
        'tree_method': 'exact',
        'silent': 1,
        'seed': random_state
    }
    # Use the fmin function from Hyperopt to find the best hyperparameters
    best = fmin(score, space, algo=tpe.suggest,
                trials=trials,
                max_evals=250)
    return best

#-------------------------------------------------#


# Load processed data

# You could use the following script to generate a well-processed train and test data sets:
# https://www.kaggle.com/yassinealouini/predicting-red-hat-business-value/features-processing
# I have only used the .head() of the data sets since the process takes a long time to run.
# I have also put the act_train and act_test data sets since I don't have the processed data sets
# loaded.

data = pd.read_excel('./static/data_train.xlsx')
test_data = pd.read_excel('./static/data_test.xlsx')
GPA_y = data[u'综合GPA']
GPA_x = data
GPA_x.pop(u'综合GPA')

#-------------------------------------------------#



# Extract the train and valid (used for validation) dataframes from the train_df
train_features, valid_features, y_train, y_valid = train_test_split(
    GPA_x, GPA_y,
    test_size=VALID_SIZE,
    random_state=SEED)

print('The training set is of length: ', len(y_train.index))
print('The validation set is of length: ', len(y_valid.index))

#-------------------------------------------------#

# Run the optimization

# Trials object where the history of search will be stored
# For the time being, there is a bug with the following version of hyperopt.
# You can read the error messag on the log file.
# For the curious, you can read more about it here: https://github.com/hyperopt/hyperopt/issues/234
# => So I am commenting it.
trials = Trials()

best_hyperparams = optimize(
                            trials,
                            )
print("The best hyperparameters are: ", "\n")
print(best_hyperparams)
