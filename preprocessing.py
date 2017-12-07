#-*- coding:utf8 -*-
import sys
import numpy as np
import pandas as pd
import cPickle as cpk

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

train_file_path = './data/TrainData.xlsx'
test_file_path = './data/TestData.xlsx'

def drop_feature(data, feature_select=None):
    # default setting
    raw_feature_list = [u'学生ID', u'生源省市', u'性别', u'出生日期', u'民族', u'政治面貌', u'裸眼视力(左)',
       u'裸眼视力(右)', u'色盲', u'身高', u'体重', u'考生类型', u'外语语种', u'学科类型', u'年份',
       u'中学', u'科类', u'成绩', u'投档成绩', u'院系', u'省市', u'优惠加分', u'大类', u'高三排名',
       u'成绩方差', u'进步情况', u'专利数', u'社会活动', u'获奖数', u'竞赛成绩', u'综合GPA']
    drop_list = [u'学生ID', u'生源省市', u'性别', u'出生日期', u'民族', u'政治面貌', u'裸眼视力(左)',
       u'裸眼视力(右)', u'色盲', u'身高', u'体重', u'考生类型', u'外语语种', u'学科类型', u'年份',
       u'中学', u'科类', u'成绩', u'专利数', u'社会活动']
    feature_list = [u'投档成绩', u'院系', u'省市', u'优惠加分', u'高三排名', u'成绩方差', u'进步情况', 
       u'获奖数', u'竞赛成绩', u'综合GPA', u'大类']
    
    # mdf based on feature_select
    if feature_select is not None:
        feature_list = feature_select
    new_data = data[feature_list]
    return new_data


def mean_z_score_group_by_kv(data, v, key):
    print("Calc the mean and std grouped by %s, and normlize the value %s." % (key, v))
    data = mean_fill_missing(data, v, key)
    v_mean = data.groupby(by=[key])[v].mean()
    v_std = data.groupby(by=[key])[v].std()
    val = (data[v].values - v_mean[data[key]].values) / v_std[data[key]].values
    data[v] = val
    data.pop(key)
    return data


def zero_fill(data, v, key):
    val = data[[v]]
    val_ = val.fillna(0)
    data[v] = val_
    return data


def zero_z_score_group_by_kv(data, v, key):
    print("Calc the mean and std grouped by %s, and normlize the value %s." % (key, v))
    data = zero_fill(data, v, key)
    v_mean = data.groupby(by=[key])[v].mean()
    v_std = data.groupby(by=[key])[v].std()
    val = (data[v].values - v_mean[data[key]].values) / v_std[data[key]].values
    data[v] = val
    # data.pop(key)
    return data


def z_score(data, v, key=None):
    val = preprocessing.scale(data[[v]])
    data[v] = val
    return data


def mean_fill_missing(data, v, key=None):
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(data[[v]]) 
    val = imp.transform(data[[v]])
    data[v] = val
    return data


def mean_fill_missing_z_score(data, v, key=None):
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(data[[v]]) 
    val = imp.transform(data[[v]])
    data[v] = val
    return z_score(data, v, key)


def zero_fill_min_max_scale(data, v, key=None):
    # zero fill
    data = zero_fill(data, v, key)
    # minMax scale
    scaler = MinMaxScaler()
    scaler.fit(data[[v]])
    val = scaler.transform(data[[v]])
    data[v] = val
    return data


def zero_fill_z_score(data, v, key=None):
    # zero fill
    data = zero_fill(data, v, key)
    return z_score(data, v, key)


def onehot(data, v, key=None):
    v_onehot = pd.get_dummies(data[v], prefix=v + '_')
    data.pop(v)
    data_onehot_encoded = pd.concat([data, v_onehot],axis=1)
    return data_onehot_encoded


def zero_one(data, v, key=None):
    data[data[v] == u'女'] = 0
    data[data[v] == u'男'] = 1
    return data


def birth(data, v, key=None):
    age = data[key] - data[v]
    data[u'年龄'] = age
    data.pop(v)
    data.pop(key)
    return z_score(data, u'年龄', None)


def preprocess_factory(data, v):
    config = {
        #u'综合GPA': GPA_z_score_group_by_kv,
        u'优惠加分' : zero_fill_min_max_scale,
        # u'优惠加分' : zero_z_score_group_by_kv,
        u'投档成绩': mean_z_score_group_by_kv, 
        u'大类': onehot,
        u'高三排名': mean_fill_missing_z_score,
        u'成绩方差': mean_fill_missing_z_score,
        u'进步情况': mean_fill_missing_z_score,
        # u'高三排名': mean_fill_missing,
        # u'成绩方差': mean_fill_missing,
        # u'进步情况': mean_fill_missing,
        u'获奖数': zero_fill_min_max_scale,
        u'竞赛成绩': zero_fill_min_max_scale,
        # u'获奖数': zero_fill_z_score,
        # u'竞赛成绩': zero_fill_z_score,
        u'院系': onehot,
        u'性别': onehot,
        u'出生日期': birth,
    }
    config_key = {
        #u'优惠加分': None,
        u'优惠加分': u'大类',
        u'投档成绩': u'省市',
        u'大类': None,
        u'高三排名': None,
        u'成绩方差': None,
        u'进步情况': None,
        u'获奖数': None,
        u'竞赛成绩': None,
        u'院系': None,
        u'性别': None,
        u'出生日期': u'年份',
    }

    preprocess_fn = config[v]
    key = config_key[v]
    return preprocess_fn(data, v, key)


def process_train(feature_list, feature_to_preprocess):
    # read data
    data = pd.read_excel(train_file_path)
    # drop data
    data = drop_feature(data, feature_list)
    # preprocess feature
    for v in feature_to_preprocess:
        data = preprocess_factory(data, v)
        #print(data.head())
        #print(v)
    # change columns order (强迫症)
    val = data[[u'综合GPA']]
    data.pop(u'综合GPA')
    data[u'综合GPA'] = val
    # save data
    with open("./static/data_train.cpk", "w") as sfh:
        cpk.dump(data, sfh)
    data.to_excel('./static/data_train.xlsx', sheet_name='sheet1')
    print ("Preprocessing Done.")


def process_test(feature_list, feature_to_preprocess):
    # read data
    data = pd.read_excel(test_file_path)
    # drop data
    data = drop_feature(data, feature_list)
    # preprocess feature
    for v in feature_to_preprocess:
        data = preprocess_factory(data, v)
        #print(data.head())
        #print(v)
    with open("./static/data_test.cpk", "w") as sfh:
        cpk.dump(data, sfh)
    data.to_excel('./static/data_test.xlsx', sheet_name='sheet1')
    print ("Preprocessing Done.")


if __name__ == '__main__':
    feature_list = [
        u'综合GPA', 
        u'省市', 
        u'性别', 
        u'出生日期',
        u'年份', 
        u'投档成绩', 
        u'优惠加分', 
        u'高三排名', 
        u'成绩方差', 
        u'进步情况', 
        u'获奖数', 
        u'院系',
        u'竞赛成绩', 
        u'大类']
    feature_to_preprocess = [
        u'性别', 
        u'出生日期', 
        u'投档成绩', 
        u'优惠加分', 
        u'高三排名', 
        u'成绩方差', 
        u'进步情况', 
        u'获奖数',
        u'院系', 
        u'竞赛成绩', 
        u'大类']
    feature_list_test = [
        u'省市', 
        u'性别', 
        u'出生日期',
        u'年份', 
        u'投档成绩', 
        u'优惠加分', 
        u'高三排名', 
        u'成绩方差', 
        u'进步情况', 
        u'获奖数', 
        u'院系',
        u'竞赛成绩', 
        u'大类']
    feature_to_preprocess_test = [
        u'性别', 
        u'出生日期', 
        u'投档成绩', 
        u'优惠加分', 
        u'高三排名', 
        u'成绩方差', 
        u'进步情况', 
        u'获奖数',
        u'院系', 
        u'竞赛成绩', 
        u'大类']
    mode = sys.argv[1]
    if mode == 'train':
        process_train(feature_list, feature_to_preprocess)
    elif mode == 'test':
        process_test(feature_list_test, feature_to_preprocess_test)
    else:
        print ("Mode %s error!" % mode)