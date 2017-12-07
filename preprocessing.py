#-*- coding:utf8 -*-
import sys
import numpy as np
import pandas as pd
import cPickle as cpk

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler

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


def GPA_z_score_group_by_kv(data, v, key):
    need_drop_key = [u'省市', u'院系']
    need_to_save_key = [u'综合GPA']
    # need_zero_fill = [u'大类']
    print("Calc the mean and std grouped by %s, and normlize the value %s." % (key, v))
    # if key is not None and key in need_zero_fill:
    #     val = data[[v]]
    #     val_ = val.fillna(0.1)
    #     data[v] = val_ 
    v_mean = data.groupby(by=[key])[v].mean()
    v_std = data.groupby(by=[key])[v].std()
    val = (data[v].values - v_mean[data[key]].values) / v_std[data[key]].values
    data[v] = val
    if key is not None and key in need_drop_key: 
        # delete key column
        data.pop(key)
    # need to save mean and std
    if key is not None and key in need_to_save_key:
        with open("./static/mean_" + key, "w") as sfh:
            cpk.dump(v_mean, sfh)
        with open("./static/std_" + key, "w") as sfh:
            cpk.dump(v_std, sfh)
    return data


def mean_fill_missing(data, v, key=None):
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(data[[v]]) 
    val = imp.transform(data[[v]])
    data[v] = val
    return data


def zero_fill_min_max_scale(data, v, key=None):
    # zero fill
    val = data[[v]]
    val_ = val.fillna(0)
    data[v] = val_
    # minMax scale
    scaler = MinMaxScaler()
    scaler.fit(data[[v]])
    val = scaler.transform(data[[v]])
    data[v] = val
    return data


def onehot(data, v, key=None):
    v_onehot = pd.get_dummies(data[v], prefix=v + '_')
    data.pop(v)
    data_onehot_encoded = pd.concat([data, v_onehot],axis=1)
    return data_onehot_encoded


def preprocess_factory(data, v):
    config = {
        u'综合GPA': zero_fill_min_max_scale,
        #u'综合GPA': GPA_z_score_group_by_kv,
        u'优惠加分' : zero_fill_min_max_scale,
        u'投档成绩': GPA_z_score_group_by_kv, 
        u'大类': onehot,
        u'高三排名': mean_fill_missing,
        u'成绩方差': mean_fill_missing,
        u'进步情况': mean_fill_missing,
        u'获奖数': zero_fill_min_max_scale,
        u'竞赛成绩': zero_fill_min_max_scale,
        u'院系': onehot,
    }
    config_key = {
        u'综合GPA': u'院系',
        u'优惠加分': None,
        # u'优惠加分': u'大类',
        u'投档成绩': u'省市',
        u'大类': None,
        u'高三排名': None,
        u'成绩方差': None,
        u'进步情况': None,
        u'获奖数': None,
        u'竞赛成绩': None,
        u'院系': None,
    }

    preprocess_fn = config[v]
    key = config_key[v]
    return preprocess_fn(data, v, key)


def process_train():
    # read data
    data = pd.read_excel(train_file_path)
    # drop data
    feature_list = [u'院系', u'投档成绩', u'省市', u'优惠加分', u'高三排名', u'成绩方差', u'进步情况', 
       u'获奖数', u'竞赛成绩', u'综合GPA', u'大类']
    data = drop_feature(data, feature_list)
    # preprocess feature
    feature_to_preprocess = [u'投档成绩', u'优惠加分', u'高三排名', u'成绩方差', u'进步情况', 
       u'获奖数', u'竞赛成绩', u'大类', u'院系']
    for v in feature_to_preprocess:
        data = preprocess_factory(data, v)
    # change columns order (强迫症)
    val = data[[u'综合GPA']]
    data.pop(u'综合GPA')
    data[u'综合GPA'] = val
    # save data
    with open("./static/data_train.cpk", "w") as sfh:
        cpk.dump(data, sfh)
    data.to_excel('./static/data_train.xlsx', sheet_name='sheet1')
    print ("Preprocessing Done.")


def process_test():
    # read data
    data = pd.read_excel(test_file_path)
    # drop data
    feature_list = [u'投档成绩', u'省市', u'优惠加分', u'高三排名', u'成绩方差', u'进步情况', 
       u'获奖数', u'竞赛成绩', u'大类']
    data = drop_feature(data, feature_list)
    # preprocess feature
    feature_to_preprocess = [u'投档成绩', u'优惠加分', u'高三排名', u'成绩方差', u'进步情况', 
       u'获奖数', u'竞赛成绩', u'大类']
    for v in feature_to_preprocess:
        data = preprocess_factory(data, v)
    # change columns order (强迫症)
    # val = data[[u'综合GPA']]
    # data.pop(u'综合GPA')
    # data[u'综合GPA'] = val
    # save data
    with open("./static/data_test.cpk", "w") as sfh:
        cpk.dump(data, sfh)
    data.to_excel('./static/data_test.xlsx', sheet_name='sheet1')
    print ("Preprocessing Done.")


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'train':
        process_train()
    elif mode == 'test':
        process_test()
    else:
        print ("Mode %s error!" % mode)