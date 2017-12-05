#-*- coding:utf8 -*-
import numpy as np
import pandas as pd

train_file_path = './data/TrainData.xlsx'
test_file_path = './data/TestData.xlsx'


def GPA_z_score(data):
    # need to save mean and std
    gpa_mean = data.groupby(by=[u'院系'])[u'综合GPA'].mean()
    gpa_std = data.groupby(by=[u'院系'])[u'综合GPA'].std()
    gpa = (data[u'综合GPA'].values - gpa_mean[data[u'院系']].values) / GPA_std[data[u'院系']].values
    data[u'综合GPA'] = gpa
    return data


if __name__ == '__main__':
    data = pd.read_excel(train_file_path)
    data = GPA_z_score(data)

