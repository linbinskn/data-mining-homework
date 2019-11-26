import pandas as pd
import os
from os.path import join
from utils import *
import model
import pre_processing
import feature_selector
"""
    Usage:: load file1csv and four questionnaires
    :param None
    :return :5 tables in form of pandas
    :rtype: list
"""


def main():
    full_data = pd.read_csv("G:/tsinghua/数据仓库与数据挖掘/数据仓库第二次大作业/clustering/data.csv")
    full_data = pre_processing.transform_categorical_encode_preprocess(full_data)
    full_data = full_data.sample(frac=1)
    y = full_data['y']
    X = full_data.drop('y', axis=1)
    X = feature_selector.identify_low_importance(X, y, 15, valid_ratio=0.2)
    train_data_upsample, y_upsample = pre_processing.up_sample(X, y)
    model_sets = model.Model()
    model_name = 'LR'
    model_sets.fit(train_data_upsample, y_upsample, model_name)
    result = model_sets.predict(model_name)
    result_auc = model_sets.evaluation_auc()
    print("auc: ", result_auc)
if __name__ == '__main__':
    main()

#full_data = pd.read_csv("C:/tsinghua/数据仓库与数据挖掘/数据仓库 第三次大作业/第三次大作业/第三次大作业/data/classification/train_set.csv")

