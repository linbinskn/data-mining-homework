#############################################
#   data cleaning and feature engineering   #
#############################################

import pandas as pd
import numpy as np
from utils import *


"""
    Usage:: fill NAN with mean
    :param data: data from file1csv
    :return :data
    :rtype: object(pandas)
"""
def fill_nan_mean(data):
    values = dict([(colname, colvalue) for colname, colvalue in zip(data.columns.tolist(), data.mean().tolist())])
    data.fillna(value=values, inplace=True)
    return data


"""
    Usage:: fill NAN with preceding value
    :param data: data from file1csv
    :return :data
    :rtype: object(pandas)
"""
def fill_nan_ffill(data):
    data.fillna(method='ffill', inplace=True)
    return data


"""
    Usage:: fill NAN with single_user_mean
    :param data: data from file1csv
    :return :data
    :rtype: object(pandas)
"""
def fill_nan_user(data):
    data = data.groupby('USERID').apply(lambda x:fill_userid_group(x))
    return data


def fill_userid_group(group):
    values = dict([(colname, colvalue) for colname, colvalue in zip(group.columns.tolist(),group.mean().tolist())])
    group.fillna(value=values, inplace=True)
    return group





"""
    Usage:: split data into train and test
    :param data: full data including file1csv and four questionnaires
    :return :train_data, test_data
    :rtype: object(pandas)
"""
@timeit
def traindata_split(full_data):
    split_data = full_data['file1csv']
    train_data = pd.DataFrame(columns=full_data.columns)
    test_data = pd.DataFrame(columns=full_data.columns)
    for day in range(565, 731, 1):
        day_data = split_data[split_data['DAY'] == day]
        test_data = pd.concat([test_data, day_data])
    for day in range(195, 565, 1):
        day_data = split_data[split_data['DAY'] == day]
        train_data = pd.concat([train_data, day_data])
    return train_data, test_data



"""
    Usage:: up sample for raw data to balance the negative/positive sample
    :param X: train data loaded from csv
           y: label loaded from csv
    :return :X: train data improved after up sample
             y: label improved after up sample
    :rtype: object(pandas)
"""
@timeit
def up_sample(X, y):
    len_y = y.shape[0]
    data1 = []
    data0 = []
    # 记录标签为1的样例和标签为0的样例
    for i in range(0, len_y):
        if y[i] == 1:
            data1.append(i)
        else:
            data0.append(i)
    if len(data1) >= len(data0):
        return X, y
    import numpy as np
    import pandas as pd
    # 如果标签为1的样例太少，那么就随机挑选一些标签为1的样例，复制几遍，让正负样例均衡
    index = np.random.randint(len(data1), size=len(data0) - len(data1))
    data = []
    for id in index:
        data.append(data1[id])

    up_X = pd.concat([X, X.iloc[data, :]], axis=0, sort=False, ignore_index=True)
    up_y = pd.concat([y, y.iloc[data]], axis=0, sort=False, ignore_index=True)

    # 先将标签与数据结合起来
    ret = pd.concat([up_X, up_y], axis=1, sort=False)

    # 再将标签和数据分开
    y = ret['y']
    X = ret.drop('y', axis=1)
    # 返回标签和数据
    return X, y


"""
    Usage:: down sample for raw data to balance the negative/positive sample
    :param X: train data loaded from csv
           y: label loaded from csv
    :return :X: train data improved after down sample
             y: label improved after down sample
    :rtype: object(pandas)
"""
@timeit
def down_sample(X, y, param, config):
    len_y = y.shape[0]
    data1 = []
    data0 = []
    # 记录标签为1的样例和标签为0的样例
    for i in range(0, len_y):
        if y[i] == 1:
            data1.append(i)
        else:
            data0.append(i)
    print('param: ', param)
    print(1, len(data1))
    print(0, len(data0))
    if len(data1) * param >= len(data0):
        return X, y

    # 如果标签为1的样例太少，那么就随机挑选部分标签为0的样例
    index = np.random.randint(len(data0), size=len(data1) * param)
    data = []
    for id in index:
        data.append(data0[id])
    # pd.concat([up_X, up_y], axis=1, sort=False)
    X = pd.concat([X.iloc[data1, :], X.iloc[data, :]], axis=0, sort=False)
    y = pd.concat([y.iloc[data1], y.iloc[data]], axis=0, sort=False)
    # 复制完成之后，需要对所有数据按照时间重新排序，保证其时间顺序
    # 先将标签与数据结合起来
    ret = pd.concat([X, y], axis=1, sort=False)
    # 再将标签和数据分开
    y = ret['y']
    X = ret.drop('y', axis=1)
    # print(pd.concat([X, y], axis=1, sort=False))
    return X, y


"""
    Usage:: frequency encoder for all category type data
    :param df: full_data
    :return :
    :rtype: object(pandas)
"""
@timeit
def transform_categorical_encode_preprocess(df):
    table = pd.DataFrame([])
    category_list=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    for c in category_list:
        col = frequency_encoder(df, c)
        table[c + '_count'] = col
    print('df.shape:', df.shape)
    df = pd.concat([df, table], axis=1, sort=False)
    df.drop(category_list, axis=1, inplace=True)
    print('df.shape', df.shape)
    print('table.shape', table.shape)
    return df

def frequency_encoder(df, c):
    dic = df[c].value_counts().to_dict()
    ret = df[c].apply(lambda x: dic[x])
    return ret



