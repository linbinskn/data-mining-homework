import pandas as pd
import os
from os.path import join
import numpy as np

map_list = {
    'POPLATEK MESICNE': '每月一次',
    'POPLATEK TYDNE': '每周一次',
    'POPLATEK PO OBRATU': '交易后',
    'POJISTNE': '保险费支付',
    'SIPO': '物业费管理',
    'LEASING': '租金缴纳',
    'UVER': '偿还贷款',
    'PRIJEM': '向账户存款',
    'VYDAJ': '向账户提款',
    'VYBER KARTOU': '信用卡提现',
    'PREVOD Z UCTU': '来自其他银行的汇款',
    'VYBER': '提取现金',
    'PREVOD NA UCET': '汇款给其他银行',
    'SLUZBY': '账单支付',
    'UROK': '储蓄账户利息收入',
    'SANKC.UROK': '账户余额为负的处罚利息',
    'DUCHOD': '养老金'
}


def load_data(filename, path):
    filepath = join(path, filename)
    print(filepath)
    data = pd.read_csv(filepath)
    return data


def fillna(data):
    data = data.to_numpy()
    for i in range(len(data)):
        if pd.isnull(data[i]):
            data[i] = data[i - 1]
    return pd.DataFrame(data)


def meaning_transform(data):
    # 471格式存在错误，修正成正确格式
    data['account']['frequency'][471] = 'POPLATEK MESICNE'
    data['account']['frequency'] = data['account']['frequency'].map(lambda x: map_list[x])
    data['order']['k_symbol'] = data['order']['k_symbol'].map(lambda x: map_list[x])
    data['trans']['type'] = data['trans']['type'].map(lambda x: map_list[x])
    data['trans']['operation'] = data['trans']['operation'].map(lambda x: map_list[x])
    data['trans']['k_symbol'] = data['trans']['k_symbol'].map(lambda x: map_list[x])
    return data


def add_age_sex(data):
    age = []
    sex = []
    age_stage = []
    data.drop(4240, inplace=True)  # 删除错误行
    data.reset_index(drop=True, inplace=True)
    for birth in data['birth_number']:
        try:
            year = int(birth[0:2])
            month = int(birth[2:4])
        except:
            print(birth[0:2])
            print(birth[2:4])
        perage = 100 - year
        age.append(perage)
        if perage < 20:
            age_stage.append('少年')
        elif perage < 30:
            age_stage.append('青年')
        elif perage < 50:
            age_stage.append('中年')
        else:
            age_stage.append('老年')
        if (month > 12):
            sex.append(0)  # 0表示女
        else:
            sex.append(1)  # 1表示男
    data['age'] = pd.DataFrame(age)
    data['sex'] = pd.DataFrame(sex)
    data['age_stage'] = pd.DataFrame(age_stage)
    return data


# 将duration不等于12倍数的行作为噪声过滤掉
def duration_cleaning(data):
    for i, num in data['duration'].items():
        if num % 12 != 0:
            data.drop(i, inplace=True)
        data.reset_index(drop=True, inplace=True)
    return data


# 删除冗余信息
def drop_feature(data):
    data.trans.drop('account', axis=1, inplace=True)
    data.client.drop('birth_number', axis=1, inplace=True)
    data.order.drop('bank_to', axis=1, inplace=True)
    data.trans.drop('bank', axis=1, inplace=True)


def main():
    rootdir = os.getcwd()
    datadir = join(rootdir, 'dataset')
    datanames = os.listdir(datadir)
    full_data = {}
    for dataname in datanames:
        full_data[dataname.split('.')[0]] = load_data(dataname, datadir)
    full_data['account'].drop(4500, inplace=True)  # 删除account表格中的重复行
    full_data['account'].reset_index(drop=True, inplace=True)

    for key in full_data:
        for columns in full_data[key]:
            full_data[key][columns] = full_data[key][columns].apply(lambda x: np.NAN if str(x).isspace() else x)
            # 如果不存在缺省值，跳过该步骤
            if full_data[key][columns].isnull().sum() == 0:
                continue
            full_data[key][columns] = fillna(full_data[key][columns])
    full_data['client'] = add_age_sex(full_data['client'])
    full_data['loan'] = duration_cleaning(full_data['loan'])
    full_data = meaning_transform(full_data)
    full_data = drop_feature(full_data)


if __name__ == '__main__':
    main()
