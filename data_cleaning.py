import pandas as pd
import os
from os.path import join
import numpy as np

def load_data(filename, path):
    filepath = join(path, filename)
    print(filepath)
    data = pd.read_csv(filepath)
    return data

def fillna(data):
    for i in data:
        print(i)
        if i == 'NaN':
            data[i] = data[i-1]
    # print(data)
    return data

def main():
    rootdir = os.getcwd()
    datadir = join(rootdir, 'dataset')
    datanames = os.listdir(datadir)
    full_data = {}
    for dataname in datanames:
        full_data[dataname.split('.')[0]] = load_data(dataname, datadir)
    for key in full_data:
        for columns in full_data[key]:
            full_data[key][columns] = full_data[key][columns].apply(lambda x : np.NAN if str(x).isspace() else x)
            # 如果不存在缺省值，跳过该步骤
            if full_data[key][columns].isnull().sum() == 0:
                continue
            full_data[key][columns] = fillna(full_data[key][columns])

    print(full_data['trans'])
if __name__ == '__main__':
    main()


