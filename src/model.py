#############################################
#   model training and predicting           #
#############################################
import numpy as np
import pandas as pd
from numpy import array
import matplotlib.pyplot as plt
from utils import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


class Model:
    #定义模型
    LR = LogisticRegression()

    def LR_train(self, x_train, y_train):
        LR = self.LR.fit(x_train, y_train)


    def fit(self, train_data, y, model):
        label = y
        x_train, x_test, y_train, y_test = train_test_split(
            train_data, label, test_size=0.30, random_state=0
        )
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        if model == 'LR':
            self.LR_train(x_train, y_train)
        '''
        if model == .***
            self.***_train(x_train, y_train)
        '''
        return


    def predict(self, model):
        if model == 'LR':
            self.result = self.LR.predict(self.x_test)
        return

    def evaluation_auc(self, sample_weight=None):
        return roc_auc_score(self.y_test, self.result)
