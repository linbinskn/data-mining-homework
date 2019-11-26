from utils import *
import numpy as np
import lightgbm as lgb




@timeit
def identify_collinear(data, correlation_threshold):
    """基于皮尔逊相关系数识别共线特征"""
    num_col = data.columns
    ops = {}
    if len(num_col) == 0:
        ops['collinear'] = []
        return

    if len(num_col) > 1000 and len(data) > 20000:
        data = data_sample(data, 8000)
    elif len(data) >= 50000:
        data = data_sample(data, 10000)
    else:
        pass

    corr_matrix = data[num_col].corr()

    # 获取相关性矩阵上三角
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column].abs() >= correlation_threshold)]
    ops['collinear'] = to_drop
    print('%d features with a correlation magnitude greater than %0.3f.' % (
    len(ops['collinear']), correlation_threshold))
    data.drop(to_drop, axis=1, inplace=True)
    return data


@timeit
def identify_low_importance(data, labels, top_k=None, top_ratio=None, free_list=[], valid_ratio=0.2, num_boost_round=500, params=None, not_need_list=[]):

    MIN_FEAT_IMPORTANT = 0

    log(f"free_list length : {len(free_list)}")
    log(f"top_k : {top_k}")
    """
    根据比例 or 个数选取重要性靠前特征（识别出重要性靠后特征）
    top_k:      个数选取
    top_ratio:  比例选取
    均不为None则选取两者中较少特征
    """
    ops = {}
    num_feats = data.columns
    if len(num_feats) == 0:
        ops['low_importance'] = []
        ops['needed_cols'] = free_list
        return

    X_train, X_val, y_train, y_val = data_split(data[num_feats], labels, valid_ratio)
    if params == None:
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "seed": 1,
            "num_threads": 4,
            "max_depth": 6,
            "num_leaves": 32,
            "feature_fraction": 0.6,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "learning_rate": 0.01,
            #"min_child_weight": 5,
        }
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val)
    model = lgb.train(params,
                      train_data,
                      num_boost_round,
                      valid_data,
                      verbose_eval=100)

    feature_importances = pd.DataFrame()
    feature_importances['features'] = X_train.columns
    feature_importances['importance_gain'] = model.feature_importance(importance_type='gain')
    print(feature_importances['importance_gain'])
    feature_importances.sort_values('importance_gain', inplace=True, ascending=False)

    total_features_without_zero = len(feature_importances[feature_importances['importance_gain'] > 0])
    needed_cols = free_list.copy()
    print('total_features_without_zero: ' ,total_features_without_zero)
    for i in range(total_features_without_zero):
        print(i)
        if len(needed_cols) >= top_k:
            break
        if feature_importances.iloc[i]['importance_gain'] < MIN_FEAT_IMPORTANT:
            break
        cur_col = feature_importances.iloc[i]['features']
        if cur_col in needed_cols:
            continue
        needed_cols.append(cur_col)

    ops['low_importance'] = list(set(list(X_train.columns)) - set(needed_cols))
    ops['needed_cols'] = needed_cols
    data.drop(ops['low_importance'], axis=1, inplace=True)
    #for c in self.ops['low_importance']:
    #    print(c)
    log('%d features with low importance.\n' % len(ops['low_importance']))
    return data