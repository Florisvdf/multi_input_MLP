import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, roc_auc_score, explained_variance_score
from sklearn.model_selection import ShuffleSplit, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import scale
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_classif
import os
import os.path as op
from xgboost import XGBRegressor

from CID import CIDGmm
from permutation_importance import permutation_importances, EvenSplitter
from utils import permutation_single_and_pair_importances


def mi_importance(perm_imps, covered):
    percentage_uncovered = 1 - covered

    perm_imps_mi = perm_imps / percentage_uncovered

    return perm_imps_mi


def plot_mi_imps_breast_cancer(X, y, data_str, n_splits=20, perm_imps=None, clf_imps=None, original_scores=None):

    train_size = 0.9
    random_state = 0

    # cov = pd.DataFrame(np.cov(np.hstack([X.values, y]).T),
    #                    columns=np.hstack([X.columns, 'y']),
    #                    index=np.hstack([X.columns, 'y']))
    # precision = pd.DataFrame(np.linalg.pinv(cov),
    #                          columns=np.hstack([X.columns, 'y']),
    #                          index=np.hstack([X.columns, 'y']))

    n_bins = 50

    import matplotlib as mpl
    mpl.use('Qt5Agg')

    splitter = ShuffleSplit(train_size=train_size, test_size=1 - train_size,
                            random_state=random_state, n_splits=n_splits)

    np.random.seed(random_state)

    X = pd.DataFrame(scale(X), columns=X.columns)
    # y=scale(y)
    param_grid_xgboost = {'max_depth': [2, 5, 10],
                          'learning_rate': [0.01, 0.1],
                          'n_estimators': [100, 300, 800, 1000, ],
                          'min_child_weight': [1, 5, ],
                          'gamma': [0.5, 2],
                          'subsample': [0.6, 0.8],
                          'colsample_bytree': [0.6, 0.8, 1.0],
                          }
    
    precision = np.linalg.pinv(np.cov(scale(np.hstack([X, y])).T))

    splitter = EvenSplitter(train_size, n_splits, random_state)
    param_grid_xgb = param_grid_xgboost
    fit_params = {'eval_metric': 'roc_auc',
                  'verbose': False,
                  'early_stopping_rounds': 50}
    model= XGBRegressor(n_jobs=1, objective='reg:squarederror', tree_method='auto')
    skf_xgb = EvenSplitter(train_size=0.9, n_splits=5, random_state=0)
    random_search = RandomizedSearchCV(model, param_grid_xgb,
                                       n_jobs=7,
                                       cv=skf_xgb, scoring='explained_variance',
                                       verbose=1, n_iter=10, random_state=0,
                                       refit=True)

    cid = CIDGmm(data=X, y=y, n_bins=n_bins, scale_data=True, discretize=True, data_std_threshold=3,
                 empirical_mi=False, redund_correction=True,
                 kwargs={'max_iter': 5000, 'alphas': [0.0001, 0.001, 0.01, 0.1, 0.3], 'tol': 1e-4})

    covered, mi_y = cid.fit(n_samples=1)

    if perm_imps is None and clf_imps is None and original_scores is None:

        original_scores, importances_permutation, importances_clf = \
            permutation_importances(X, y, random_search, splitter, score='roc_auc_score')

        np.save(f'perm_imps_{data_str}_n_splits_{n_splits}', importances_permutation)
        np.save(f'clf_imps_{data_str}_n_splits_{n_splits}', importances_clf)
        np.save(f'original_scores_{data_str}_n_splits_{n_splits}', original_scores)

    else:

        importances_permutation = perm_imps
        importances_clf = clf_imps
        original_scores = original_scores

    print(f'\n\n SCORE {np.mean(original_scores)}\n\n\n')

    importances_permutation = np.array(importances_permutation)

    mi_imps = mi_importance(importances_permutation, covered)

    median_mi = np.median(mi_imps, axis=0)
    median_mi /= np.sum(median_mi)

    order_mi_imps = np.argsort(median_mi)
    mi_df = pd.DataFrame(mi_imps, columns=X.columns)

    fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
    ax1.boxplot(mi_df.values[:, order_mi_imps], vert=False,
                labels=mi_df.columns[order_mi_imps])
    ax1.set_title('Covered Information Disentanglement')
    fig.tight_layout()
    plt.savefig(f'{data_str}_feat_imp_ranking_plot')
    plt.show()

##############################################################################
# Real data test
##############################################################################

n_splits = 20

plot_imps = True

quantile_transform = True

load_pre_computed_imps = False

X = np.load('X_mvn.npy')
y = np.load('y_mvn.npy')

vars_names = [f'x_{i}' for i in range(X.shape[1])]

y_ = y.reshape(-1, 1)

X_ = pd.DataFrame(X, columns=vars_names)

data_str = 'mvn'

if quantile_transform:
    q = QuantileTransformer(output_distribution='normal')
    X_ = pd.DataFrame(q.fit_transform(X_), columns=X_.columns)

    if len(np.unique(y_)) > 2:
        y_ = q.fit_transform(y_)

if plot_imps:

    if load_pre_computed_imps:

        importances_permutation = np.load(f'perm_imps_{data_str}_n_splits_{n_splits}.npy')
        importances_clf = np.load(f'clf_imps_{data_str}_n_splits_{n_splits}.npy')
        original_scores = np.load(f'original_scores_{data_str}_n_splits_{n_splits}.npy')

    else:
        importances_permutation = None
        importances_clf = None
        original_scores = None

    plot_mi_imps_breast_cancer(X_, y_, data_str, n_splits=n_splits,
                               clf_imps=importances_clf,
                               perm_imps=importances_permutation,
                               original_scores=original_scores)
