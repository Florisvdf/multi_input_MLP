import datetime
import os

import numpy as np
import pandas as pd
import json

import multiprocessing
from sklearn.base import is_classifier, clone
from sklearn.utils import indexable, check_random_state, safe_indexing
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils._joblib import Parallel, delayed
from sklearn.metrics import check_scoring, r2_score, roc_auc_score, explained_variance_score
from sklearn.model_selection._validation import _permutation_test_score, _shuffle
from sklearn.model_selection._split import check_cv
from collections import OrderedDict


def calculate_feature_permutations(X_train, X_test, y_train, y_test, estimator, importance_perms,
                                   random_state, score_name='r2_score'):

    np.random.seed(random_state)

    estimator_ = clone(estimator)
    importances_permutation = importance_perms.copy()
    estimator_.fit(X_train, y_train.ravel())
    if hasattr(estimator_, 'best_estimator_'):
        if hasattr(estimator_.best_estimator_, 'feature_importances_'):
            estimator_importances = estimator_.best_estimator_.feature_importances_
        else:
            estimator_importances = np.zeros(X_test.shape[1])
    elif hasattr(estimator_, 'feature_importances_'):
        estimator_importances = estimator_.feature_importances_
    else:
        estimator_importances = np.zeros(X_test.shape[1])
    print(f'\n\n\n\n\n estimator importances: {estimator_importances}\n\n\n')
    y_pred = estimator_.predict(X_test)
    score = eval(f'{score_name}')(y_test, y_pred)
    print(f'SCORE: {score}')
    test_inds = np.arange(X_test.shape[0])
    np.random.shuffle(test_inds)
    train_inds = np.arange(X_train.shape[0])
    np.random.shuffle(train_inds)

    for i, feat_1 in enumerate(X_test.columns):
        # estimator_ = clone(estimator)

        X_test_ = X_test.copy()
        X_test_.values[:,i] = X_test_.iloc[test_inds, i]
        X_train_ = X_train.copy()
        X_train_.iloc[:, i] = X_train_.iloc[train_inds, i]
        # estimator_.fit(X_train_, y_train.ravel())
        y_pred_random = estimator_.predict(X_test_)

        score_ = eval(f'{score_name}')(y_test, y_pred_random)

        importances_permutation[feat_1] = score - score_

        print(f'SCORE DIFF {feat_1}: {score - score_}')

    return score, importances_permutation, estimator_importances


def permutation_importances(X, y, estimator, splitter, score='r2_score'):

    n_splits = splitter.get_n_splits()

    num_cores = multiprocessing.cpu_count()

    """Prepare data structures"""

    importances_perms = OrderedDict()

    for feat in X.columns:

        importances_perms[feat] = []

    random_states = np.arange(n_splits)

    splits = [(train, test) for train, test in splitter.split(X, y)]

    """Run the permutation importance algorithm"""

    r = Parallel(n_jobs=num_cores)(
        delayed(calculate_feature_permutations)(X.iloc[splits[i][0], :], X.iloc[splits[i][1], :],
                                                y[splits[i][0]], y[splits[i][1]], estimator, importances_perms,
                                                random_states[i], score)
        for i in range(len(random_states)))
    original_scores, perm_scores, estimator_importances = zip(*r)
    estimator_importances = np.vstack(estimator_importances)
    perm_scores = pd.DataFrame(perm_scores)

    return original_scores, perm_scores, estimator_importances


class EvenSplitter:

    '''
    Splits data into train test with equal ratios of two classes
    '''

    def __init__(self, train_size, n_splits, random_state):
        self.train_size = train_size

        self.n_splits = n_splits

        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):

        return self.n_splits

    def split(self, X, y, groups=None):
        y = y.squeeze()
        print(len(np.argwhere(y == 0).flatten()))
        print(len(np.argwhere(y == 1).flatten()))
        pos = [np.argwhere(y == 0).flatten(), np.argwhere(y == 1).flatten()]

        lens = [len(pos[0]), len(pos[1])]

        small_class = int(np.argmin(lens))

        n_small = lens[small_class]

        for split in range(self.n_splits):
            np.random.seed(split + self.random_state)
            selected_large = np.random.choice(pos[1 - small_class], n_small, replace=False)
            small_pos = pos[small_class]
            np.random.seed(split + self.random_state)
            np.random.shuffle(small_pos)

            train = np.r_[small_pos[:int(self.train_size * len(small_pos))],
                          selected_large[:int(self.train_size * len(selected_large))]]
            np.random.seed(split + self.random_state)
            np.random.shuffle(train)
            test = np.r_[small_pos[int(self.train_size * len(small_pos)):],
                         selected_large[int(self.train_size * len(selected_large)):]]
            np.random.seed(split + self.random_state)
            np.random.shuffle(test)

            yield (train, test)