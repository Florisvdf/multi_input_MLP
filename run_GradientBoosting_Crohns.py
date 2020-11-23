import os
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit,ShuffleSplit,StratifiedKFold,KFold
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import roc_curve, auc, mean_squared_error, roc_auc_score

import matplotlib
import matplotlib.pylab as plt
from scipy import interp

import seaborn as sns

import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode()
import shutil


np.random.seed(256)
random_seed = 42

n_shuffles = 10
n_cv_folds = 5
test_data_ratio = 0.2

f_sele = True
f_percent = 30

try:
    os.mkdir("output_data/")
except OSError:
    print("Directory already exists")


# Training data
X = pd.read_csv("input_data/crohns/TN_crohn.unoise3.ASV.table_FINAL.txt", sep = "\t", index_col = 0)
X = X.transpose()
labels = np.load("input_data/crohns/labels.npy", allow_pickle = True)
ids = np.load("input_data/crohns/IDs.npy", allow_pickle = True)
# Taxa data
tax_table = pd.read_csv("input_data/crohns/ASV_taxonomy_TN_crohn_SILVA_138.csv", index_col = "ASV_ID")


# Delete control
drop_id = ids[np.where(labels == "control")[0][0]]
labels = np.delete(labels, np.where(labels == "control")[0][0])
X = X.drop(drop_id)


counts = Counter(labels)
len_healthy = counts["no"]
cd_idx = np.random.choice(np.where(labels == "CD")[0], len_healthy, replace = False)
healthy_idx = np.where(labels == "no")[0]
joined_idx = np.concatenate((cd_idx, healthy_idx))
X = X.iloc[joined_idx]
labels = labels[joined_idx]


enc = OneHotEncoder(sparse = False)
y = enc.fit_transform(labels.astype(str).reshape(-1, 1))
y = y[:, 0]


# Filter out Eukaryotes?
eukaryote_zotus = tax_table.index[(tax_table["Kingdom"] == "Eukaryota").values]
X = X.drop(eukaryote_zotus, 1)


def join_header(pref, header):
    try:
        header = pref + header
    except TypeError:
        header = pref + str(header)
    return header



def zotus_to_tax_dfs(zotu_abundances, tax_table):
    dataframes = []
    prefixes = ["k__", "p__", "c__", "o__", "f__", "g__", "s__"]
    ranks = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    for prefix, rank in zip(prefixes, ranks):
        headers = tax_table[rank].unique()
        headers = [join_header(prefix, header) for header in headers]
        #headers = [rank + str(header) for header in headers]
        rows = range(len(zotu_abundances))
        dataframe = pd.DataFrame(columns = headers, index = rows)
        dataframe = dataframe.fillna(0)
        for zotu_name in zotu_abundances.columns:
            if zotu_name not in tax_table.index:
                continue
            try:
                tax = prefix + tax_table.loc[zotu_name][rank]
            except TypeError:
                tax = prefix + str(tax_table.loc[zotu_name][rank])
            summed_series = dataframe[tax].values + zotu_abundances[zotu_name].values
            dataframe[tax] = summed_series
        #dataframe.reset_index(drop=True, inplace=True)
        dataframes.append(dataframe)
    dataframes.append(zotu_abundances)
    return dataframes


list_auc=[]

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
StratShufSpl=StratifiedShuffleSplit(n_shuffles,
                                    test_size = test_data_ratio, 
                                    random_state = random_seed)

plt.style.use('ggplot')
params = {'legend.fontsize': 'medium',
          'figure.figsize': (10, 10),
          'axes.labelsize': 'medium',
          'axes.titlesize':'medium',
          'xtick.labelsize':'medium',
          'ytick.labelsize':'medium'}
plt.rcParams.update(params)
plt.rcParams["font.family"] = "sans-serif"
shuffle_counter = 0


for samples,test in StratShufSpl.split(X,y):
    shuffle_counter += 1
    X_train, y_train= X.iloc[samples], y[samples]
    X_test, y_test = X.iloc[test], y[test]
    
    if f_sele:
        print("Univariately selecting features\n")
        selector = SelectPercentile(f_classif, f_percent)
        selector.fit(X_train, y_train)
        support = selector.get_support(indices = True)
        X_train = X_train.iloc[:, support]
        X_test = X_test.iloc[:, support]
    
    print("Splitting data in different taxanomic ranks\n")
    train_dfs = zotus_to_tax_dfs(X_train, tax_table)
    test_dfs = zotus_to_tax_dfs(X_test, tax_table)
    for df in train_dfs:
        df.reset_index(drop=True, inplace=True)
    X_train = pd.concat(train_dfs, axis = 1)
    for df in test_dfs:
        df.reset_index(drop=True, inplace=True)
    X_test = pd.concat(test_dfs, axis = 1)  
    
    param_grid = {"learning_rate": [0.1, 0.01, 0.001],
                  "n_estimators": [400],
                  "criterion": ["friedman_mse", "mse", "mae"]}
    
    clf = GradientBoostingClassifier()
    skf = StratifiedKFold(n_splits = n_cv_folds, random_state = random_seed)
    grid_search = GridSearchCV(clf, param_grid, scoring='roc_auc',
                               cv=skf, n_jobs=-1, verbose = 1,
                               refit = True, iid = False)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print("Best score: {}".format(grid_search.best_score_))
    
    try:
        y_pred_test = best_model.predict_proba(X_test)
    except AttributeError:
        y_pred_test = best_model.predict(X_test)
    if len(y_pred_test.shape) == 2:
        y_pred_test = y_pred_test[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_test, pos_label=1)
    auc_roc1 = auc(fpr, tpr)
    list_auc.append(auc_roc1)

    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    aucs.append(auc_roc1)
    plt.plot(fpr, tpr, lw=1, alpha=0.3)
    list_auc.append(auc_roc1)

    aucs.append(auc_roc1)
    
    print('')
    print(('Shuffle %i is done!!!' %shuffle_counter))


plt.plot([0, 1], [0, 1], linestyle='--', lw = 1, color='r',
             label="Random guess", alpha = .8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw = 1, alpha = .8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color = 'grey', alpha = .2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('output_data/GradientBoosting_auc_avg.pdf', bbox_inches='tight')
plt.show()
