import os
from collections import Counter
from copy import copy

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import StratifiedShuffleSplit,ShuffleSplit,StratifiedKFold,KFold
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import roc_curve, auc, mean_squared_error, roc_auc_score
from sklearn.preprocessing import scale, minmax_scale, MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Lambda, Dense, concatenate, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.random import set_seed
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

import matplotlib
import matplotlib.pylab as plt
from scipy import interp

import pandas as pd
import seaborn as sns

import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode()
import shutil

from CID import *


np.random.seed(256)
random_seed = 42

set_seed(42)


n_shuffles = 10
n_cv_folds = 5
test_data_ratio = 0.2

f_sele = True
f_percent = 30

# CD': 731, 'no': 335, 'UC': 219, 'IC': 73,
phenotype = "CD"


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


counts = Counter(labels)
print(counts)


# Delete control
drop_id = ids[np.where(labels == "control")[0][0]]
labels = np.delete(labels, np.where(labels == "control")[0][0])
X = X.drop(drop_id)


len_healthy = counts["no"]
len_disease = counts[phenotype]
if len_healthy < len_disease:
    disease_idx = np.random.choice(np.where(labels == phenotype)[0], len_healthy, replace = False)
    healthy_idx = np.where(labels == "no")[0]
else:
    healthy_idx = np.random.choice(np.where(labels == "no")[0], len_disease, replace = False)
    disease_idx = np.where(labels == phenotype)[0]
joined_idx = np.concatenate((disease_idx, healthy_idx))
X = X.iloc[joined_idx]
labels = labels[joined_idx]


enc = OneHotEncoder(sparse = False)
y = enc.fit_transform(labels.astype(str).reshape(-1, 1))
y = y[:, 0]


# Filter out Eukaryotes?
eukaryote_zotus = tax_table.index[(tax_table["Kingdom"] == "Eukaryota").values]
X = X.drop(eukaryote_zotus, 1)


def zotus_to_tax_dfs(zotu_abundances, tax_table):
    dataframes = []
    for rank in ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]:
        headers = tax_table[rank].unique()
        rows = range(len(zotu_abundances))
        dataframe = pd.DataFrame(columns = headers, index = rows)
        dataframe = dataframe.fillna(0)
        for zotu_name in zotu_abundances.columns:
            if zotu_name not in tax_table.index:
                continue
            tax = tax_table.loc[zotu_name][rank]
            summed_series = dataframe[tax].values + zotu_abundances[zotu_name].values
            dataframe[tax] = summed_series
        dataframes.append(dataframe)
    dataframes.append(zotu_abundances)
    return dataframes


def group_by_params(df, n_combinations, n_scores = 2):
    mean_score_df = pd.DataFrame(index = df.index[:n_scores], columns = [comb + 1 for comb in range(n_combinations)])
    params = []
    for i in range(n_combinations):
        filtered = df.filter(regex=("CV_.*_GS_{}$".format(i+1)))
        params.append(filtered.loc['Params', "CV_1_GS_{}".format(i+1)])
        mean_score_df[i+1] = filtered.loc[filtered.index[:n_scores]].mean(axis = 1).values
    mean_score_df.loc['Params'] = params
    return mean_score_df, params 


def mi_importance(perm_imps, covered):
    percentage_uncovered = 1 - covered

    perm_imps_mi = perm_imps / percentage_uncovered

    return perm_imps_mi


def build_fn(input_dimensions, learning_rate, activation, n_branch_outputs, dropout, optimizer, reg = 0.01):
    
    dim_k, dim_p, dim_c, dim_o, dim_f, dim_g, dim_s, dim_z = input_dimensions
    
    # Inputs
    inputKingdom = Input(shape=(dim_k,), name = "input_kingdom")
    inputPhylum = Input(shape=(dim_p,), name = "input_phylum")
    inputClass = Input(shape=(dim_c,), name = "input_class")
    inputOrder = Input(shape=(dim_o,), name = "input_order")
    inputFamily = Input(shape=(dim_f,), name = "input_family")
    inputGenus = Input(shape=(dim_g,), name = "input_genus")
    inputSpecies = Input(shape=(dim_s,), name = "input_species")
    inputZotus = Input(shape=(dim_z,), name = "input_zotu")

    # Kingdom branch
    k = Dense(n_branch_outputs, kernel_regularizer=l2(reg), bias_regularizer=l2(reg), activation=activation)(inputKingdom)
    k = Model(inputs=inputKingdom, outputs=k)
    # Phylum branch
    p = Dense(16, kernel_regularizer=l2(reg), bias_regularizer=l2(reg), activation=activation)(inputPhylum)
    p = Dense(n_branch_outputs, kernel_regularizer=l2(reg), bias_regularizer=l2(reg), activation=activation)(p)
    p = Model(inputs=inputPhylum, outputs=p)
    # Class branch
    c = Dense(32, activation=activation)(inputClass)
    c = Dense(n_branch_outputs, kernel_regularizer=l2(reg), bias_regularizer=l2(reg), activation=activation)(c)
    c = Model(inputs=inputClass, outputs=c)
    # Order branch
    o = Dense(32, kernel_regularizer=l2(reg), bias_regularizer=l2(reg), activation=activation)(inputOrder)
    o = Dense(n_branch_outputs, kernel_regularizer=l2(reg), bias_regularizer=l2(reg), activation=activation)(o)
    o = Model(inputs=inputOrder, outputs=o)
    # Family branch
    f = Dense(64, kernel_regularizer=l2(reg), bias_regularizer=l2(reg), activation=activation)(inputFamily)
    f = Dense(n_branch_outputs, kernel_regularizer=l2(reg), bias_regularizer=l2(reg), activation=activation)(f)
    f = Model(inputs=inputFamily, outputs=f)
    # Genus branch
    g = Dense(64, kernel_regularizer=l2(reg), bias_regularizer=l2(reg), activation=activation)(inputGenus)
    g = Dense(n_branch_outputs, kernel_regularizer=l2(reg), bias_regularizer=l2(reg), activation=activation)(g)
    g = Model(inputs=inputGenus, outputs=g)
    # Species branch
    s = Dense(64, kernel_regularizer=l2(reg), bias_regularizer=l2(reg), activation=activation)(inputSpecies)
    s = Dense(n_branch_outputs, kernel_regularizer=l2(reg), bias_regularizer=l2(reg), activation=activation)(s)
    s = Model(inputs=inputSpecies, outputs=s)
    # Zotu branch
    z = Dense(64, kernel_regularizer=l2(reg), bias_regularizer=l2(reg), activation=activation)(inputZotus)
    z = Dense(32, kernel_regularizer=l2(reg), bias_regularizer=l2(reg), activation=activation)(z)
    z = Dense(n_branch_outputs, kernel_regularizer=l2(reg), bias_regularizer=l2(reg), activation=activation)(z)
    z = Model(inputs=inputZotus, outputs=z)
    # Concatenating outputs
    combined = concatenate([k.output, p.output, c.output, o.output, f.output, g.output, s.output, z.output])
    x = Dropout(dropout)(combined)
    # Fully connected layers
    x = Dense(32, kernel_regularizer=l2(reg), bias_regularizer=l2(reg), activation=activation)(x)
    x = Dense(1, activation="sigmoid")(x)
    # Final model
    model = Model(inputs=[k.input, p.input, c.input, o.input, f.input, g.input, s.input, z.input], outputs = x)
    if optimizer == "adam":
        model.compile(optimizer = Adam(lr = learning_rate), loss = MeanSquaredError())
    elif optimizer == "sgd":
        model.compile(optimizer = SGD(lr = learning_rate), loss = MeanSquaredError())
    return model


list_auc=[]

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
StratShufSpl=StratifiedShuffleSplit(n_shuffles,
                                    test_size = test_data_ratio, 
                                    random_state = random_seed)

taxa_fis = []
list_covered = []

plt.style.use('ggplot')
params = {'legend.fontsize': 'medium',
          'figure.figsize': (10, 10),
          'axes.labelsize': 'medium',
          'axes.titlesize':'medium',
          'xtick.labelsize':'medium',
          'ytick.labelsize':'medium'}
plt.rcParams.update(params)
plt.rcParams["font.family"] = "sans-serif"
#"cursive"
shuffle_counter=0
param_grid = {"epochs": [5000],
              "validation_split": [0.1, 0.2],
              #"validation_split": [0.2],
              "callbacks": [[EarlyStopping(patience = 20)]],
              "batch_size": [64], 
              #"batch_size": [32],
              #"optimizer": ["adam", "sgd"],
              "optimizer": ["sgd"],
              #"learning_rate": [0.001, 0.005],
              "learning_rate": [0.5, 0.1, 0.05],
              "activation": ["elu"],
              "n_branch_outputs": [4],
              "dropout": [0, 0.125],
              #"dropout": [0],
              "reg": [0.005]}
    
build_keys = ["learning_rate", "activation", "n_branch_outputs", "dropout", "reg", "optimizer"]

param_iterator = ParameterGrid(param_grid)
grid_size = len(param_iterator)
    
cv_list = ["CV_{}_GS_{}".format(str(i+1), str(j+1)) for i in range(n_cv_folds) for j in range(grid_size)]
best_param_df = pd.DataFrame(index = list(range(1, n_shuffles + 1)), columns = param_grid.keys())

for train_val_idx, test_idx in StratShufSpl.split(X, y):
    shuffle_counter += 1
    print("--------------------------------")
    print("Beginning shuffle {}".format(shuffle_counter))
    print("--------------------------------")
    print("\n")
    build_keys = ["learning_rate", "activation", "n_branch_outputs", "dropout", "reg", "optimizer"]
    
    stat_df = pd.DataFrame(index = ["AUC", "MSE", "Params"], columns = cv_list)
    
    X_train_val, y_train_val = X.iloc[train_val_idx], y[train_val_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]
    
    if f_sele:
        print("Univariately selecting features\n")
        selector = SelectPercentile(f_classif, f_percent)
        selector.fit(X_train_val, y_train_val)
        support = selector.get_support(indices = True)
        X_train_val = X_train_val.iloc[:, support]
        X_test = X_test.iloc[:, support]
    
    #StratShufSplVal = StratifiedShuffleSplit(n_cv_folds,
    #                                         test_size = test_data_ratio, 
    #                                         random_state = random_seed)
    kfold_cv = KFold(n_splits = n_cv_folds, random_state = random_seed)
    n_candidates = grid_size
    cv_fold = 0
    #for train_idx, val_idx in StratShufSplVal.split(X_train_val, y_train_val):
    for train_idx, val_idx in kfold_cv.split(X_train_val, y_train_val):
        cv_fold += 1
        print("--------------------------------")
        print("Beginning cross validation fold {} in shuffle {} with {} candidates".format(cv_fold, shuffle_counter, n_candidates))
        print("--------------------------------")
        print("\n")
        X_train, y_train = X_train_val.iloc[train_idx], y_train_val[train_idx]
        X_val, y_val = X_train_val.iloc[val_idx], y_train_val[val_idx]

        
        print("Splitting data in different taxanomic ranks\n")
        train_dfs = zotus_to_tax_dfs(X_train, tax_table)
        val_dfs = zotus_to_tax_dfs(X_val, tax_table)
        
        print("Fitting scalers")
        dimensions = []
        train_inputs = []
        val_inputs = []
        for i in range(len(train_dfs)):
            train_dfs[i] = np.log(train_dfs[i]+1)
            val_dfs[i] = np.log(val_dfs[i]+1)
            scaler = MinMaxScaler()
            scaler.fit(train_dfs[i])
            train_inputs.append(np.clip(scaler.transform(train_dfs[i]), 0, 1))
            val_inputs.append(np.clip(scaler.transform(val_dfs[i]), 0, 1))
            dimensions.append(train_dfs[i].shape[-1])
        print("Finished scaling")
        
        gs_it = 0
        for params in param_iterator:
            gs_it += 1
            print("--------------------------------")
            print("CV fold: {} \nShuffle: {} \nCandidate: {} out of {} \nParameters:\n{}".format(cv_fold, shuffle_counter, gs_it, n_candidates, params))
            print("--------------------------------")
            print("\n")
            saved_params = params.copy()
            build_params = {key: params.pop(key) for key in build_keys}
            model = build_fn(dimensions, **build_params)
            model.fit(train_inputs, y_train, **params)
            y_pred_val = model.predict(val_inputs)
            val_mse = mean_squared_error(y_val, y_pred_val)
            val_auc = roc_auc_score(y_val, y_pred_val)
            print("\n")
            print("Val AUC: {}".format(val_auc))
            print("\n")
            
            stat_df.loc["MSE"]["CV_{}_GS_{}".format(cv_fold, gs_it)] = val_mse
            stat_df.loc["AUC"]["CV_{}_GS_{}".format(cv_fold, gs_it)] = val_auc
            stat_df.loc["Params"]["CV_{}_GS_{}".format(cv_fold, gs_it)] = saved_params
    
    # Fitting on entire train_val
    print("--------------------------------")
    print("Refitting on train + val and evaluating on test")
    print("--------------------------------")
    print("\n")
    print("Splitting data in different taxanomic ranks")
    train_val_dfs = zotus_to_tax_dfs(X_train_val, tax_table)
    test_dfs = zotus_to_tax_dfs(X_test, tax_table)
        
    print("Fitting scalers")
    train_val_inputs = []
    test_inputs = []
    for i in range(len(train_dfs)):
        train_val_dfs[i] = np.log(train_val_dfs[i]+1)
        test_dfs[i] = np.log(test_dfs[i]+1)
        scaler = MinMaxScaler()
        scaler.fit(train_val_dfs[i])
        train_val_inputs.append(np.clip(scaler.transform(train_val_dfs[i]), 0, 1))
        test_inputs.append(np.clip(scaler.transform(test_dfs[i]), 0, 1))
    print("Finished scaling")
    
    grouped_df, params = group_by_params(stat_df, n_combinations = grid_size)
    #grouped_df.to_csv('output_data/grouped_validation_results_{}.csv'.format(shuffle_counter))
    best_score_index = np.argmin(list(grouped_df.loc['AUC']))
    best_params = params[best_score_index]
    print('Best found parameters:\n')
    print(best_params)

    for key, value in best_params.items():
        best_param_df.loc[shuffle_counter, key] = value

    build_params = {key: best_params.pop(key) for key in build_keys}
    model = build_fn(dimensions, **build_params)
    model.fit(train_inputs, y_train, **best_params)
    y_pred_test = model.predict(test_inputs)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_auc = roc_auc_score(y_val, y_pred_val)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_test, pos_label = 1)
    auc_roc1 = auc(fpr, tpr)
    list_auc.append(auc_roc1)

    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    aucs.append(auc_roc1)
    plt.plot(fpr, tpr, lw=1, alpha=0.3)
    
    # Getting the taxa feature importances
    '''
    importances_permutation = {}

    inter_output_model = tf.keras.Model(model.input, model.get_layer(index = 26).output)
    res_model = Sequential()
    for layer in model.layers[27:]:
        res_model.add(layer)
    total_inputs = []
    for i in range(len(train_val_inputs)):
        total_inputs.append(np.vstack((train_val_inputs[i], test_inputs[i])))
    
    total_inter_outputs = inter_output_model.predict(total_inputs)
    test_inter_outputs = inter_output_model.predict(test_inputs)
    n_features = test_inter_outputs.shape[1]
    n_branch_features = int(n_features/8)
    input_names = ["zotu", "kingdom", "phylum", "class", "order", "family", "genus", "species"]
    feature_names = []
    for branch_name in input_names:
        for i in range(n_branch_features):
            feature_names.append(branch_name + "_" + str(i))

    #y_test = y_test.reshape(-1, 1)
    y = y.reshape(-1, 1)

    cid = CIDGmm(data = total_inter_outputs, y = y, n_bins=50, scale_data=True, discretize=True, data_std_threshold=3,
                 empirical_mi=False, redund_correction=True,
                 kwargs={'max_iter': 5000, 'alphas': [0.0001, 0.001, 0.01, 0.1, 0.3], 'tol': 1e-4})
    covered, mi_y = cid.fit(n_samples=1)
    list_covered.append(covered)
    test_inds = np.arange(test_inter_outputs.shape[0])
    np.random.shuffle(test_inds)

    for i, feat_1 in enumerate(feature_names):
        test_inter_outputs_ = test_inter_outputs.copy()
        test_inter_outputs_[:,i] = test_inter_outputs[test_inds, i]
        y_pred_random = res_model.predict(test_inter_outputs_)

        auc_ = roc_auc_score(y_test, y_pred_random)
        importances_permutation[feat_1] = test_auc - auc_

    importances_permutation = pd.DataFrame(importances_permutation, index = [0]).transpose()

    mi_imps = mi_importance(importances_permutation, covered.reshape(-1, 1))

    agg_mi_imps = mi_imps.groupby([mi_imps.index.str[:4]]).mean()
    print("Importances : {}".format(agg_mi_imps))

    taxa_fis.append(agg_mi_imps)
    '''

    print("\n")
    print("Shuffle {} is done!\n".format(shuffle_counter))

best_param_df.to_csv("output_data/MLP_best_params.csv")

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
plt.savefig('output_data/MLP_auc_avg.pdf', bbox_inches='tight')
plt.show()
