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
from tensorflow.keras.layers import Input, Lambda, Dense, concatenate, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
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

# Random seeds
np.random.seed(256) # For Numpy
random_seed = 42 # For SKlearn
set_seed(42) # For tensorflow

# Training settings
n_shuffles = 10
n_cv_folds = 5
test_data_ratio = 0.2

# Feature selection settings
f_sele = True
f_percent = 30

# Phenotype distribution
# CD': 731, 'no': 335, 'UC': 219, 'IC': 73,
phenotype = "CD"


try:
    os.mkdir("output_data/")
except OSError:
    print("Directory already exists")

# Loading data
# Training data
X = pd.read_csv("input_data/crohns/TN_crohn.unoise3.ASV.table_FINAL.txt", sep = "\t", index_col = 0)
X = X.transpose()
labels = np.load("input_data/crohns/labels.npy", allow_pickle = True)
ids = np.load("input_data/crohns/IDs.npy", allow_pickle = True)

counts = Counter(labels)
print(counts)


# Delete control
drop_id = ids[np.where(labels == "control")[0][0]]
labels = np.delete(labels, np.where(labels == "control")[0][0])
X = X.drop(drop_id)

# Balancing the dataset 50/50 case control
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

# Encoding the labels to be binary 
enc = OneHotEncoder(sparse = False)
y = enc.fit_transform(labels.astype(str).reshape(-1, 1))
y = y[:, 0]

# Filter out Eukaryotes as they are most likely artefacts
eukaryote_zotus = tax_table.index[(tax_table["Kingdom"] == "Eukaryota").values]
X = X.drop(eukaryote_zotus, 1)

# Function used in GS CV. Groups all run results by the parameters used in the run. Used for finding the best parameters in a grid search.
def group_by_params(df, n_combinations, n_scores = 2):
    mean_score_df = pd.DataFrame(index = df.index[:n_scores], columns = [comb + 1 for comb in range(n_combinations)])
    params = []
    for i in range(n_combinations):
        filtered = df.filter(regex=("CV_.*_GS_{}$".format(i+1)))
        params.append(filtered.loc['Params', "CV_1_GS_{}".format(i+1)])
        mean_score_df[i+1] = filtered.loc[filtered.index[:n_scores]].mean(axis = 1).values
    mean_score_df.loc['Params'] = params
    return mean_score_df, params 

# Function for building the neural net.
def build_fn(input_dimensions, learning_rate, activation, n_branch_outputs, dropout, optimizer, reg = 0.01):
    
    dim_z = input_dimensions
    
    # Inputs
    inputZotus = Input(shape=(dim_z,), name = "input_zotu")

    # Zotu branch
    z = Dense(64, kernel_regularizer=l2(reg), bias_regularizer=l2(reg), activation=activation)(inputZotus)
    z = Dense(32, kernel_regularizer=l2(reg), bias_regularizer=l2(reg), activation=activation)(z)
    z = Dense(n_branch_outputs, kernel_regularizer=l2(reg), bias_regularizer=l2(reg), activation=activation)(z)
    z = Model(inputs=inputZotus, outputs=z)
    # Concatenating outputs
    x = Dropout(dropout)(z)
    #x = BatchNormalization()(x)
    # Fully connected layers
    #x = Dense(32, kernel_regularizer=l2(reg), bias_regularizer=l2(reg), activation=activation)(x)
    x = Dense(1, activation="sigmoid")(x)
    # Final model
    #optimizer = SGD(lr = lr_schedule)
    model = Model(inputs=z.input, outputs = x)
    if optimizer == "adam":
        model.compile(optimizer = Adam(lr = learning_rate), loss = MeanSquaredError())
    elif optimizer == "sgd":
        #model.compile(optimizer = SGD(learning_rate = lr_schedule), loss = MeanSquaredError())
        model.compile(optimizer = SGD(learning_rate = learning_rate), loss = MeanSquaredError())
    return model

# Creating empty lists for appending ROC AUC properties
list_auc=[]
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
# Instantiating the stratified shuffle split object for the main shuffle loop
StratShufSpl=StratifiedShuffleSplit(n_shuffles,
                                    test_size = test_data_ratio, 
                                    random_state = random_seed)

# Some matplotlib settings that I think are mostly redundant
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
# Parameter grid for the grid search
param_grid = {"epochs": [5000],
              "validation_split": [0.1],
              "callbacks": [[EarlyStopping(patience = 20)]],
              "batch_size": [64],
              "optimizer": ["sgd"],
              "learning_rate": [0.1],
              "activation": ["elu"],
              "n_branch_outputs": [4],
              "dropout": [0, 0.125],
              "reg": [0.001, 0.005]}

# Specific keys that are used for building the model. They parameter dict is later split, because one part is used for building the model and the other part is used for fitting the model.
build_keys = ["learning_rate", "activation", "n_branch_outputs", "dropout", "reg", "optimizer"]

param_iterator = ParameterGrid(param_grid)
grid_size = len(param_iterator)

# The cv_list is later used as columns for a dataframe keeping track of the results of all parameter combinations    
cv_list = ["CV_{}_GS_{}".format(str(i+1), str(j+1)) for i in range(n_cv_folds) for j in range(grid_size)]
# A dataframe that will store the best parameters for every shuffle
best_param_df = pd.DataFrame(index = list(range(1, n_shuffles + 1)), columns = param_grid.keys())

# Beginning the shuffle loop
for train_val_idx, test_idx in StratShufSpl.split(X, y):
    shuffle_counter += 1
    print("--------------------------------")
    print("Beginning shuffle {}".format(shuffle_counter))
    print("--------------------------------")
    print("\n")
    
    # Creating a dataframe that keeps track of the AUC, MSE and parameters for all parameter combinations in every fold of the cross validation
    stat_df = pd.DataFrame(index = ["AUC", "MSE", "Params"], columns = cv_list)
    
    # Splitting the dataset in train+val and test
    X_train_val, y_train_val = X.iloc[train_val_idx], y[train_val_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]
    
    # Performating univariate feature selection
    if f_sele:
        print("Univariately selecting features\n")
        selector = SelectPercentile(f_classif, f_percent)
        selector.fit(X_train_val, y_train_val)
        support = selector.get_support(indices = True)
        X_train_val = X_train_val.iloc[:, support]
        X_test = X_test.iloc[:, support]
    
    # Instantiating the kfold cv object 
    kfold_cv = KFold(n_splits = n_cv_folds, random_state = random_seed)
    n_candidates = grid_size
    cv_fold = 0

    # Beginning the the cross validation loop
    for train_idx, val_idx in kfold_cv.split(X_train_val, y_train_val):
        cv_fold += 1
        print("--------------------------------")
        print("Beginning cross validation fold {} in shuffle {} with {} candidates".format(cv_fold, shuffle_counter, n_candidates))
        print("--------------------------------")
        print("\n")
        # Spltting train+val in train and val
        X_train, y_train = X_train_val.iloc[train_idx], y_train_val[train_idx]
        X_val, y_val = X_train_val.iloc[val_idx], y_train_val[val_idx]
        
        print("Fitting scalers")
        X_train = np.log(X_train+1)
        X_val = np.log(X_val+1)
        scaler = MinMaxScaler()
        scaler.fit(X_train[i])
        X_train = np.clip(scaler.transform(X_train), 0, 1)
        X_val = np.clip(scaler.transform(X_val), 0, 1)
        dimensions = train_dfs[i].shape[-1]
        print("Finished scaling")
        
        gs_it = 0
        # Iterating over the parameter grid
        for params in param_iterator:
            gs_it += 1
            print("--------------------------------")
            print("CV fold: {} \nShuffle: {} \nCandidate: {} out of {} \nParameters:\n{}".format(cv_fold, shuffle_counter, gs_it, n_candidates, params))
            print("--------------------------------")
            print("\n")
            saved_params = params.copy()
            build_params = {key: params.pop(key) for key in build_keys}
            # Instantiating model
            model = build_fn(dimensions, **build_params)
            # Fitting model
            model.fit(X_train, y_train, **params)
            # Making predictions on the validation set
            y_pred_val = model.predict(X_val)
            # Scoring the predictions
            val_mse = mean_squared_error(y_val, y_pred_val)
            val_auc = roc_auc_score(y_val, y_pred_val)
            print("\n")
            print("Val AUC: {}".format(val_auc))
            print("\n")
            
            # Storing the results in the dataframe used for finging the best parameters in the current shuffle
            stat_df.loc["MSE"]["CV_{}_GS_{}".format(cv_fold, gs_it)] = val_mse
            stat_df.loc["AUC"]["CV_{}_GS_{}".format(cv_fold, gs_it)] = val_auc
            stat_df.loc["Params"]["CV_{}_GS_{}".format(cv_fold, gs_it)] = saved_params
    
    # Fitting on entire train+val
    print("--------------------------------")
    print("Refitting on train + val and evaluating on test")
    print("--------------------------------")
    print("\n")
        
    print("Fitting scalers")
    X_train_val = np.log(X_train_val+1)
    X_test = np.log(X_test+1)
    scaler = MinMaxScaler()
    scaler.fit(X_train_val)
    X_train_val = np.clip(scaler.transform(train_val_dfs[i]), 0, 1)
    X_test = np.clip(scaler.transform(X_test), 0, 1)
    print("Finished scaling")
    
    # Finding the average score over the folds for every parameter combination
    grouped_df, params = group_by_params(stat_df, n_combinations = grid_size)
    #grouped_df.to_csv('output_data/grouped_validation_results_{}.csv'.format(shuffle_counter))
    # Finding the best parameters based on the highest average AUC
    best_score_index = np.argmin(list(grouped_df.loc['AUC']))
    best_params = params[best_score_index]
    print('Best found parameters:\n')
    print(best_params)

    # Storing the best parameters for this shuffle
    for key, value in best_params.items():
        best_param_df.loc[shuffle_counter, key] = value
    
    # Fitting on the entire train+val
    build_params = {key: best_params.pop(key) for key in build_keys}
    # Instatiating model
    model = build_fn(dimensions, **build_params)
    # Fitting model
    model.fit(X_train_val, y_train_val, **best_params)
    # Makign predictions on test
    y_pred_test = model.predict(test_inputs)
    # Scoring on test
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_auc = roc_auc_score(y_val, y_pred_val)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_test, pos_label = 1)
    auc_roc1 = auc(fpr, tpr)
    list_auc.append(auc_roc1)

    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    aucs.append(auc_roc1)
    plt.plot(fpr, tpr, lw=1, alpha=0.3)

best_param_df.to_csv("output_data/MLP/MLP_best_params.csv")

# Plotting the ROC AUC
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

print("Mean AUC: {}".format(mean_auc))
