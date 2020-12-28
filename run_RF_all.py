import numpy as np
import time
import os

from sklearn.model_selection import StratifiedShuffleSplit,ShuffleSplit,StratifiedKFold,KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, mean_absolute_error,accuracy_score
from sklearn.preprocessing import scale, minmax_scale
from sklearn.feature_selection import SelectPercentile, f_classif, f_regression, VarianceThreshold, SelectFdr, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline

import matplotlib
#matplotlib.use('agg')
import matplotlib.pylab as plt
from scipy import interp

import pandas as pd
import seaborn as sns

import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode()
import shutil


startTime = time.time()
#pl.close('all')
np.random.seed(256)
random_seed = 42
data_type='RF_Class'

stability_samples_run_number=10
num_cv_folds=5
num_top_feat=15
test_data_ratio=0.2
train_ratio = 1

uni_feat_percent=30
occurrence_ratio = 0.90

extremes_frac = 0.10
f_selection_method = "f_test"
n_estimators = 200

leakage = True
result_dir = "final_all"

try:
    os.mkdir("output_data/{}".format(result_dir))
except OSError:
    print("Directory already exists")


ids = pd.read_csv("input_data/AllSubjects.csv", sep = ";")["Heliusnr"].values
labels = pd.read_csv("input_data/AllSubjects.csv", sep = ";")["aTPOGroup"].values
neg_idx = np.where(labels == "Low")[0]
pos_idx = np.where(labels == "High")[0]

neg_ids = ids[neg_idx].astype(str)
pos_ids = ids[pos_idx].astype(str)
neg_size = len(neg_ids)
pos_size = len(pos_ids)


def filter_features(X, min_occurence_ratio):
    print("Original dimensions: {}".format(X.shape))
    # Filters out the features that are constant
    X = X[:, np.invert(np.all(X == X[0,:], axis = 0))]
    print("Dimensions after filtering out constant features: {}".format(X.shape))
    # Filters out features that have more than min_occurence_ratio of n_subjects zero counts
    # Aka, the feature has to occur in at least min_occurence_ratio of the samples
    X = X[:, (X == 0).sum(axis =0 ) < X.shape[0] * min_occurence_ratio]
    print("Dimensions after filtering based on an occurence threshold of {}: {}".format(min_occurence_ratio, X.shape))
    # makes a new column with the sum of the values in each column of a row is <20
    # removing all rows that have less than 5% of counts more than 20
    return X


print('Data loading')
print('')
X_train_df=pd.read_csv(filepath_or_buffer='input_data/X_train.csv',
                       index_col = 0)
y_train_df=pd.read_csv(filepath_or_buffer = 'input_data/aTPO_no_meds.csv',
                       index_col='Heliusnr')
X_train_df.fillna(method='ffill',inplace=True)
print('Finished')


X_train_df = X_train_df.transpose()
y_train_df = y_train_df.sort_index(axis = 0)
X_train_df_pos = X_train_df.loc[pos_ids]
X_train_df_neg = X_train_df.loc[neg_ids]
X_train_df_total = pd.concat([X_train_df_pos, X_train_df_neg], axis = 0)
y_binary = np.concatenate((np.zeros(pos_size), np.ones(neg_size)))
X_full = X_train_df_total.values
X_full = filter_features(X_full, occurrence_ratio)

label_names = {0:'aTPO Negative',1:'aTPO Positive'}

y_verbal = np.asarray([label_names[i] for i in y_binary], dtype=str)
sample_names = np.asarray(list(X_train_df_total.index), dtype=str)


list_auc=[]

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
StratShufSpl=StratifiedShuffleSplit(stability_samples_run_number,
                                    test_size=test_data_ratio, 
                                    random_state = random_seed)
print('Calculating Stratified Shuffled Split coefficients:')

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

X = X_full


selected_features = {}
selected_features["Features"] = []
selected_features["Scores"] = []
selected_features["p_values"] = []

list_feat_names = []
list_feat_importances = []


#b = VarianceThreshold(0.005)
b = SelectPercentile(f_classif,uni_feat_percent)
#b.fit(X_raw)
b.fit(X,y_binary)
X = X[:, b.get_support(indices=True)]
y=y_binary
selected_features["Features"] = X_train_df.columns[b.get_support(indices=True)]
feat_names = selected_features["Features"]
selected_features["Scores"] = b.scores_[b.get_support(indices=True)]
#selected_features["Variances"] = b.variances_[b.get_support(indices=True)]
selected_features["p_values"] = b.pvalues_[b.get_support(indices=True)]
print("Number of features after feature selection: {}".format(len(X[0])))

for samples,test in StratShufSpl.split(X,y):
    shuffle_counter+=1
    X_train,y_train=X[samples],y[samples]
    X_test,y_test=X[test],y[test]

    trees=ExtraTreesRegressor(n_estimators=100,n_jobs=1)

    param_grid = {"n_estimators":[n_estimators],
                  "criterion":['mse','mae'],
                  "max_depth":[None],
                  "min_samples_split":[2],
                  "min_samples_leaf":[1],
                  "max_features":['auto','sqrt'],
                  "random_state":[512],
                  "bootstrap": [False,True]}

    skf=StratifiedKFold(n_splits=num_cv_folds, random_state = random_seed)
    grid_search=GridSearchCV(trees,param_grid,scoring='roc_auc',
                               cv=skf,n_jobs=32,verbose=20,
                               refit=True,iid=False)
    grid_search.fit(X_train,y_train)
    best_model=grid_search.best_estimator_
    y_pred_test=best_model.predict(X_test)

    fpr, tpr, thresholds = roc_curve(y_test,y_pred_test,pos_label=1)
    auc_roc1 = auc(fpr, tpr)
    
    np.save("output_data/{}/y_test_{}.npy".format(result_dir, shuffle_counter), y_test)
    np.save("output_data/{}/y_pred_test_{}.npy".format(result_dir, shuffle_counter), y_pred_test)
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_test, pos_label=1)
    auc_roc1 = auc(fpr, tpr)
    list_auc.append(auc_roc1)

    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    aucs.append(auc_roc1)
    plt.plot(fpr, tpr, lw=1, alpha=0.3)
    list_auc.append(auc_roc1)

    aucs.append(auc_roc1)
    i += 1
    
    feature_importances = best_model.feature_importances_
    list_feat_importances.append(feature_importances)
    
    print("Best score: {}\n".format(grid_search.best_score_))
    print('')
    print(('Shuffle %i is done!!!' %shuffle_counter))


sns.set_style("whitegrid")
plt.plot([0, 1], [0, 1], linestyle='--', lw = 1, color='k',
             label="Random guess", alpha = .8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='k',
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
plt.savefig('output_data/{}/auc_avg.pdf'.format(result_dir), bbox_inches='tight')
#plt.show()


fi = np.array(list_feat_importances)
fi_avg = np.nanmean(fi, axis=0)

feature_importance = np.round(100.0 * (fi_avg / np.nanmax(fi_avg)), 2)
sorted_idx = np.argsort(feature_importance)
top_sorted_idx=sorted_idx[-30:]
pos = np.arange(top_sorted_idx.shape[0]) + .5


header=np.reshape(np.asarray(['FeatName','RelFeatImp']),(1,2))
feat_imp_vector=np.column_stack((feat_names,np.asarray(feature_importance,dtype=str)))
feat_vector_save=np.vstack((header,feat_imp_vector))
np.savetxt('output_data/{}/feat_imp_'.format(result_dir)+data_type+'.txt'.format(result_dir),feat_vector_save,fmt='%s',delimiter='\t')


# Scaling features for boxplots
X_scaled = minmax_scale(np.log(X + 1))

dataset_df = pd.DataFrame(data = X_scaled, index = sample_names, columns = feat_names)
dataset_df['class_label'] = y
dataset_df['class_label_verbal'] = y_verbal

sel_features = np.loadtxt('output_data/{}/feat_imp_'.format(result_dir)+data_type+'.txt'.format(result_dir),dtype=bytes, delimiter ='\t',skiprows=1).astype(str)
feat_scores = np.asarray(sel_features[:,1],dtype=float)
feat_scores_sorted_idx = np.argsort(feat_scores)[::-1]
feat_scores_sorted = feat_scores[feat_scores_sorted_idx]
top_feat_names_sorted = sel_features[feat_scores_sorted_idx[:num_top_feat],0]


# Subsampled
df_dataset_neg = dataset_df.loc[dataset_df['class_label'] == 0][top_feat_names_sorted]
df_dataset_pos = dataset_df.loc[dataset_df['class_label'] == 1][top_feat_names_sorted]

df_dataset_neg_avg = df_dataset_neg.mean(axis=0)
df_dataset_pos_avg = df_dataset_pos.mean(axis=0)


plt.figure()
plt.barh(pos, feature_importance[top_sorted_idx], align='center')
plt.yticks(pos, feat_names[top_sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Top 30 Relative Variable Importances')
plt.savefig('output_data/{}/feat_imp_'.format(result_dir)+data_type+'.pdf'.format(result_dir), bbox_inches='tight')
plt.close()


fdr_selector=SelectFdr()
fdr_selector.fit(X, y)
p_values_fdr=-np.log10(fdr_selector.pvalues_)
plt.figure()
plt.barh(pos, p_values_fdr[top_sorted_idx], align='center', color='g')
plt.axvline(x=-np.log10(0.05),color='r')
plt.yticks(pos, feat_names[top_sorted_idx])
plt.xlabel('-log10(p) value')
plt.title('Top 30 FDR corrected -log10(p) values ')
plt.savefig('output_data/{}/feat_imp_log10p_'.format(result_dir)+data_type+'.pdf'.format(result_dir), bbox_inches='tight')
plt.close()


# Sub
box_count=0
for feat_name in top_feat_names_sorted:
    box_count+=1
    plt.figure(figsize=(10,10))
    # plot boxplot with seaborn
    bplot=sns.boxplot(y=feat_name, x='class_label_verbal',
                     data=dataset_df,
                     width=0.5,
                     palette="colorblind",
                     showfliers=False)
    # add stripplot to boxplot with Seaborn
    bplot=sns.stripplot(y=feat_name, x='class_label_verbal',
                       data=dataset_df,
                       jitter=True,
                       marker='o',
                       alpha=0.5,
                       color='black')
    plot_file_name="output_data/{}/boxplot".format(result_dir)+"_"+str(box_count)+"_sub.png".format(result_dir)
    # save as jpeg
    bplot.figure.savefig(plot_file_name,
                        format='png',
                        dpi=100)


# Sub
radar_pos_data = minmax_scale(df_dataset_pos_avg.values,feature_range=(0.1,1))
radar_neg_data = minmax_scale(df_dataset_neg_avg.values,feature_range=(0.1,1))

data = [
    go.Scatterpolar(
      r = radar_pos_data,
      theta = top_feat_names_sorted,
      fill = 'toself',
      name = label_names[1]
    ),
    go.Scatterpolar(
      r = radar_neg_data,
      theta = top_feat_names_sorted,
      fill = 'toself',
      name = label_names[0]
    )
]

layout = go.Layout(
  polar = dict(
    radialaxis = dict(
      visible = True,
      range = [0, 1]
    )
  ),
    title = "Radar plot",
    height = 768,
    width = 1024,
    font = dict(
     size = 10
    ),
  showlegend = True
)

fig_radar = go.Figure(data=data, layout=layout)
py.plot(fig_radar, auto_open=False, image = None, image_filename='radar_chart_sub',
         output_type='file', filename='radar_chart_sub.html', validate=False,
         image_width=1024, image_height=768, show_link=False)
shutil.move('radar_chart_sub.html','output_data/{}/radar_chart_sub.html'.format(result_dir))


dataset_heatmap_all=dataset_df.copy().sort_values(by=['class_label'])
dataset_heatmap=dataset_heatmap_all[top_feat_names_sorted]


list_feat_scatter=[str(top_feat_names_sorted[i]) for i in range(0,5)]
list_feat_scatter.append('class_label_verbal')
dataset_scatter=dataset_heatmap_all.loc[:,list_feat_scatter]
plt.figure(figsize=(10,10))
sns.set(style="ticks")
g = sns.PairGrid(dataset_scatter,hue='class_label_verbal')
g = g.map_diag(plt.hist)
g = g.map_upper(plt.scatter)
g = g.map_lower(sns.kdeplot, cmap="Blues_d")
g = g.add_legend()
plt.savefig("output_data/{}/scatter_top_feat.png".format(result_dir),
                 format='png',
                 dpi=100)


