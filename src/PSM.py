import sys
import os
import warnings
import argparse
import copy
import pandas as pd
import numpy as np
from numpy import mean, std
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utility import get_data, get_classifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from Measure import measure_final_score_ind
from aif360.datasets import BinaryLabelDataset
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from scipy.stats import ttest_ind
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import RandomOverSampler
sys.path.append(os.path.abspath('.'))
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, required=True,
                    choices=['adult', 'german', 'compas', 'bank', 'mep', 'heart', 'default'], help="Dataset")
parser.add_argument("-c", "--clf", type=str, required=True,
                    choices=['rf', 'svm', 'lr', "dt", "gb"], help="Classifier")
parser.add_argument("-p", "--protected", type=str, required=True,
                    help="Protected Attribute")
parser.add_argument("-r", "--repeat", type=int, required=True, default=10,
                    help="Repeat Times")

args = parser.parse_args()

dataset_used = args.dataset
attr = args.protected
clf_name = args.clf
smote = False
repeat_time = args.repeat

val_name = "PSM_{}_{}_{}.txt".format(clf_name, dataset_used, attr)
fout = open(val_name, 'w')

dataset_orig, privileged_groups, unprivileged_groups = get_data(dataset_used, attr)

results = {}
performance_index = ['accuracy', 'recall', 'precision', 'f1score', 'aod', 'eod', 'spd', 'di']
for p_index in performance_index:
    results[p_index] = []


def conditional(X_no_matched, keyword, clf, thresh0=.5, thresh1=.5):
    proba = clf.predict_proba(X_no_matched)[:, 1]
    col = X_no_matched.columns.tolist().index(keyword)
    X_no_matched['proba'] = proba
    X_no_matched['pred'] = 0
    X_no_matched['pred'].loc[(X_no_matched[keyword] == 0) & (X_no_matched['proba'] >= thresh0)] = 1
    X_no_matched['pred'].loc[(X_no_matched[keyword] == 1) & (X_no_matched['proba'] >= thresh1)] = 1
    res = X_no_matched['pred']
    X_no_matched.drop(['pred', 'proba'], inplace=True, axis=1)
    return res


def search_threshold(df_no_matched, keyword, prior_knowledge=0):
    thresh0 = df_no_matched[df_no_matched[keyword] == 0].ps.mean()
    thresh1 = df_no_matched[df_no_matched[keyword] == 1].ps.mean()

    # print("Original:", thresh0, thresh1)
    res0, res1 = 0, 0
    utility_best = -1
    for theta0 in range(0, 100, 2):
        theta0 /= 100
        for theta1 in range(0, 100, 2):
            theta1 /= 100
            temp0 = df_no_matched[df_no_matched[keyword] == 0].ps + theta0
            temp1 = df_no_matched[df_no_matched[keyword] == 1].ps - theta1
            local_p = ttest_ind(temp1, temp0).pvalue
            local_utility = local_p - 0.5*((temp0.mean() - 0.5) ** 2 + (temp1.mean() - 0.5) ** 2) ** .5

            if local_utility >= utility_best:
                utility_best = local_utility
                res0, res1 = theta0, theta1

    # print("Best theta:", res0, res1)
    return res0, res1


for r in range(repeat_time):
    print(r)
    keyword = attr
    np.random.seed(r)
    dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.3)
    train, test = train_test_split(dataset_orig, test_size=0.3)

    dataset_orig_train = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train,
                                            label_names=['Probability'],
                                            protected_attribute_names=[attr])
    dataset_orig_test = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test,
                                           label_names=['Probability'],
                                           protected_attribute_names=[attr])

    scaler = MinMaxScaler()
    train = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
    test = pd.DataFrame(scaler.transform(test), columns=test.columns)

    X_train = train.loc[:, train.columns != 'Probability']
    y_train = train.loc[:, 'Probability']
    X_test = test.loc[:, test.columns != 'Probability']
    y_test = test.loc[:, 'Probability']
    ratio = sum(y_train) / len(y_train)
    print("Ratio:", ratio)

    clf = get_classifier(clf_name)
    if clf_name == 'svm':
        clf = CalibratedClassifierCV(base_estimator=clf)
    clf1 = clf.fit(dataset_orig_train.features, dataset_orig_train.labels)
    clf = get_classifier(clf_name)
    if clf_name == 'svm':
        clf = CalibratedClassifierCV(base_estimator=clf)
    if smote:
        ros = RandomOverSampler()
        ros.fit(X_train, y_train)
        X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
        clf.fit(X_resampled, y_resampled)

    else:
        clf.fit(X_train, y_train)
    final_predictions = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]

    proba = clf.predict_proba(X_test)[:, 1]
    X_test_copy = X_test.copy()
    X_test_copy['ps'] = proba
    X_test_copy['Probability'] = y_test
    # print("=========Clustering Start============")
    caliper = np.std(proba)
    # print(f'caliper (radius) is: {caliper:.4f}')
    n_neighbors = 10
    # setup knn
    knn = NearestNeighbors(n_neighbors=n_neighbors, radius=caliper)
    ps = X_test_copy[['ps']]  # double brackets as a dataframe
    knn.fit(ps)
    # distances and indexes
    distances, neighbor_indexes = knn.kneighbors(ps)

    matched_control = []  # keep track of the matched observations in control
    for current_index, row in X_test_copy.iterrows():  # iterate over the dataframe
        if row[keyword] == 0:  # the current row is in the control group
            X_test_copy.loc[current_index, 'matched'] = np.nan  # set matched to nan
        else:
            for i in range(len(neighbor_indexes[current_index, :])):  # for each row in treatment, find the k neighbors
                # make sure the current row is not the idx - don't match to itself
                # and the neighbor is in the control
                idx = neighbor_indexes[current_index, :][i]
                if (current_index != idx) and (X_test_copy.loc[idx][keyword] == 0):
                    if idx not in matched_control:  # this control has not been matched yet
                        X_test_copy.loc[current_index, 'matched'] = idx  # record the matching
                        matched_control.append(idx)  # add the matched to the list
                        break
    # print("=========Clustering END============")
    treatment_matched = X_test_copy.dropna(subset=['matched'])  # drop not matched
    # print(treatment_matched.shape)
    # matched control observation indexes
    control_matched_idx = treatment_matched.matched
    control_matched_idx = control_matched_idx.astype(int)  # change to int
    control_matched = X_test_copy.loc[control_matched_idx, :]  # select matched control observations
    # combine the matched treatment and control
    df_matched = pd.concat([treatment_matched, control_matched])
    pair_idx = control_matched_idx.tolist() + treatment_matched.index.tolist()

    y_matched = df_matched['Probability']
    X_matched = df_matched.drop(['ps', 'matched', 'Probability'], axis=1)
    y_pred_matched = clf.predict(X_matched)

    # pair_idx = control_matched_idx.tolist() + treatment_matched.index.tolist()
    df_no_matched = X_test_copy.drop(index=pair_idx, inplace=False)
    # df_no_matched = X_test[X_test['matched'].isnull()] # drop not matched
    df_no_matched_control = df_no_matched[df_no_matched[keyword] == 0]
    df_no_matched_treatment = df_no_matched[df_no_matched[keyword] == 1]
    df_no_matched.rename({'y': 'Probability'}, axis=1, inplace=True)
    no_match_idx = df_no_matched.index

    print("=========Search Start============")
    threshold0, threshold1 = search_threshold(df_no_matched, keyword, prior_knowledge=ratio)

    y_no_matched = df_no_matched['Probability']
    X_no_matched = df_no_matched.drop(['ps', 'matched', 'Probability'], axis=1)
    y_pred_no_matched = conditional(X_no_matched, keyword, clf, 0.5 - threshold0, 0.5 + threshold1)
    y_pred_all = list(y_pred_matched) + list(y_pred_no_matched)
    y_true_all = list(y_matched) + list(y_no_matched)
    df_all = pd.concat([df_matched, df_no_matched], axis=0)

    cm1 = confusion_matrix(y_true_all, y_pred_all)
    acc = (np.round(accuracy_score(y_true_all, y_pred_all), 2))
    pre = (np.round(precision_score(y_true_all, y_pred_all), 2))
    rec = (np.round(recall_score(y_true_all, y_pred_all), 2))
    f = (np.round(f1_score(y_true_all, y_pred_all), 2))
    aod = (np.absolute(measure_final_score_ind(df_all, y_pred_all, cm1, keyword, 'aod')))
    eod = (np.absolute(measure_final_score_ind(df_all, y_pred_all, cm1, keyword, 'eod')))
    spd = (np.absolute(measure_final_score_ind(df_all, y_pred_all, cm1, keyword, 'SPD')))
    di = (np.absolute(measure_final_score_ind(df_all, y_pred_all, cm1, keyword, 'DI')))
    round_result = [acc * 100, pre * 100, rec * 100, f * 100, aod * 100, eod * 100, spd * 100, di * 100]
    print(acc * 100, pre * 100, rec * 100, f * 100, aod * 100, eod * 100, spd * 100, di * 100)

    test_df_copy = copy.deepcopy(dataset_orig_test)
    test_df_copy.labels = final_predictions

    # round_result= measure_final_score(dataset_orig_test,test_df_copy,privileged_groups,unprivileged_groups)
    for i in range(len(performance_index)):
        results[performance_index[i]].append(round_result[i])

for p_index in performance_index:
    fout.write(p_index + '\t')
    for i in range(repeat_time):
        fout.write('%f\t' % results[p_index][i])
    fout.write('%f\t%f\n' % (mean(results[p_index]), std(results[p_index])))
    print("Result:", p_index, np.round(mean(results[p_index]), 2))
fout.close()
