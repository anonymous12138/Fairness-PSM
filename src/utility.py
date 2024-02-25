from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset, BankDataset,MEPSDataset19
import numpy as np
import pandas as pd


def load_data(input_dataset, protected):
    dataset_orig = None
    if input_dataset == "adult":
        if protected == "sex":
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
        else:
            privileged_groups = [{'race': 1}]
            unprivileged_groups = [{'race': 0}]
        dataset_orig = AdultDataset().convert_to_dataframe()[0]
        dataset_orig.columns = dataset_orig.columns.str.replace("income-per-year", "Probability")
    elif input_dataset == "german":
        if protected == "sex":
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
        else:
            privileged_groups = [{'age': 1}]
            unprivileged_groups = [{'age': 0}]
        dataset_orig = GermanDataset().convert_to_dataframe()[0]
        dataset_orig['credit'] = np.where(dataset_orig['credit'] == 1, 1, 0)
        dataset_orig.columns = dataset_orig.columns.str.replace("credit", "Probability")
    elif input_dataset == "compas":
        if protected == "sex":
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
        else:
            privileged_groups = [{'race': 1}]
            unprivileged_groups = [{'race': 0}]
        dataset_orig = CompasDataset().convert_to_dataframe()[0]
        dataset_orig.columns = dataset_orig.columns.str.replace("two_year_recid", "Probability")
        dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 1, 0, 1)
    elif input_dataset == "bank":
        privileged_groups = [{'age': 1}]
        unprivileged_groups = [{'age': 0}]
        dataset_orig = BankDataset().convert_to_dataframe()[0]
        dataset_orig.rename(columns={'y': 'Probability'}, inplace=True)
    elif input_dataset == "mep":
        privileged_groups = [{'RACE': 1}]
        unprivileged_groups = [{'RACE': 0}]
        dataset_orig = MEPSDataset19().convert_to_dataframe()[0]
        dataset_orig.rename(columns={'UTILIZATION': 'Probability'}, inplace=True)
    elif input_dataset == "heart":
        privileged_groups = [{'age': 1}]
        unprivileged_groups = [{'age': 0}]
        dataset_orig = pd.read_csv("heart_processed.csv")
    elif input_dataset == "default":
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        dataset_orig = pd.read_csv("default_revised_processed.csv")

    return dataset_orig, privileged_groups,unprivileged_groups


def load_clf(name):
    if name == "lr":
        clf = LogisticRegression(solver='liblinear')
    elif name == "svm":
        clf = LinearSVC()
    elif name == "rf":
        clf = RandomForestClassifier()
    elif name == "dt":
        clf = DecisionTreeClassifier(min_impurity_decrease=0.001)
    elif name == "gb":
        clf = GradientBoostingClassifier()
    return clf
