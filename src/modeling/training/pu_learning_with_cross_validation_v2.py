import os
import pandas as pd
import numpy as np
from src.data_statistics.pqn_dataset_extraction import create_datasets
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.inspection import permutation_importance
from sklearn.utils import resample
from sklearn.metrics import precision_score, recall_score, f1_score, auc, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from pulearn import (
    ElkanotoPuClassifier,
    WeightedElkanotoPuClassifier
)
from tqdm import tqdm

import matplotlib.pyplot as plt
import os
import time

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def code_churn_preprocessing(df):
    # Adjust the value of has_source_code column based on conditions
    df.loc[(df['lines_of_source_code_added'] != 0) |
           (df['lines_of_source_code_deleted'] != 0) |
           (df['lines_of_source_code_modified'] != 0), 'has_source_code'] = 1
    df.loc[(df['lines_of_source_code_added'] == 0) &
           (df['lines_of_source_code_deleted'] == 0) &
           (df['lines_of_source_code_modified'] == 0), 'has_source_code'] = 0

    df.loc[(df['lines_of_config_file_added'] != 0) |
           (df['lines_of_config_file_deleted'] != 0) |
           (df['lines_of_config_file_modified'] != 0), 'has_config_files'] = 1
    df.loc[(df['lines_of_config_file_added'] == 0) &
           (df['lines_of_config_file_deleted'] == 0) &
           (df['lines_of_config_file_modified'] == 0), 'has_config_files'] = 0

    # Set the value of has_contains_code_patch based on conditions
    df.loc[(df['has_config_files'] == 1) | (df['has_source_code'] == 1), 'has_contains_code_patch'] = 1
    df.loc[(df['has_config_files'] != 1) & (df['has_source_code'] != 1), 'has_contains_code_patch'] = 0
    return df


def comment_date_preprocessing(df):
    # Convert comment_created_at to datetime
    df['comment_created_at'] = pd.to_datetime(df['comment_created_at'], format='%a %d %b %Y %H:%M:%S %z')
    df['comment_created_at_str'] = df['comment_created_at'].astype(str)
    df['comment_created_at_ts'] = pd.Series(dtype='int64')
    for i, value in enumerate(df['comment_created_at_str']):
        timestamp = pd.Timestamp(value)
        ts_int = int(timestamp.timestamp())
        df.at[i, 'comment_created_at_ts'] = ts_int
    df = df.drop(columns=['comment_created_at_str'])
    df['created'] = pd.to_datetime(df['created'], format='%a %d %b %Y %H:%M:%S %z')
    # df['comment_created_at'] = pd.to_timestamp(df['comment_created_at'], format='%a %d %b %Y %H:%M:%S %z')
    df['created_str'] = df['created'].astype(str)
    df['created_ts'] = pd.Series(dtype='int64')
    for i, value in enumerate(df['created_str']):
        timestamp = pd.Timestamp(value)
        ts_int = int(timestamp.timestamp())
        df.at[i, 'created_ts'] = ts_int
    df = df.drop(columns=['created_str'])
    df['gap_days'] = (df['comment_created_at_ts'] - df['created_ts']) / 86400
    df['comment_created_at'] = pd.to_datetime(df['comment_created_at'], format='%Y-%m-%d %H:%M:%S%z', utc=True)
    return df


def drop_unused_columns(df):
    columns_to_drop = ['type_id', 'priority_id', 'has_contain_Test_files', 'lines_of_Test_classes_modified',
                       'lines_of_Test_classes_added', 'lines_of_Test_classes_deleted', 'lines_of_Test_files',
                       'deletions', 'insertions', 'files', 'Time_difference', 'commit_num', 'is_share_similar_emsg',
                       'is_share_same_emsg', 'num_parrel_commits', 'num_of_developers', 'created', 'created_ts',
                       'comment_created_at', 'time', 'day']

    columns_to_drop = ['type_id', 'priority_id', 'Time_difference', 'commit_num', 'created', 'created_ts',
                       'comment_created_at', 'time', 'day']
    # columns_to_drop = columns_to_drop + cols_to_rm
    # df = df.drop(columns=[col for col in df.columns if col not in selected_columns])
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    return df


def column_rename(df):
    '''
    Rename the column names
    :param df:
    :return:
    '''
    # Dictionary of column name mappings
    column_mapping = {
        'num_parrel_issues': 'Number of Parallel Issues',
        'num_prior_comments': 'Number of Prior Comments',
        'if_cross_project': 'Is Cross Projects',
        'type_id': 'Type Id',
        'priority_id': 'Priority',
        'num_of_sf_comments': 'Number of Similar Failures',
        'share_same_emsg': 'Is Shared Same Emsg',
        'has_config_files': 'Has Config Files',
        'has_source_code': 'Has Source Code',
        'lines_of_source_code_added': 'Source Code Lines Added',
        'lines_of_source_code_deleted': 'Source Code Lines Deleted',
        'lines_of_source_code_modified': 'Source Code Lines Modified',
        'num_of_modified_source_code_files': 'Modified Source Code Files',
        'lines_of_config_file_added': 'Config Lines Added',
        'lines_of_config_file_deleted': 'Config Lines Deleted',
        'lines_of_config_file_modified': 'Config Lines Modified',
        'num_of_modified_config_files': 'Modified Config Files',
        'has_contains_code_patch': 'Has Code Patch',
        'lines_of_Test_classes_added': 'Test Classes Added',
        'lines_of_Test_classes_deleted': 'Test Classes Deleted',
        'lines_of_Test_classes_modified': 'Test Classes Lines Modified',
        'lines_of_Test_files': 'Modified Test Classes Files',
        'has_contain_Test_files': 'Has Test Files',
        'gap_days': 'CI Latency',
        'is_daily_time': 'Daily Time',
        'is_night_time': 'Night Time',
        'is_weekday': 'Weekday',
        'is_weekend': 'Weekend'
    }
    # Rename columns using the dictionary
    df.rename(columns=column_mapping, inplace=True)
    # Loop through columns and rename if they start with "is_"
    for column in df.columns:
        if column.startswith('is_'):
            new_column = "Is " + column[3:]
            df.rename(columns={column: new_column}, inplace=True)
    return df


def fill_nans(df):
    # List of columns to exclude from filling NaN values with 0
    columns_to_exclude = ['label']
    # Fill NaN values with -1 in all columns except 'label'
    df.loc[:, ~df.columns.isin(columns_to_exclude)] = df.loc[:, ~df.columns.isin(columns_to_exclude)].fillna(0)
    return df


def feature_selection(project, df):
    '''
    This is a function that only return the df with features and remove unsed features.
    The Feature Selection is done by redn and varclus analysis by R.
    We also public the R script in the repo
    :param project:
    :param df:
    :return:
    '''
    selection_map = {
        "ambari": ["Number of Parallel Issues", "Number of Prior Comments", "Is Cross Projects",
                   "Number of Similar Failures", "Is Reference", "Is Duplicate", "Is Regression", "Is Blocker",
                   "Is Container", "Is dependent", "Is Required", "Is Cloners", "Is Blocked", "Config Lines Modified",
                   "Has Code Patch", "CI Latency"],
        'hadoop': ["Number of Parallel Issues", "Number of Prior Comments", "Number of Similar Failures",
                   "Is Shared Same Emsg", "Is dependent", "Is Duplicate", "Is Reference", "Is Cloners", "Is Regression",
                   "Is Blocker", "Is Supercedes", "Is Container", "Is Required", "Is Problem/Incident", "Is Blocked",
                   "Is Dependent", "Is Completes", "Is Child-Issue", "Config Lines Modified", "Has Code Patch",
                   "CI Latency"],
        'hbase': ["Number of Parallel Issues", "Number of Prior Comments", "Number of Similar Failures",
                  "Is Shared Same Emsg", "Is Incorporates", "Is Reference", "Is dependent", "Is Duplicate",
                  "Is Supercedes", "Is Cloners", "Is Regression", "Is Required", "Is Dependent", "Is Container",
                  "Is Child-Issue", "Is Problem/Incident", "Is Blocked", "Is Completes", "Config Lines Modified",
                  "Has Code Patch", "CI Latency"],
        'hdds': ["Number of Parallel Issues", "Number of Prior Comments", "Number of Similar Failures",
                 "Is Shared Same Emsg", "Is Duplicate", "Is Reference", "Is Blocker", "Is Cloners", "Is Child-Issue",
                 "Is dependent", "Is Problem/Incident", "Is Regression", "Is Container", "Is Dependent",
                 "Is Supercedes", "Is Required", "Is Blocked", "Is Dependency", "Config Lines Deleted",
                 "Config Lines Modified", "Has Code Patch", "CI Latency"],
        'hdfs': ["Number of Parallel Issues", "Number of Prior Comments", "Number of Similar Failures",
                 "Is Shared Same Emsg", "Is Reference", "Is Incorporates", "Is Blocker", "Is Container", "Is dependent",
                 "Is Supercedes", "Is Regression", "Is Cloners", "Is Required", "Is Problem/Incident", "Is Dependent",
                 "Is Child-Issue", "Is Blocked", "Is Completes", "Is Parent Feature", "Is Dependency",
                 "Config Lines Deleted", "Config Lines Modified", "Has Code Patch", "CI Latency"],
        'hive': ["Number of Parallel Issues", "Number of Prior Comments", "Number of Similar Failures",
                 "Is Shared Same Emsg", "Is dependent", "Is Blocker", "Is Reference", "Is Incorporates",
                 "Is Regression", "Is Cloners", "Is Child-Issue", "Is Required", "Is Supercedes", "Is Container",
                 "Is Problem/Incident", "Is Blocked", "Is Completes", "Is Dependent", "Is Parent Feature",
                 "Is Dependency", "Config Lines Modified", "Has Code Patch", "CI Latency"],
        'yarn': ["Number of Parallel Issues", "Number of Prior Comments", "Number of Similar Failures",
                 "Is Shared Same Emsg", "Is Incorporates", "Is Reference", "Is dependent", "Is Duplicate",
                 "Is Regression", "Is Cloners", "Is Required", "Is Supercedes", "Is Container", "Is Blocked",
                 "Is Problem/Incident", "Is Child-Issue", "Is Dependency", "Is Dependent", "Is Completes",
                 "Config Lines Deleted", "Config Lines Modified", "Has Code Patch", "CI Latency"]
    }
    selected_columns = ['project_name', 'comment_id', 'issue_id', 'label', 'comment_created_at_ts'] + selection_map[
        project]
    return df[selected_columns]


def RandomGussing(P, Q, N):
    P_features = P.drop(columns=['label'])
    P_labels = P['label']

    Q_features = Q.drop(columns=['label'])
    Q_labels = Q['label']

    N_features = N.drop(columns=['label'])
    N_labels = N['label']

    positive_ratio = float(len(Q_labels) / (len(Q_labels) + len(N_labels)))
    print(
        f'''The ratio of the positive data :{float(len(P_labels) + len(Q_labels)) / (len(P_labels) + len(Q_labels) + len(N_labels))}''')

    # Initialize KFold
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    # Initialize variables to store confusion matrix values
    a, b, c, d = 0, 0, 0, 0
    all_true_labels = []
    all_pred_probabilities = []

    for (train_index_P, test_index_P), (train_index_Q, test_index_Q), (train_index_N, test_index_N) in zip(
            kf.split(P_features), kf.split(Q_features), kf.split(N_features)):
        P_train, P_test = P_features.iloc[train_index_P], P_features.iloc[test_index_P]
        Q_train, Q_test = Q_features.iloc[train_index_Q], Q_features.iloc[test_index_Q]
        N_train, N_test = N_features.iloc[train_index_N], N_features.iloc[test_index_N]

        # Combine evaluation data
        P_Q_test_data = pd.concat([P_test, Q_test])
        P_Q_test_labels = pd.concat([P_labels.iloc[test_index_P], Q_labels.iloc[test_index_Q]])

        # Define the number of samples for each dataset
        num_samples_PQ = P_Q_test_data.shape[0]
        num_samples_N = N_test.shape[0]

        # Generate random probabilities
        P_Q_test_prob = np.random.rand(num_samples_PQ)
        N_test_prob = np.random.rand(num_samples_N)
        # Generate random predictions
        P_Q_test_pred = np.where(P_Q_test_prob > 0.5, 1, -1)
        N_test_pred = np.where(N_test_prob > 0.5, 1, -1)

        all_true_labels.extend(P_Q_test_labels.tolist() + N_labels.iloc[test_index_N].tolist())
        all_pred_probabilities.extend(P_Q_test_prob.tolist() + N_test_prob.tolist())

        # Calculate confusion matrix values for this fold
        ai = np.sum((P_Q_test_labels == 1) & (P_Q_test_pred == 1))
        bi = np.sum((P_Q_test_labels == 1) & (P_Q_test_pred == -1))
        ci = np.sum((N_labels.iloc[test_index_N] == -1) & (N_test_pred == 1))
        di = np.sum((N_labels.iloc[test_index_N] == -1) & (N_test_pred == -1))
        a += ai
        b += bi
        c += ci
        d += di

    # Calculate precision and recall
    precision = a / (a + c)
    recall = a / (a + b)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # Calculate AUC
    auc = roc_auc_score(all_true_labels, all_pred_probabilities)
    return precision, recall, f1_score, auc


def PULearning(P, Q, N):
    P_features = P.drop(columns=['label'])
    P_labels = P['label']

    Q_features = Q.drop(columns=['label'])
    Q_labels = Q['label']

    N_features = N.drop(columns=['label'])
    N_labels = N['label']

    positive_ratio = float(len(Q_labels) / (len(Q_labels) + len(N_labels)))
    print(
        f'''The ratio of the positive data :{float(len(P_labels) + len(Q_labels)) / (len(P_labels) + len(Q_labels) + len(N_labels))}''')
    estimator = RandomForestClassifier(
        n_estimators=100,
        criterion='gini',
        bootstrap=True,
        n_jobs=1,
    )

    # pu_estimator = ElkanotoPuClassifier(estimator=estimator, hold_out_ratio=0.1)
    pu_estimator = WeightedElkanotoPuClassifier(estimator=estimator, labeled=len(P), unlabeled=len(Q) + len(N))
    # estimator.f
    # pu_estimator = estimator

    # Initialize KFold
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    # Initialize variables to store confusion matrix values
    a, b, c, d = 0, 0, 0, 0
    all_true_labels = []
    all_pred_probabilities = []

    for (train_index_P, test_index_P), (train_index_Q, test_index_Q), (train_index_N, test_index_N) in zip(
            kf.split(P_features), kf.split(Q_features), kf.split(N_features)):
        P_train, P_test = P_features.iloc[train_index_P], P_features.iloc[test_index_P]
        Q_train, Q_test = Q_features.iloc[train_index_Q], Q_features.iloc[test_index_Q]
        N_train, N_test = N_features.iloc[train_index_N], N_features.iloc[test_index_N]

        # Combine training data
        pu_train_data = pd.concat([P_train, Q_train, N_train])
        pu_train_labels = pd.concat(
            [P_labels.iloc[train_index_P], Q_labels.iloc[train_index_Q], N_labels.iloc[train_index_N]])

        # Train the PU estimator
        # pu_estimator = estimator
        pu_estimator.fit(pu_train_data.values, pu_train_labels.values)
        # Combine evaluation data
        P_Q_test_data = pd.concat([P_test, Q_test])
        P_Q_test_labels = pd.concat([P_labels.iloc[test_index_P], Q_labels.iloc[test_index_Q]])

        # Predict on test data
        P_Q_test_pred = pu_estimator.predict(P_Q_test_data.values)
        N_test_pred = pu_estimator.predict(N_test.values)

        # Predict probabilities for AUC
        P_Q_test_prob = pu_estimator.predict_proba(P_Q_test_data.values)
        N_test_prob = pu_estimator.predict_proba(N_test.values)

        all_true_labels.extend(P_Q_test_labels.tolist() + N_labels.iloc[test_index_N].tolist())
        all_pred_probabilities.extend(P_Q_test_prob.tolist() + N_test_prob.tolist())

        # Calculate confusion matrix values for this fold
        ai = np.sum((P_Q_test_labels == 1) & (P_Q_test_pred == 1))
        bi = np.sum((P_Q_test_labels == 1) & (P_Q_test_pred == -1))
        ci = np.sum((N_labels.iloc[test_index_N] == -1) & (N_test_pred == 1))
        di = np.sum((N_labels.iloc[test_index_N] == -1) & (N_test_pred == -1))
        a += ai
        b += bi
        c += ci
        d += di

    # Calculate precision and recall
    precision = a / (a + c)
    recall = a / (a + b)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # Calculate AUC
    auc = roc_auc_score(all_true_labels, all_pred_probabilities)
    return precision, recall, f1_score, auc


current_file_path = os.getcwd()
base_path = '/'.join(current_file_path.split('/')[:-3] + ['data', 'modeling_data', 'training'])
data_path = '/'.join(current_file_path.split('/')[:-3] + ['data'])
project_name_group = ['hive', 'hadoop', 'yarn', 'hdfs', 'hbase', 'ambari', 'hdds']

test_project_name = project_name_group[-1]
project_name = test_project_name

data_with_features = pd.read_csv(f'''{base_path}/features2/{project_name}_data_with_features.csv''')
data_with_features = code_churn_preprocessing(data_with_features)
data_with_features = comment_date_preprocessing(data_with_features)
data_with_features = drop_unused_columns(data_with_features)
data_with_features = column_rename(data_with_features)
if test_project_name == 'hive':
    data_with_features['num_of_tests_failed'] = data_with_features['num_of_tests_failed'].apply(
        lambda x: 0 if x == -1 else x)
# data_with_features['CI Latency'] = data_with_features['CI Latency'].apply(lambda x: 1 if x > 6 else x)
# grouped = data_with_features.groupby('label')['CI Latency'].describe()
# print(grouped)
# data_with_features['CI Latency'] = data_with_features['CI Latency'].apply(lambda x: 0 if x >= 1 else x)
# data_with_features = feature_selection(project_name, data_with_features)

P, Q, N = create_datasets(project_name=project_name, data_path=data_path, df_with_features=data_with_features)
print(f"Datasets P, Q, and N have been constructed. Dataset P: {len(P)}, Dataset Q: {len(Q)}, Dataset N: {len(N)}")
P['label'] = 1
Q['label'] = 1
N['label'] = -1
# Drop columns
columns_to_drop = ['project_name', 'comment_id', 'issue_id', 'comment_created_at_ts', 'combined_issue_id',
                   'insertion', 'deletion', 'files', 'deletions', 'insertions']
# columns_to_drop = columns_to_drop + [c for c in list(P.columns) if "Is " in c and "share" not in c and "Cross" not in c]
P = P.drop(columns=columns_to_drop, errors='ignore')
Q = Q.drop(columns=columns_to_drop, errors='ignore')
N = N.drop(columns=columns_to_drop, errors='ignore')
precision_list = []
recall_list = []
f1_score_list = []
auc_list = []
precision, recall, f1_score, auc = PULearning(P, Q, N)
print(f" PU Learning Precision: {precision}\nRecall: {recall}\nF1 Score: {f1_score}\nAUC:{auc}")
r_precision, r_recall, r_f1_score, r_auc = RandomGussing(P, Q, N)
print(f" Random Guessing Precision: {r_precision}, Recall: {r_recall}, F1 Score: {r_f1_score}, AUC:{r_auc}")

