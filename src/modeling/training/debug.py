import os
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance
from pulearn import ElkanotoPuClassifier
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
    selected_columns = ['project_name', 'comment_id', 'issue_id','label','comment_created_at_ts'] + selection_map[project]
    return df[selected_columns]


current_file_path = os.getcwd()
base_path = '/'.join(current_file_path.split('/')[:-3] + ['data', 'modeling_data','training'])
project_name_group = ['hive', 'hadoop', 'yarn', 'hdfs', 'hbase', 'ambari', 'hdds']

for pi in range(len(project_name_group)):
    project_name = project_name_group[pi]
    positive_data = pd.read_csv(f'''{base_path}/labeled_data/{project_name}_labeled_data.csv''')
    positive_data = positive_data[['issue_id', 'comment_id']]
    # Set labels in benchmark_data to 1
    positive_data['label'] = 1
    # Get unique 'issue_id' and 'comment_id' in benchmark_data
    positive_ids = positive_data[['issue_id', 'comment_id']].values
    data_with_features = pd.read_csv(f'''{base_path}/features/{project_name}_data_with_features.csv''')
    data_with_features['issue_id'] = data_with_features['project_name'].str.cat(data_with_features['issue_id'].astype(str), sep='-')
    data_with_features['label'] = np.nan
    df = data_with_features.merge(positive_data, on=['issue_id', 'comment_id'], how='left', suffixes=('_feature_df', '_positive_df'))
    df['label'] = df.apply(lambda row: 1 if row['label_positive_df'] == 1 else -1, axis=1)
    # Getting the distribution of the 'label' column
    label_distribution = df['label'].value_counts()

    # print(label_distribution)
    df.drop(['label_feature_df'], axis=1, inplace=True)
    df.drop('label_positive_df', axis=1, inplace=True)
    # print(df.head(10))
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
    # df['created'] = pd.to_datetime(df['created'], unit='s',utc=True).dt.tz_convert('UTC')
    df['gap_days'] = (df['comment_created_at_ts'] - df['created_ts']) / 86400
    df['comment_created_at'] = pd.to_datetime(df['comment_created_at'], format='%Y-%m-%d %H:%M:%S%z', utc=True)
    columns_to_drop = ['type_id', 'priority_id', 'has_contain_Test_files', 'lines_of_Test_classes_modified',
                           'lines_of_Test_classes_added', 'lines_of_Test_classes_deleted', 'lines_of_Test_files',
                           'deletions', 'insertions', 'files', 'Time_difference', 'commit_num', 'is_share_similar_emsg',
                           'is_share_same_emsg', 'num_parrel_commits', 'num_of_developers','created', 'created_ts', 'comment_created_at', 'time', 'day']

    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    # List of columns to exclude from filling NaN values with 0
    columns_to_exclude = ['label']
    # Fill NaN values with -1 in all columns except 'label'
    df.loc[:, ~df.columns.isin(columns_to_exclude)] = df.loc[:, ~df.columns.isin(columns_to_exclude)].fillna(0)

    df = column_rename(df)

    print(df['label'].value_counts())
    print(df.columns)

    # project_name = 'hive'
    # Feature selection function call (implementation not shown)
    df = feature_selection(project_name, df)
    prediction_df = df

    # Extract the integer part of the issue_id and sort by it and comment_id
    df = df.sort_values(by=['comment_created_at_ts'], ascending=True).reset_index(drop=True)
    print(df.columns)
    # Split data into positive (labeled) and unlabeled portions
    positive_data = df[df["label"] == 1].reset_index(drop=True)
    df_for_prediction = df.drop(columns=['project_name', 'comment_id', 'issue_id', 'comment_created_at_ts', 'label'])
    unlabeled_data = df[df["label"].isnull()]
    num_positive_samples = len(df[df['label'] == 1])

    # Initialize random seed for reproducibility
    random.seed(42)
    train_data = positive_data.copy().drop("label", axis=1)
    val_data = pd.DataFrame()
    # columns = train_data.columns.tolist()
    # Performance metrics
    precision_pu, recall_pu, f1_pu, auc_pu = [], [], [], []
    precision_rf, recall_rf, f1_rf, auc_rf = [], [], [], []
    model_performance_data = []

    best_model_recall = 0
    best_model = None
    best_model_importance = pd.DataFrame(columns=['feature_name', 'importance_mean', 'importance_std'])

    # LOOCV implementation
    for i in range(98, num_positive_samples):
        # Select the last `i` positive samples as evaluation data
        test_positive_data = positive_data.iloc[-i:]
        # Select the first `num_pos - i` positive samples for training
        train_positive_data = positive_data.iloc[:num_positive_samples - i]
        # Select `num_pos - i` negative samples where the order is before the evaluation dataset
        train_negative_data = df[(df['label'] == -1) &
                                 (df['comment_created_at_ts'] < test_positive_data['comment_created_at_ts'].min())]

        for j in range(100):
            precision_pu_round, recall_pu_round, f1_pu_round, auc_pu_round = [], [], [], []
            precision_rf_round, recall_rf_round, f1_rf_round, auc_rf_round = [], [], [], []

            # train_negative_sample = train_negative_data.sample((num_positive_samples - i)*10, random_state=42)
            train_negative_sample = train_negative_data.sample(200 - (num_positive_samples - i), random_state=42)
            print(f'''[LOG] This is the size of positive samples: {len(train_positive_data)}, and this is the size of negative samples: {len(train_negative_sample)}''')
            # train_negative_sample = train_negative_data
            # Combine training data
            pu_data = pd.concat([train_positive_data, train_negative_sample])
            # Extract the 'label' column as a NumPy array
            pu_labels = pu_data['label'].to_numpy()
            pu_data_with_features_only = pu_data.drop(
                columns=['project_name', 'comment_id', 'issue_id', 'comment_created_at_ts', 'label'])
            columns = pu_data_with_features_only.columns.tolist()
            print(f'''This is the distribution of label:{np.unique(pu_labels, return_counts=True)}''')
            pu_combined = np.column_stack((pu_data_with_features_only, pu_labels))
            np.random.shuffle(pu_combined)
            # Separate the shuffled data and labels
            pu_data_with_features_only = pu_combined[:, :-1]
            pu_labels = pu_combined[:, -1].astype(int)
            from sklearn.ensemble import RandomForestClassifier

            estimator = RandomForestClassifier(
                n_estimators=100,
                bootstrap=True,
                n_jobs=1,
            )
            pu_estimator = ElkanotoPuClassifier(estimator=estimator, hold_out_ratio=0.1)
            print(f'''{pu_data_with_features_only.shape},{pu_labels.shape}''')
            pu_estimator.fit(pu_data_with_features_only, pu_labels)
            importance = pu_estimator.estimator.feature_importances_
            combined_dict = dict(zip(columns, importance))
            sorted_dict = dict(sorted(combined_dict.items(), key=lambda x: x[1], reverse=True))
            # for k, v in sorted_dict.items():
            #     print(f'''{project_name.upper()} feature {k} score is {v}''')

            test_data_only_with_features = test_positive_data.drop(
                columns=['project_name', 'comment_id', 'issue_id', 'comment_created_at_ts', 'label'])
            X_pred = test_data_only_with_features.values
            y_pred = pu_estimator.predict(X_pred)
            precision = precision_score(np.ones(len(X_pred)), y_pred, zero_division=1)
            recall = recall_score(np.ones(len(X_pred)), y_pred, zero_division=1)
            f1 = f1_score(np.ones(len(X_pred)), y_pred, zero_division=1)
            print(type(np.ones(len(X_pred))))
            print(type(y_pred))
            # auc_score = roc_auc_score(np.ones(len(X_pred), y_pred))

            round_results_pu = {
                "dropout": i,
                "f1": f1,
                "recall": recall,
                "model": "PULearning",
                "iteration": j
            }
            best_model = pu_estimator
            feature_importance_list = []
            scoring = ['recall', 'f1']
            r_multi = permutation_importance(pu_estimator.estimator, X_pred, np.ones(len(X_pred)), n_repeats=10,
                                             random_state=0,
                                             scoring=scoring)
            for metric in r_multi:
                print(f'{metric}')
                r = r_multi[metric]
                for j in r["importances_mean"].argsort()[::-1]:
                    # if r["importances_mean"][i] - 2 * r["importances_std"][i] > 0:
                    feature_dict = {
                        "feature_name": columns[j],  # Replace with your feature names
                        "importance_mean": r["importances_mean"][j],
                        "importance_std": r["importances_std"][j],
                        "metric": metric,
                    }
                    # print(
                    #     f'''{project_name.upper()} {columns[j]:<8} {r["importances_mean"][j]:.3f} +/- {r["importances_std"][j]:.3f}''')
                    feature_importance_list.append(feature_dict)
                new_df = pd.DataFrame(feature_importance_list)
                best_model_importance = pd.concat([best_model_importance, new_df], ignore_index=True)
            model_performance_data.append(round_results_pu)
            # results_df = results_df.append(round_results_pu, ignore_index=True)
            print(
                f'''{project_name.upper()} at iteration {i} PULearning -- precision:{precision}, recall:{recall}, f1:{f1}''')
            precision_pu_round.append(precision)
            recall_pu_round.append(recall)
            f1_pu_round.append(f1)
            # auc_pu_round.append(auc_score)

            from sklearn.ensemble import RandomForestClassifier

            newRF = RandomForestClassifier(
                n_estimators=100,
                bootstrap=True,
                n_jobs=1,
            )
            newRF.fit(pu_data_with_features_only, pu_labels)
            X_pred = test_data_only_with_features.values
            y_pred = newRF.predict(X_pred)
            precision = precision_score(np.ones(len(test_data_only_with_features)), y_pred, zero_division=1)
            recall = recall_score(np.ones(len(test_data_only_with_features)), y_pred, zero_division=1)
            f1 = f1_score(np.ones(len(test_data_only_with_features)), y_pred, zero_division=1)
            # auc = roc_auc_score(np.ones(len(test_data_only_with_features)), y_pred, zero_division=1)

            precision_rf_round.append(precision)
            recall_rf_round.append(recall)
            f1_rf_round.append(f1)
            round_results_rf = {
                "dropout": i,
                "f1": f1,
                "recall": recall,
                "model": "RandomForest",
                "iteration": j
            }
            model_performance_data.append(round_results_rf)
            print(f'''{project_name.upper()} Normal Random Forest -- precision:{precision}, recall:{recall}, f1:{f1}''')
            train_data = positive_data.copy()
            test_data_only_with_features = pd.DataFrame()
            train_data = train_data.drop("label", axis=1)

        precision_pu.append(np.mean(precision_pu_round))
        recall_pu.append(np.mean(recall_pu_round))
        f1_pu.append(np.mean(f1_pu_round))

        precision_rf.append(np.mean(precision_rf_round))
        recall_rf.append(np.mean(recall_rf_round))
        f1_rf.append(np.mean(f1_rf_round))

    X_pred = df_for_prediction.values
    y_pred = best_model.predict(X_pred)
    prediction_df["prediction"] = y_pred
    print('/'.join(base_path.split("/") + ['prediction_with_LOOCV_results', f'''{project_name}_prediction.csv''']))
    prediction_df.to_csv(
        '/'.join(base_path.split("/") + ['prediction_with_LOOCV_results', f'''{project_name}_prediction.csv''']),
        index=False)
    results_df = pd.DataFrame(model_performance_data)
    print(
        f"{project_name.upper()} Writting to {base_path}/training_with_LOOCV_performance/{project_name}_model_performance.csv")
    results_df.to_csv(f'''{base_path}/training_with_LOOCV_performance/{project_name}_model_performance.csv''',
                      index=False)
    best_model_importance.to_csv(f'''{base_path}/training_with_LOOCV_performance/{project_name}_model_importance.csv''',
                                 index=False)

# print(pu_combined)
# break
# print("Training data")
# print(pu_data[['issue_id', 'comment_id', 'label', 'issue_id_int']])
