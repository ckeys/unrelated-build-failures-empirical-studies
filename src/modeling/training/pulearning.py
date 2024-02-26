import pandas as pd
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.utils import resample
from sklearn.metrics import precision_score, recall_score, f1_score
from pulearn import (
    ElkanotoPuClassifier
)
import matplotlib.pyplot as plt
import os
import time


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
                   "Has Code Patch", "CI Latency", "Night Time", "Weekend"],
        'hadoop': ["Number of Parallel Issues", "Number of Prior Comments", "Number of Similar Failures",
                   "Is Shared Same Emsg", "Is dependent", "Is Duplicate", "Is Reference", "Is Cloners", "Is Regression",
                   "Is Blocker", "Is Supercedes", "Is Container", "Is Required", "Is Problem/Incident", "Is Blocked",
                   "Is Dependent", "Is Completes", "Is Child-Issue", "Config Lines Modified", "Has Code Patch",
                   "CI Latency", "Night Time", "Weekend"],
        'hbase': ["Number of Parallel Issues", "Number of Prior Comments", "Number of Similar Failures",
                  "Is Shared Same Emsg", "Is Incorporates", "Is Reference", "Is dependent", "Is Duplicate",
                  "Is Supercedes", "Is Cloners", "Is Regression", "Is Required", "Is Dependent", "Is Container",
                  "Is Child-Issue", "Is Problem/Incident", "Is Blocked", "Is Completes", "Config Lines Modified",
                  "Has Code Patch", "CI Latency", "Night Time", "Weekend"],
        'hdds': ["Number of Parallel Issues", "Number of Prior Comments", "Number of Similar Failures",
                 "Is Shared Same Emsg", "Is Duplicate", "Is Reference", "Is Blocker", "Is Cloners", "Is Child-Issue",
                 "Is dependent", "Is Problem/Incident", "Is Regression", "Is Container", "Is Dependent",
                 "Is Supercedes", "Is Required", "Is Blocked", "Is Dependency", "Config Lines Deleted",
                 "Config Lines Modified", "Has Code Patch", "CI Latency", "Night Time", "Weekend"],
        'hdfs': ["Number of Parallel Issues", "Number of Prior Comments", "Number of Similar Failures",
                 "Is Shared Same Emsg", "Is Reference", "Is Incorporates", "Is Blocker", "Is Container", "Is dependent",
                 "Is Supercedes", "Is Regression", "Is Cloners", "Is Required", "Is Problem/Incident", "Is Dependent",
                 "Is Child-Issue", "Is Blocked", "Is Completes", "Is Parent Feature", "Is Dependency",
                 "Config Lines Deleted", "Config Lines Modified", "Has Code Patch", "CI Latency", "Night Time",
                 "Weekend"],
        'hive': ["Number of Parallel Issues", "Number of Prior Comments", "Number of Similar Failures",
                 "Is Shared Same Emsg", "Is dependent", "Is Blocker", "Is Reference", "Is Incorporates",
                 "Is Regression", "Is Cloners", "Is Child-Issue", "Is Required", "Is Supercedes", "Is Container",
                 "Is Problem/Incident", "Is Blocked", "Is Completes", "Is Dependent", "Is Parent Feature",
                 "Is Dependency", "Config Lines Modified", "Has Code Patch", "CI Latency", "Night Time", "Weekend"],
        'yarn': ["Number of Parallel Issues", "Number of Prior Comments", "Number of Similar Failures",
                 "Is Shared Same Emsg", "Is Incorporates", "Is Reference", "Is dependent", "Is Duplicate",
                 "Is Regression", "Is Cloners", "Is Required", "Is Supercedes", "Is Container", "Is Blocked",
                 "Is Problem/Incident", "Is Child-Issue", "Is Dependency", "Is Dependent", "Is Completes",
                 "Config Lines Deleted", "Config Lines Modified", "Has Code Patch", "CI Latency", "Night Time",
                 "Weekend"]
    }
    selected_columns = ['project_name', 'comment_id', 'issue_id','label'] + selection_map[project]
    return df[selected_columns]


def reliable_negative_instances_selection(P, U, sample_ratio):
    # Step 1: Initialize RN as an empty set
    RN = set()

    # Step 2: Sample s% of instances from P as S
    num_samples = int(len(P) * sample_ratio)
    S = P.sample(num_samples)

    # Step 3: Create Ps and Us with appropriate labels
    Ps = P.drop(S.index).assign(label=1)
    Us = U.append(S).assign(label=-1)

    # Step 4: Train a classifier LR with Ps and Us
    X_train = pd.concat([Ps.drop('label', axis=1), Us.drop('label', axis=1)])
    y_train = pd.concat([Ps['label'], Us['label']])
    LR = LogisticRegression()
    LR.fit(X_train, y_train)

    # Step 5: Classify instances in U using LR and output class-conditional probabilities
    probabilities = LR.predict_proba(U.drop('label', axis=1))

    # Step 6: Select a threshold θ based on class-conditional probabilities of instances in S
    threshold = probabilities[S.index].mean()

    # Step 7-9: Iterate through instances in U and add to RN if P(1|d) ≤ θ
    for idx, prob in enumerate(probabilities):
        if prob[1] <= threshold:
            RN.add(U.index[idx])

    # Step 10: Output RN
    return RN


def performance_plot(project_name, base_path):
    data = pd.read_csv(f'''{base_path}/performance/{project_name}_model_performance.csv''')
    # Assuming you have a DataFrame named 'data' with the columns 'dropout', 'f1', 'recall', 'model', and 'iteration'

    # Filter the data for the desired models
    pu_data = data[data['model'] == 'PULearning']
    rf_data = data[data['model'] == 'RandomForest']

    # Group the data by 'dropout' and calculate the mean for 'f1' and 'recall'
    pu_mean = pu_data.groupby('dropout')[['f1', 'recall']].mean()
    rf_mean = rf_data.groupby('dropout')[['f1', 'recall']].mean()

    # Plot the mean values
    plt.plot(pu_mean.index, pu_mean['f1'], label='PULearning')
    # plt.plot(pu_mean.index, pu_mean['recall'], label='PULearning (Recall)')
    plt.plot(rf_mean.index, rf_mean['f1'], label='RandomForest')
    # plt.plot(rf_mean.index, rf_mean['recall'], label='RandomForest (Recall)')

    # Set plot labels and title
    plt.xlabel('Dropout')
    plt.ylabel('Mean F1/Recall')
    plt.title(f'''Mean F1/Recall vs Dropout - {project_name}''')

    # Add legend
    plt.legend()
    # Show the plot
    plt.show()


def PULearning(benchmark_df, df, project_name, base_path):
    df = feature_selection(project_name, df)
    prediction_df = df
    # df = df.drop(columns=['project_name', 'comment_id', 'issue_id', ])
    # positive_data -> train_data
    # df_for_prediction
    # unlabeled_data
    #
    # label_ratio = float(len(benchmark_df)) / float(len(df))
    label_ratio = 0.5
    print(f'''Label Ratio is : {label_ratio}''')
    # Split the data into positive (labeled) and unlabeled portions
    positive_data = df[df["label"] == 1].reset_index(drop=True)
    df_for_prediction = df.drop("label", axis=1)
    unlabeled_data = df[df["label"].isnull()]
    num_positive_samples = len(benchmark_df[benchmark_df['label'] == 1])
    # Define the random seed for reproducibility
    random.seed(42)
    train_data = positive_data.copy()
    val_data = pd.DataFrame()
    train_data = train_data.drop("label", axis=1)
    columns = train_data.columns.tolist()
    precision_pu = []
    recall_pu = []
    f1_pu = []
    precision_rf = []
    recall_rf = []
    f1_rf = []
    model_performance_data = []
    best_model_recall = 0
    best_model = None
    best_model_importance = pd.DataFrame(columns=['feature_name', 'importance_mean', 'importance_std'])
    best_model_permutation_importance = None
    for dropout in range(1, len(benchmark_df[benchmark_df['label'] == 1]) - 1):

        if dropout > (len(train_data)):
            print(f'''dropout is {dropout} but len of train data is {len(train_data)}''')
            break

        precision_pu_round = []
        recall_pu_round = []
        f1_pu_round = []
        precision_rf_round = []
        recall_rf_round = []
        f1_rf_round = []
        for i in range(100):
            random_rows = random.sample(range(len(train_data)), dropout)
            removed_rows = train_data.iloc[random_rows]
            # Remove the selected rows from train_df
            train_data = train_data.drop(random_rows)
            train_data.reset_index(drop=True, inplace=True)
            # train_data = train_data.drop("label", axis=1)
            # Concatenate the removed rows with unlabeled_df
            val_data = pd.concat([val_data, removed_rows])
            # val_data = val_data.drop("label", axis=1)
            # Combine positive data with unlabeled data (for PU learning)
            print(
                f'''Now dropout size is {dropout}, Training data size is {len(train_data)}, Validation data size is {len(val_data)}''')
            unlabeled_data_resampled = resample(unlabeled_data, n_samples=int(
                (num_positive_samples / label_ratio)) - num_positive_samples,
                                                random_state=42)
            pu_data = pd.concat([train_data, unlabeled_data_resampled.drop("label", axis=1)])
            # Set up the PU learning scenario
            pu_labels = np.concatenate([np.ones(len(train_data)), np.full(len(unlabeled_data_resampled), -1)]).astype(
                int)
            print(f'''This is the distribution of label:{np.unique(pu_labels, return_counts=True)}''')
            # Combine pu_data and pu_labels into a single array
            pu_combined = np.column_stack((pu_data, pu_labels))
            print(f'''pu data size {len(pu_data)}''')
            # Shuffle the combined array
            np.random.shuffle(pu_combined)
            # Separate the shuffled data and labels
            pu_data = pu_combined[:, :-1]
            print(f'''shuffled data size {len(pu_data)}''')
            pu_labels = pu_combined[:, -1].astype(int)
            from sklearn.ensemble import RandomForestClassifier
            estimator = RandomForestClassifier(
                n_estimators=100,
                bootstrap=True,
                n_jobs=1,
            )
            pu_estimator = ElkanotoPuClassifier(estimator=estimator, hold_out_ratio=0.1)
            print(f'''{pu_data.shape},{pu_labels.shape}''')
            pu_estimator.fit(pu_data, pu_labels)
            importance = pu_estimator.estimator.feature_importances_

            combined_dict = dict(zip(columns, importance))
            sorted_dict = dict(sorted(combined_dict.items(), key=lambda x: x[1], reverse=True))
            for k, v in sorted_dict.items():
                print(f'''{project_name.upper()} feature {k} score is {v}''')

            X_pred = val_data.values
            y_pred = pu_estimator.predict(X_pred)
            precision = precision_score(np.ones(len(val_data)), y_pred, zero_division=1)
            recall = recall_score(np.ones(len(val_data)), y_pred, zero_division=1)
            f1 = f1_score(np.ones(len(val_data)), y_pred, zero_division=1)
            round_results_pu = {
                "dropout": dropout,
                "f1": f1,
                "recall": recall,
                "model": "PULearning",
                "iteration": i
            }

            if recall >= 0.85 and dropout <= 50:
                best_model = pu_estimator

            new_df = pd.DataFrame(columns=['feature_name', 'importance_mean', 'importance_std'])
            # if recall > best_model_recall:
            # best_model_importance = pd.DataFrame(columns=['feature_name', 'importance_mean', 'importance_std'])
            best_model_recall = recall
            best_model = pu_estimator
            feature_importance_list = []
            if recall >= 0.8:
                scoring = ['recall', 'f1']
                r_multi = permutation_importance(pu_estimator.estimator, X_pred, np.ones(len(val_data)), n_repeats=10,
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
                        print(
                            f'''{project_name.upper()} {columns[j]:<8} {r["importances_mean"][j]:.3f} +/- {r["importances_std"][j]:.3f}''')
                        feature_importance_list.append(feature_dict)
                new_df = pd.DataFrame(feature_importance_list)
                # print(feature_importance_list)
                best_model_importance = pd.concat([best_model_importance, new_df], ignore_index=True)
                # best_model_importance = best_model_importance.append(list(combined_dict.items()), ignore_index=True)
            model_performance_data.append(round_results_pu)
            # results_df = results_df.append(round_results_pu, ignore_index=True)
            print(
                f'''{project_name.upper()} at iteration {i} PULearning -- precision:{precision}, recall:{recall}, f1:{f1}''')
            precision_pu_round.append(precision)
            recall_pu_round.append(recall)
            f1_pu_round.append(f1)

            from sklearn.ensemble import RandomForestClassifier
            newRF = RandomForestClassifier(
                n_estimators=100,
                bootstrap=True,
                n_jobs=1,
            )
            newRF.fit(pu_data, pu_labels)
            X_pred = val_data.values
            y_pred = newRF.predict(X_pred)
            precision = precision_score(np.ones(len(val_data)), y_pred, zero_division=1)
            recall = recall_score(np.ones(len(val_data)), y_pred, zero_division=1)
            f1 = f1_score(np.ones(len(val_data)), y_pred, zero_division=1)
            precision_rf_round.append(precision)
            recall_rf_round.append(recall)
            f1_rf_round.append(f1)
            round_results_rf = {
                "dropout": dropout,
                "f1": f1,
                "recall": recall,
                "model": "RandomForest",
                "iteration": i
            }
            model_performance_data.append(round_results_rf)
            print(f'''{project_name.upper()} Normal Random Forest -- precision:{precision}, recall:{recall}, f1:{f1}''')
            train_data = positive_data.copy()
            val_data = pd.DataFrame()
            train_data = train_data.drop("label", axis=1)
        precision_pu.append(np.mean(precision_pu_round))
        recall_pu.append(np.mean(recall_pu_round))
        f1_pu.append(np.mean(f1_pu_round))

        precision_rf.append(np.mean(precision_rf_round))
        recall_rf.append(np.mean(recall_rf_round))
        f1_rf.append(np.mean(f1_rf_round))

    X_pred = df_for_prediction.values
    y_pred = best_model.predict(X_pred)
    prediction_df["new_label"] = y_pred
    print('/'.join(base_path.split("/")[:-1] + ['prediction_results', f'''{project_name}_prediction.csv''']))
    prediction_df.to_csv('/'.join(base_path.split("/")[:-1] + ['prediction_results', f'''{project_name}_prediction.csv''']),
                         index=False)
    results_df = pd.DataFrame(model_performance_data)
    print(f"{project_name.upper()} Writting to {base_path}/training_performance/{project_name}_model_performance.csv")
    results_df.to_csv(f'''{base_path}/training_performance/{project_name}_model_performance.csv''',
                      index=False)
    best_model_importance.to_csv(f'''{base_path}/training_performance/{project_name}_model_importance.csv''', index=False)


current_file_path = os.path.abspath(__file__)
base_path = '/'.join(current_file_path.split('/')[:-4] + ['data', 'modeling_data','training'])
print(base_path)
project_name_group = ['hive', 'hadoop', 'yarn', 'hdfs', 'hbase', 'ambari', 'hdds']
start_time = time.time()
for project_name in project_name_group:
    benchmark_data = pd.read_csv(f'''{base_path}/labeled_data/{project_name}_labeled_data.csv''')
    benchmark_data = benchmark_data[benchmark_data['label'].notna()]
    print(benchmark_data)
    benchmark_data = benchmark_data[['issue_id', 'comment_id']]
    # Set labels in benchmark_data to 1
    benchmark_data['label'] = 1
    # Get unique 'issue_id' and 'comment_id' in benchmark_data
    benchmark_ids = benchmark_data[['issue_id', 'comment_id']].values

    df1 = pd.read_csv(f'''{base_path}/features/{project_name}_data_with_features.csv''')
    df1['issue_id'] = df1['project_name'].str.cat(df1['issue_id'].astype(str), sep='-')
    unlabeled_value = np.nan
    # Set labels in df1 to NaN for entries not present in benchmark_data
    df1['label'] = unlabeled_value
    # df1['label'] = df1.apply(lambda row: 1 if [row['issue_id'], row['comment_id']] in benchmark_ids else
    # unlabeled_value, axis=1)
    df1 = df1.merge(benchmark_data, on=['issue_id', 'comment_id'], how='left', suffixes=('_df1', '_benchmark'))
    df1['label'] = df1.apply(lambda row: 1 if row['label_benchmark'] == 1 else row['label_df1'], axis=1)
    df1.drop(['label_benchmark'], axis=1, inplace=True)
    df1.drop('label_df1', axis=1, inplace=True)
    df = df1
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
    # df['comment_created_at'] = pd.to_timestamp(df['comment_created_at'], format='%a %d %b %Y %H:%M:%S %z')

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
    # df['comment_created_at'] = pd.to_datetime(df['comment_created_at'])
    # print(df['comment_created_at'].dtype)
    # df['comment_created_at'] = pd.to_datetime(df['comment_created_at'], format='%Y-%m-%d %H:%M:%S%z')
    # Extract hour from the timestamp
    df['comment_created_at'] = pd.to_datetime(df['comment_created_at'], format='%Y-%m-%d %H:%M:%S%z', utc=True)
    df['hour'] = df['comment_created_at'].dt.strftime('%H').astype(int)

    # Determine if it's daily time (between 6 AM and 6 PM)
    df['is_daily_time'] = df['hour'].between(6, 18).astype(int)

    # Determine if it's night time (between 6 PM and 6 AM)
    # df['is_night_time'] = (~df['is_daily_time']).astype(int)
    df['is_night_time'] = (df['hour'] >= 19) | (df['hour'] < 6)
    df['is_night_time'] = df['is_night_time'].astype(int)

    # Determine if it's a weekday (Monday=0, Sunday=6)
    df['weekday'] = df['comment_created_at'].dt.weekday
    df['is_weekday'] = df['weekday'].isin(range(0, 5)).astype(int)

    # Determine if it's a weekend (Saturday or Sunday)
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)

    # Drop the intermediate columns if needed
    df = df.drop(['hour', 'weekday'], axis=1)

    columns_to_drop = ['type_id', 'priority_id', 'has_contain_Test_files', 'lines_of_Test_classes_modified',
                       'lines_of_Test_classes_added', 'lines_of_Test_classes_deleted', 'lines_of_Test_files',
                       'deletions', 'insertions', 'files', 'Time_difference', 'commit_num', 'is_share_similar_emsg',
                       'is_share_same_emsg', 'num_parrel_commits', 'num_of_developers']

    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    print(df.columns)
    df.drop(['created', 'created_ts', 'comment_created_at', 'comment_created_at_ts', 'time', 'day'], axis=1,
            inplace=True)
    # List of columns to exclude from filling NaN values with 0
    columns_to_exclude = ['label']
    # Fill NaN values with 0 in all columns except 'label'
    df.loc[:, ~df.columns.isin(columns_to_exclude)] = df.loc[:, ~df.columns.isin(columns_to_exclude)].fillna(0)
    df = column_rename(df)
    PULearning(benchmark_data, df, project_name, base_path)
# Record the end time
end_time = time.time()
# Calculate the execution time
execution_time = end_time - start_time
# Convert execution time to minutes
execution_time_minutes = execution_time / 60
# Print the execution time in minutes
print(f"Execution time: {execution_time_minutes:.2f} minutes")
