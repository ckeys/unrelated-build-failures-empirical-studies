import os
import pandas as pd
import numpy as np
from src.data_statistics.pqn_dataset_extraction import create_datasets
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.utils import resample
from sklearn.metrics import precision_score, recall_score, f1_score, auc, roc_curve, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from pulearn import ElkanotoPuClassifier, WeightedElkanotoPuClassifier
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
from sklearn.dummy import DummyClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set Pandas display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def preprocess_data(df, project_name):
    """
    Preprocesses the input DataFrame by performing code churn processing,
    comment date processing, dropping unused columns, renaming columns, and filling NaNs.

    Args:
        df (pd.DataFrame): The input DataFrame.
        project_name (str): Name of the project for feature selection.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    df = code_churn_preprocessing(df)
    df = comment_date_preprocessing(df)
    df = drop_unused_columns(df)
    df = column_rename(df)

    if project_name == 'hive':
        df['num_of_tests_failed'] = df['num_of_tests_failed'].apply(lambda x: 0 if x == -1 else x)

    df = fill_nans(df)
    # Uncomment the line below if feature selection is needed
    # df = feature_selection(project_name, df)
    return df


def code_churn_preprocessing(df):
    # Adjust the value of has_source_code column based on conditions
    df['has_source_code'] = np.where(
        (df['lines_of_source_code_added'] != 0) |
        (df['lines_of_source_code_deleted'] != 0) |
        (df['lines_of_source_code_modified'] != 0), 1, 0
    )

    df['has_config_files'] = np.where(
        (df['lines_of_config_file_added'] != 0) |
        (df['lines_of_config_file_deleted'] != 0) |
        (df['lines_of_config_file_modified'] != 0), 1, 0
    )

    # Set the value of has_contains_code_patch based on conditions
    df['has_contains_code_patch'] = np.where(
        (df['has_config_files'] == 1) | (df['has_source_code'] == 1), 1, 0
    )
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
    columns_to_drop = [
        'type_id', 'priority_id', 'has_contain_Test_files', 'lines_of_Test_classes_modified',
        'lines_of_Test_classes_added', 'lines_of_Test_classes_deleted', 'lines_of_Test_files',
        'deletions', 'insertions', 'files', 'Time_difference', 'commit_num', 'is_share_similar_emsg',
        'is_share_same_emsg', 'num_parrel_commits', 'num_of_developers', 'created', 'created_ts',
        'comment_created_at', 'time', 'day'
    ]
    # Drop only the columns that exist in the DataFrame
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    return df


def column_rename(df):
    """
    Rename the column names based on a predefined mapping.
    """
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
    # List of columns to exclude from filling NaN values
    columns_to_exclude = ['label']
    # Fill NaN values with 0 in all columns except 'label'
    df.loc[:, ~df.columns.isin(columns_to_exclude)] = df.loc[:, ~df.columns.isin(columns_to_exclude)].fillna(0)
    return df


def feature_selection(project, df):
    """
    Selects features based on the project.
    """
    selection_map = {
        "ambari": [...],  # Same as your existing selection_map
        # Add other projects here
    }
    selected_columns = ['project_name', 'comment_id', 'issue_id', 'label', 'comment_created_at_ts'] + selection_map.get(
        project, [])
    return df[selected_columns]


def select_features_by_spearman(df, correlation_threshold=0.7):
    preference_list = ['CI Latency', 'Number of Parallel Issues', 'Is Cross Projects', 'Number of Prior Comments',
                       'Number of Similar Failures', 'Is Shared Same Emsg', 'Has Source Code', 'Has Code Patch',
                       'Has Config Files']

    # Step 1: Temporarily remove the columns to exclude
    columns_to_hide = ['project_name', 'comment_id', 'issue_id', 'label', 'comment_created_at_ts']
    excluded_columns = df[columns_to_hide]  # Save excluded columns separately
    df_dropped = df.drop(columns=columns_to_hide, errors='ignore')  # Drop them from the main DataFrame

    # Step 2: Compute the Spearman correlation matrix on the remaining columns
    corr_matrix = df_dropped.corr(method='spearman').abs()

    # Step 3: Extract upper triangle of the correlation matrix to avoid redundancy
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Step 4: Initialize a set to collect columns to drop
    to_drop = set()

    # Step 5: Iterate through columns to find pairs with correlation > threshold
    for col in upper_tri.columns:
        # Find highly correlated features with the current column
        high_corr_features = [idx for idx, val in upper_tri[col].items() if val >= correlation_threshold]
        for feature in high_corr_features:
            # Check preference list to decide which feature to keep
            if preference_list:
                if col in preference_list and feature in preference_list:
                    # Keep the one that appears first in the preference list
                    to_keep = col if preference_list.index(col) < preference_list.index(feature) else feature
                    to_drop.add(feature if to_keep == col else col)
                elif col in preference_list:
                    # If only `col` is in the preference list, keep it
                    to_drop.add(feature)
                elif feature in preference_list:
                    # If only `feature` is in the preference list, keep it
                    to_drop.add(col)
                else:
                    # If neither is in the preference list, arbitrarily drop the second
                    to_drop.add(feature)
            else:
                # Default behavior if no preference list is provided (keep the first)
                to_drop.add(feature)

    # Step 6: Drop the selected features due to high correlation
    df_reduced = df_dropped.drop(columns=list(to_drop), errors='ignore')
    # Step 7: Bring back the temporarily excluded columns by concatenating them
    df_final = pd.concat([df_reduced, excluded_columns], axis=1)

    # Step 7: Return the df_final DataFrame and the list of dropped columns
    return df_final, to_drop


def calculate_metrics(true_labels, predicted_labels, predicted_probabilities):
    """Calculate precision, recall, F1 score, and AUC."""
    predicted_labels = np.where(predicted_labels, 1, -1)

    a = np.sum((true_labels == 1) & (predicted_labels == 1))
    b = np.sum((true_labels == 1) & (predicted_labels == -1))
    c = np.sum((true_labels == -1) & (predicted_labels == 1))

    # Precision, recall, and F1 score
    precision = a / (a + c) if (a + c) != 0 else 0
    recall = a / (a + b) if (a + b) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # AUC
    auc = roc_auc_score(true_labels, predicted_probabilities)
    return precision, recall, f1_score, auc


def random_model(P, Q, N):
    """Perform random model using K-Fold cross-validation."""
    P_features, P_labels = P.drop(columns=['label']), P['label']
    Q_features, Q_labels = Q.drop(columns=['label']), Q['label']
    N_features, N_labels = N.drop(columns=['label']), N['label']

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    all_true_labels, all_pred_probabilities = [], []

    for (train_idx_P, test_idx_P), (train_idx_Q, test_idx_Q), (train_idx_N, test_idx_N) in zip(
            kf.split(P_features), kf.split(Q_features), kf.split(N_features)):
        # Extract test data
        P_test = P_features.iloc[test_idx_P]
        Q_test = Q_features.iloc[test_idx_Q]
        N_test = N_features.iloc[test_idx_N]

        P_Q_test_labels = pd.concat([P_labels.iloc[test_idx_P], Q_labels.iloc[test_idx_Q]])

        # Generate random probabilities and predictions
        P_Q_test_prob = np.random.rand(len(P_test) + len(Q_test))
        N_test_prob = np.random.rand(len(N_test))
        P_Q_test_pred = np.where(P_Q_test_prob > 0.5, 1, -1)
        N_test_pred = np.where(N_test_prob > 0.5, 1, -1)

        all_true_labels.extend(P_Q_test_labels.tolist() + N_labels.iloc[test_idx_N].tolist())
        all_pred_probabilities.extend(P_Q_test_prob.tolist() + N_test_prob.tolist())

    precision, recall, f1_score, auc = calculate_metrics(np.array(all_true_labels),
                                                         np.array(all_pred_probabilities) > 0.5, all_pred_probabilities)
    return precision, recall, f1_score, auc


def elkanoto_pu_learning(P, Q, N):
    """Perform PU learning using ElkanotoPuClassifier with K-Fold cross-validation."""
    P_features, P_labels = P.drop(columns=['label']), P['label']
    Q_features, Q_labels = Q.drop(columns=['label']), Q['label']
    N_features, N_labels = N.drop(columns=['label']), N['label']

    estimator = RandomForestClassifier(n_estimators=100, criterion='gini', bootstrap=True, n_jobs=-1)
    pu_estimator = ElkanotoPuClassifier(estimator=estimator, hold_out_ratio=0.1)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    all_true_labels, all_pred_probabilities = [], []

    for (train_idx_P, test_idx_P), (train_idx_Q, test_idx_Q), (train_idx_N, test_idx_N) in zip(
            kf.split(P_features), kf.split(Q_features), kf.split(N_features)):
        # Combine training data
        pu_train_data = pd.concat(
            [P_features.iloc[train_idx_P], Q_features.iloc[train_idx_Q], N_features.iloc[train_idx_N]])
        pu_train_labels = pd.concat(
            [P_labels.iloc[train_idx_P], Q_labels.iloc[train_idx_Q], N_labels.iloc[train_idx_N]])

        # Train the PU estimator
        pu_estimator.fit(pu_train_data.values, pu_train_labels.values)

        # Combine testing data
        P_Q_test_data = pd.concat([P_features.iloc[test_idx_P], Q_features.iloc[test_idx_Q]])
        P_Q_test_labels = pd.concat([P_labels.iloc[test_idx_P], Q_labels.iloc[test_idx_Q]])

        # Predict on test data
        P_Q_test_pred = pu_estimator.predict(P_Q_test_data.values)
        N_test_pred = pu_estimator.predict(N_features.iloc[test_idx_N].values)

        # Predict probabilities for AUC
        P_Q_test_prob = pu_estimator.predict_proba(P_Q_test_data.values)
        N_test_prob = pu_estimator.predict_proba(N_features.iloc[test_idx_N].values)

        all_true_labels.extend(P_Q_test_labels.tolist() + N_labels.iloc[test_idx_N].tolist())
        all_pred_probabilities.extend(P_Q_test_prob.tolist() + N_test_prob.tolist())

    precision, recall, f1_score, auc = calculate_metrics(np.array(all_true_labels),
                                                         np.array(all_pred_probabilities) > 0.5, all_pred_probabilities)
    return precision, recall, f1_score, auc


def weighted_elkanoto_pu_learning(P, Q, N, feature_names):
    """Perform PU learning using K-Fold cross-validation."""
    P_features, P_labels = P.drop(columns=['label']), P['label']
    Q_features, Q_labels = Q.drop(columns=['label']), Q['label']
    N_features, N_labels = N.drop(columns=['label']), N['label']

    estimator = RandomForestClassifier(n_estimators=100, criterion='gini', bootstrap=True, n_jobs=-1)
    pu_estimator = WeightedElkanotoPuClassifier(estimator=estimator, labeled=len(P), unlabeled=len(Q) + len(N))

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    all_true_labels, all_pred_probabilities = [], []
    feature_importances = np.zeros(len(feature_names))  # Initialize to collect feature importance for this iteration

    for (train_idx_P, test_idx_P), (train_idx_Q, test_idx_Q), (train_idx_N, test_idx_N) in zip(
            kf.split(P_features), kf.split(Q_features), kf.split(N_features)):
        # Combine training data
        pu_train_data = pd.concat(
            [P_features.iloc[train_idx_P], Q_features.iloc[train_idx_Q], N_features.iloc[train_idx_N]])
        pu_train_labels = pd.concat(
            [P_labels.iloc[train_idx_P], Q_labels.iloc[train_idx_Q], N_labels.iloc[train_idx_N]])

        # Train the PU estimator
        pu_estimator.fit(pu_train_data.values, pu_train_labels.values)
        feature_importances += pu_estimator.estimator.feature_importances_

        # Combine evaluation data
        P_Q_test_data = pd.concat([P_features.iloc[test_idx_P], Q_features.iloc[test_idx_Q]])
        P_Q_test_labels = pd.concat([P_labels.iloc[test_idx_P], Q_labels.iloc[test_idx_Q]])

        # Predict on test data
        P_Q_test_pred = pu_estimator.predict(P_Q_test_data.values)
        N_test_pred = pu_estimator.predict(N_features.iloc[test_idx_N].values)

        # Predict probabilities for AUC
        P_Q_test_prob = pu_estimator.predict_proba(P_Q_test_data.values)
        N_test_prob = pu_estimator.predict_proba(N_features.iloc[test_idx_N].values)

        all_true_labels.extend(P_Q_test_labels.tolist() + N_labels.iloc[test_idx_N].tolist())
        all_pred_probabilities.extend(P_Q_test_prob.tolist() + N_test_prob.tolist())

    # Average feature importances over K folds
    feature_importances /= kf.get_n_splits()

    precision, recall, f1_score, auc = calculate_metrics(np.array(all_true_labels),
                                                         np.array(all_pred_probabilities) > 0.5, all_pred_probabilities)
    return precision, recall, f1_score, auc, feature_importances


def random_forest_benchmark(P, Q, N):
    """Perform RandomForest classification using K-Fold cross-validation."""
    P_features, P_labels = P.drop(columns=['label']), P['label']
    Q_features, Q_labels = Q.drop(columns=['label']), Q['label']
    N_features, N_labels = N.drop(columns=['label']), N['label']

    # Initialize the standard RandomForestClassifier
    estimator = RandomForestClassifier(n_estimators=100, criterion='gini', bootstrap=True, n_jobs=-1)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    all_true_labels, all_pred_probabilities = [], []

    for (train_idx_P, test_idx_P), (train_idx_Q, test_idx_Q), (train_idx_N, test_idx_N) in zip(
            kf.split(P_features), kf.split(Q_features), kf.split(N_features)):
        # Combine training data
        train_data = pd.concat(
            [P_features.iloc[train_idx_P], Q_features.iloc[train_idx_Q], N_features.iloc[train_idx_N]])
        train_labels = pd.concat([P_labels.iloc[train_idx_P], Q_labels.iloc[train_idx_Q], N_labels.iloc[train_idx_N]])

        # Train the RandomForest estimator
        estimator.fit(train_data.values, train_labels.values)

        # Combine evaluation data
        P_Q_test_data = pd.concat([P_features.iloc[test_idx_P], Q_features.iloc[test_idx_Q]])
        P_Q_test_labels = pd.concat([P_labels.iloc[test_idx_P], Q_labels.iloc[test_idx_Q]])

        # Predict on test data
        P_Q_test_pred = estimator.predict(P_Q_test_data.values)
        N_test_pred = estimator.predict(N_features.iloc[test_idx_N].values)

        # Predict probabilities for AUC
        P_Q_test_prob = estimator.predict_proba(P_Q_test_data.values)[:, 1]
        N_test_prob = estimator.predict_proba(N_features.iloc[test_idx_N].values)[:, 1]

        all_true_labels.extend(P_Q_test_labels.tolist() + N_labels.iloc[test_idx_N].tolist())
        all_pred_probabilities.extend(P_Q_test_prob.tolist() + N_test_prob.tolist())

    precision, recall, f1_score, auc = calculate_metrics(np.array(all_true_labels),
                                                         np.array(all_pred_probabilities) > 0.5, all_pred_probabilities)
    return precision, recall, f1_score, auc


def constant_positive_model(P, Q, N):
    """Perform constant positive model using K-Fold cross-validation."""
    # Extract features and labels from P, Q, N
    P_features, P_labels = P.drop(columns=['label']), P['label']
    Q_features, Q_labels = Q.drop(columns=['label']), Q['label']
    N_features, N_labels = N.drop(columns=['label']), N['label']

    # Define the constant positive classifier
    constant_positive_clf = DummyClassifier(strategy="constant", constant=1)

    # Prepare cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    all_true_labels, all_pred_probabilities = [], []

    # Perform K-fold cross-validation for P, Q, and N
    for (train_idx_P, test_idx_P), (train_idx_Q, test_idx_Q), (train_idx_N, test_idx_N) in zip(
            kf.split(P_features), kf.split(Q_features), kf.split(N_features)):
        # Extract test data
        P_test = P_features.iloc[test_idx_P]
        Q_test = Q_features.iloc[test_idx_Q]
        N_test = N_features.iloc[test_idx_N]

        P_Q_test_labels = pd.concat([P_labels.iloc[test_idx_P], Q_labels.iloc[test_idx_Q]])
        N_test_labels = N_labels.iloc[test_idx_N]

        # Combine test features for P and Q
        P_Q_test_features = pd.concat([P_test, Q_test])

        # Fit the Dummy Classifier on the combined training data (P, Q, N)
        constant_positive_clf.fit(P_Q_test_features,
                                  P_Q_test_labels)  # Fit on features and labels (though it always predicts constant)

        # Predict constant value (1) for P, Q, and N
        P_Q_test_pred_prob = constant_positive_clf.predict_proba(P_Q_test_features)[:, 0]  # Probabilities for class 1
        N_test_pred_prob = constant_positive_clf.predict_proba(N_test)[:, 0]

        # Append true labels and predicted probabilities
        all_true_labels.extend(P_Q_test_labels.tolist() + N_test_labels.tolist())
        all_pred_probabilities.extend(P_Q_test_pred_prob.tolist() + N_test_pred_prob.tolist())

    # Calculate metrics based on true labels and predicted probabilities
    precision, recall, f1, auc = calculate_metrics(
        np.array(all_true_labels),
        np.array(all_pred_probabilities) > 0.5,  # Convert probabilities to binary predictions
        all_pred_probabilities  # Pass predicted probabilities for AUC calculation
    )

    return precision, recall, f1, auc


def run_iterations(num_iterations, P, Q, N, project_name, base_path):
    logging.info(
        f"The ratio of positive data: {(len(P) + len(Q)) / (len(P) + len(Q) + len(N)):.2f}")
    # Initialize lists to store metrics for each method
    pu_metrics = {'precision': [], 'recall': [], 'f1': [], 'auc': []}
    elkanoto_pu_metrics = {'precision': [], 'recall': [], 'f1': [], 'auc': []}
    rf_metrics = {'precision': [], 'recall': [], 'f1': [], 'auc': []}
    random_guess_metrics = {'precision': [], 'recall': [], 'f1': [], 'auc': []}
    constant_positive_model_metrics = {'precision': [], 'recall': [], 'f1': [], 'auc': []}

    feature_names = P.drop(columns=['label']).columns.tolist()
    feature_importance_list = []  # List to store feature importances for each iteration

    for i in range(num_iterations):
        logging.info(f"Iteration {i + 1}/{num_iterations}")

        # Run PU Learning (Weighted)
        precision, recall, f1, auc_score, feature_importances = weighted_elkanoto_pu_learning(P, Q, N, feature_names)
        pu_metrics['precision'].append(precision)
        pu_metrics['recall'].append(recall)
        pu_metrics['f1'].append(f1)
        pu_metrics['auc'].append(auc_score)
        feature_importance_list.append(feature_importances)  # Collect feature importance

        # Run PU Learning (Elkanoto)
        el_precision, el_recall, el_f1, el_auc_score = elkanoto_pu_learning(P, Q, N)
        elkanoto_pu_metrics['precision'].append(el_precision)
        elkanoto_pu_metrics['recall'].append(el_recall)
        elkanoto_pu_metrics['f1'].append(el_f1)
        elkanoto_pu_metrics['auc'].append(el_auc_score)

        # Run RandomForest Benchmark
        rf_precision, rf_recall, rf_f1, rf_auc_score = random_forest_benchmark(P, Q, N)
        rf_metrics['precision'].append(rf_precision)
        rf_metrics['recall'].append(rf_recall)
        rf_metrics['f1'].append(rf_f1)
        rf_metrics['auc'].append(rf_auc_score)

        # Run Random Guessing
        r_precision, r_recall, r_f1, r_auc_score = random_model(P, Q, N)
        random_guess_metrics['precision'].append(r_precision)
        random_guess_metrics['recall'].append(r_recall)
        random_guess_metrics['f1'].append(r_f1)
        random_guess_metrics['auc'].append(r_auc_score)

        # Run Constant Model
        c_precision, c_recall, c_f1, c_auc_score = constant_positive_model(P, Q, N)
        constant_positive_model_metrics['precision'].append(c_precision)
        constant_positive_model_metrics['recall'].append(c_recall)
        constant_positive_model_metrics['f1'].append(c_f1)
        constant_positive_model_metrics['auc'].append(c_auc_score)

    # Convert metrics into DataFrames for easier persistence
    pu_df = pd.DataFrame(pu_metrics)
    elkanoto_pu_df = pd.DataFrame(elkanoto_pu_metrics)
    rf_df = pd.DataFrame(rf_metrics)
    random_guess_df = pd.DataFrame(random_guess_metrics)
    constant_df = pd.DataFrame(constant_positive_model_metrics)

    # Add a column to identify the method
    pu_df['method'] = 'PU Learning (Weighted)'
    elkanoto_pu_df['method'] = 'PU Learning (Elkanoto)'
    rf_df['method'] = 'RandomForest'
    random_guess_df['method'] = 'Random Model'
    constant_df['method'] = 'Constant Positive'

    # Concatenate all DataFrames
    all_metrics_df = pd.concat([pu_df, elkanoto_pu_df, rf_df, random_guess_df, constant_df], ignore_index=True)

    # Define the output directory and file names
    evaluation_folder = os.path.join(base_path, 'evaluation')
    feature_importance_folder = os.path.join(base_path, 'feature_importance')

    os.makedirs(evaluation_folder, exist_ok=True)  # Create the folder if it doesn't exist
    os.makedirs(feature_importance_folder, exist_ok=True)

    metrics_output_path = os.path.join(evaluation_folder, f"{project_name}_10fold_validation_res.csv")
    stats_output_path = os.path.join(evaluation_folder, f"{project_name}_10fold_validation_res_stats.csv")
    feature_importance_output_path = os.path.join(feature_importance_folder,
                                                  f"{project_name}_model_feature_importance_res.csv")
    feature_importance_stats_path = os.path.join(feature_importance_folder,
                                                 f"{project_name}_model_feature_importance_stats.csv")

    # Save iteration results
    all_metrics_df.to_csv(metrics_output_path, index=False)

    # Compute mean, standard deviation, and median for each metric
    def compute_stats(metrics):
        return {
            'mean_precision': np.mean(metrics['precision']),
            'std_precision': np.std(metrics['precision']),
            'median_precision': np.median(metrics['precision']),
            'mean_recall': np.mean(metrics['recall']),
            'std_recall': np.std(metrics['recall']),
            'median_recall': np.median(metrics['recall']),
            'mean_f1': np.mean(metrics['f1']),
            'std_f1': np.std(metrics['f1']),
            'median_f1': np.median(metrics['f1']),
            'mean_auc': np.mean(metrics['auc']),
            'std_auc': np.std(metrics['auc']),
            'median_auc': np.median(metrics['auc']),
        }

    # Get summary statistics
    pu_stats = compute_stats(pu_metrics)
    elkanoto_pu_stats = compute_stats(elkanoto_pu_metrics)
    rf_stats = compute_stats(rf_metrics)
    random_guess_stats = compute_stats(random_guess_metrics)
    constant_positive_stats = compute_stats(constant_positive_model_metrics)

    # Combine all statistics into a DataFrame
    stats_df = pd.DataFrame(
        [pu_stats, elkanoto_pu_stats, rf_stats, random_guess_stats, constant_positive_stats],
        index=['PU Learning (Weighted)', 'PU Learning (Elkanoto)', 'RandomForest', 'Random Model', 'Constant Positive Model']
    )
    stats_df.reset_index(inplace=True)
    stats_df.rename(columns={'index': 'method'}, inplace=True)

    # Save statistics results
    stats_df.to_csv(stats_output_path, index=False)
    # Save feature importance results
    feature_importance_df = pd.DataFrame(feature_importance_list, columns=feature_names)
    feature_importance_df.to_csv(feature_importance_output_path, index=False)

    # Calculate and save feature importance statistics (mean and std)
    feature_importance_stats = {
        'feature': feature_names,
        'mean_importance': np.mean(feature_importance_list, axis=0),
        'std_importance': np.std(feature_importance_list, axis=0)
    }
    logging.info(f"PU Learning (Elkanoto) Feature Importance: {feature_importance_stats}")

    feature_importance_stats_df = pd.DataFrame(feature_importance_stats)
    feature_importance_stats_df.sort_values(by='mean_importance', ascending=False, inplace=True)
    feature_importance_stats_df.to_csv(feature_importance_stats_path, index=False)

    # Log the results
    logging.info(f"PU Learning (Elkanoto): {elkanoto_pu_stats}")
    logging.info(f"PU Learning (Weighted): {pu_stats}")
    logging.info(f"RandomForest Benchmark: {rf_stats}")
    logging.info(f"Random Model: {random_guess_stats}")
    logging.info(f"Constant Positive Model: {constant_positive_stats}")


def main():
    # Define paths using os.path
    current_file_path = os.getcwd()
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))), 'data',
                             'modeling_data', 'training')
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))), 'data')

    project_name_group = ['hive', 'hadoop', 'yarn', 'hdfs', 'hbase', 'ambari', 'hdds']
    # project_name_group = ['hive']
    for project_name in project_name_group:
        # test_project_name = project_name_group[0]
        # project_name = test_project_name

        # Read and preprocess data
        data_with_features = pd.read_csv(os.path.join(base_path, 'features2', f'{project_name}_data_with_features.csv'))
        data_with_features = preprocess_data(data_with_features, project_name)
        df_reduced, to_drop = select_features_by_spearman(data_with_features)
        # Create datasets
        P, Q, N = create_datasets(project_name=project_name, data_path=data_path, df_with_features=df_reduced)
        logging.info(
            f"Datasets P, Q, and N have been constructed. Dataset P: {len(P)}, Dataset Q: {len(Q)}, Dataset N: {len(N)}")

        # Make a copy of each dataset to avoid SettingWithCopyWarning
        P = P.copy()
        Q = Q.copy()
        N = N.copy()

        # Assign labels
        P['label'] = 1
        Q['label'] = 1
        N['label'] = -1

        # Drop columns
        columns_to_drop = [
            'project_name', 'comment_id', 'issue_id', 'comment_created_at_ts', 'combined_issue_id',
            'insertion', 'deletion', 'files', 'deletions', 'insertions'
        ]
        P = P.drop(columns=columns_to_drop, errors='ignore')
        Q = Q.drop(columns=columns_to_drop, errors='ignore')
        N = N.drop(columns=columns_to_drop, errors='ignore')

        # Run benchmarks for 1000 iterations
        run_iterations(100, P, Q, N, project_name, base_path)


if __name__ == "__main__":
    main()
