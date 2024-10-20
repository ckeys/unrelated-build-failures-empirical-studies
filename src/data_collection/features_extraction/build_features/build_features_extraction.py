import pandas as pd
import os
import re


def clean_html_tags(text):
    """Remove HTML tags from the given text."""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def extract_features(comment, project_name, issue_id, comment_id):
    # Remove HTML tags from the comment
    cleaned_comment = clean_html_tags(comment)

    features = {
        'project_name': project_name,
        'issue_id': issue_id,
        'comment_id': comment_id
    }

    # Extract the number of failed tests and their names
    match_failed_tests = re.search(r'-1 due to (\d+) failed/errored test\(s\)', cleaned_comment, re.IGNORECASE)
    if match_failed_tests:
        features['num_of_tests_failed'] = int(match_failed_tests.group(1))
    else:
        features['num_of_tests_failed'] = -1

    return features


# Get the current project path
project_path = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-4])
project_name = "HIVE"
# Define the subdirectory path
subdirectory_path = os.path.join(project_path, 'data', 'modeling_data', 'training', 'features')
# Define the complete file path
file_path = os.path.join(subdirectory_path, f'{project_name}_data_with_features.csv')
# Read the CSV file
feature_df = pd.read_csv(file_path)
# feature_df['combined_key'] = feature_df['project_name'] + '_' + feature_df['issue_id'].astype(str) + '_' + feature_df['comment_id'].astype(str)
print(feature_df.shape)
# Define the complete file path for df2
raw_comment_df_path = f'/Users/andie/Library/CloudStorage/OneDrive-UniversityofOtago/Otago/Project1/jira_data/final_res/{project_name}_step1.csv'
# Read the second CSV file
raw_comment_df = pd.read_csv(raw_comment_df_path)
comment_df_filtered = raw_comment_df[raw_comment_df['comment_id'].isin(feature_df['comment_id'])]
print(comment_df_filtered.shape)

feature_rows = comment_df_filtered.apply(
    lambda row: extract_features(row['comment_content'], row['project_name'], row['issue_id'], row['comment_id']),
    axis=1).tolist()
build_features_df = pd.DataFrame(feature_rows)
print(build_features_df)
# Perform the left join with an indicator
sample_data_path = f'/Users/andie/PycharmProjects/unrelated-build-failures-empirical-studies/data/data_sampling/HIVE_sample_overall_data.csv'
sample_df = pd.read_csv(sample_data_path)

merged_df = feature_df.merge(build_features_df, on=['project_name', 'issue_id', 'comment_id'], how='left',
                            indicator=False)
# merged_df['label2'] = merged_df['label2'].replace(-1, 0)
merged_df.to_csv(f'/Users/andie/PycharmProjects/unrelated-build-failures-empirical-studies/data/modeling_data/training/features2/{project_name}_data_with_features.csv', index=False)
# print(merged_df)
# #
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.tree import DecisionTreeClassifier
# # Scatter plot
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='num_of_tests_failed', y='label', data=merged_df)
# plt.title('Scatter Plot of Number of Tests Failed vs Label2')
# plt.xlabel('Number of Tests Failed')
# plt.ylabel('Label2')
# plt.show()
# # Equal-frequency binning
# quantiles = 3  # Number of bins
# merged_df['binned'] = pd.qcut(merged_df['num_of_tests_failed'], q=quantiles)
# print(merged_df)
# # Visualize the binned data
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='label2', y='binned', data=merged_df)
# plt.title('Box Plot of Label2 by Binned Number of Tests Failed')
# plt.xlabel('Binned Number of Tests Failed')
# plt.ylabel('Label2')
# plt.show()