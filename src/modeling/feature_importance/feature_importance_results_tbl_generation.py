import os
import pandas as pd

def column_rename(df):
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

current_file_path = os.path.abspath(__file__)
data_directory = current_file_path.split("/")[:-4] + ['data', 'modeling_data', 'training','training_performance']
data_directory = '/'.join(data_directory)
directory_path = data_directory
# Initialize an empty DataFrame to store the results
results_df = pd.DataFrame(columns=["Project", "Feature", "Importance Mean", "Importance Std", "Importance Range"])
project_name_group = ['hive', 'hadoop', 'yarn', 'hdfs', 'hbase', 'ambari', 'hdds']

dfs = []
for project in project_name_group:
    csv_path = os.path.join(directory_path, f'{project}_model_importance.csv')
    # Read the CSV file
    df = pd.read_csv(csv_path)
    df = df[df['metric'] == 'recall']
    df = df.groupby('feature_name')[['importance_mean', 'importance_std']].mean().reset_index()
    df['rank'] = df['importance_mean'].rank(ascending=False)
    df = df.sort_values(by = ['rank'])
    df['importance_mean'] = df['importance_mean'].round(4)
    df['importance_std'] = df['importance_std'].round(4)
    df['project'] = project.upper()
    # Calculate the importance range as a string
    df["Importance Range"] = df.apply(lambda row: fr'''{row['importance_mean']} \textpm {row['importance_std']} (Top {int(row['rank'])})''' if int(row['rank'])>3 else fr'''{row['importance_mean']} \textpm {row['importance_std']} \highlight{{\textbf{{(Top {int(row['rank'])})}}}}''', axis=1)

    # Sort by importance_mean in descending order and select top 10
    top_features = df.sort_values(by="importance_mean", ascending=False).head(3)

    # Append the top_features DataFrame to the list of DataFrames
    dfs.append(top_features)

# Save the results to a new CSV file
# results_df.to_csv("top_10_features_per_project.csv", index=False)
results_df = pd.concat(dfs, ignore_index=True)
results_df = column_rename(results_df)
results_df['Rank Hint'] = results_df.groupby('project')['importance_mean'].rank(ascending=False, method='first').astype(int)
pivot_df = results_df.pivot(index='feature_name', columns='project', values='Importance Range')
pivot_df = pivot_df.fillna('-')
print(pivot_df.to_latex(index=True, escape=False))
