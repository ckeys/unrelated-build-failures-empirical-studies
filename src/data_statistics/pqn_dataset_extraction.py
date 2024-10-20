import pandas as pd


def create_datasets(project_name, data_path, df_with_features=None):
    """
    Create P, Q, and N datasets based on the given project name and data path.

    Parameters:
    project_name (str): The name of the project.
    data_path (str): The path to the data directory.

    Returns:
    tuple: A tuple containing the P, Q, and N datasets as DataFrames.
    """
    # Read data from CSV files
    if df_with_features is None:
        df_with_features = pd.read_csv(
            f"{data_path}/modeling_data/training/features/{project_name.lower()}_data_with_features.csv")
    positive_df = pd.read_csv(
        f"{data_path}/modeling_data/training/labeled_data/{project_name.lower()}_labeled_data.csv")
    sample_df = pd.read_csv(
        f"{data_path}/data_sampling/{project_name.upper()}_sample_overall_data.csv")

    # Create the issue_id and comment_id tuples for positive and sample datasets
    positive_tuples = set(zip(positive_df["issue_id"], positive_df["comment_id"]))
    sample_df['combined_issue_id'] = sample_df['project_name'] + "-" + sample_df["issue_id"].astype(str)
    sample_tuples = set(zip(sample_df['combined_issue_id'], sample_df["comment_id"]))

    # Add combined_issue_id to df_with_features for matching
    df_with_features['combined_issue_id'] = df_with_features['project_name'] + "-" + df_with_features[
        "issue_id"].astype(str)

    # Construct P dataset
    P = df_with_features[df_with_features.apply(
        lambda row: (row["combined_issue_id"], row["comment_id"]) in positive_tuples, axis=1)]

    # Construct Q dataset
    Q = df_with_features[df_with_features.apply(
        lambda row: (row["combined_issue_id"], row["comment_id"]) in sample_tuples and sample_df.loc[
            (sample_df["combined_issue_id"] == row["combined_issue_id"]) &
            (sample_df["comment_id"] == row["comment_id"]), "manual_label"].values[0] == 1, axis=1)]

    # Construct N dataset
    N = df_with_features[df_with_features.apply(
        lambda row: (row["combined_issue_id"], row["comment_id"]) in sample_tuples and sample_df.loc[
            (sample_df["combined_issue_id"] == row["combined_issue_id"]) &
            (sample_df["comment_id"] == row["comment_id"]), "manual_label"].values[0] in [-1, 0], axis=1)]

    return P, Q, N


# Example usage:
if __name__ == "__main__":
    project_name = "hive"
    data_path = '/Users/andie/PycharmProjects/unrelated-build-failures-empirical-studies/data'
    P, Q, N = create_datasets(project_name, data_path)
    print("Datasets P, Q, and N have been constructed.")
