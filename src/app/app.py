import streamlit as st
import pandas as pd
import os
import plotly.express as px

# Get the base path (two levels up from the current file's location)
current_file_path = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.abspath(os.path.join(current_file_path, "../.."))

# Construct data paths based on base_path
DATA_PATH = os.path.join(base_path, 'data/modeling_data/training/evaluation')
FEATURE_PATH = os.path.join(base_path, 'data/modeling_data/training/feature_importance')


# Function to load the CSV data
def load_data(dataset):
    # Load 100 iterations results (res file)
    eval_file = os.path.join(DATA_PATH, f"{dataset}_10fold_validation_res.csv")
    eval_df = pd.read_csv(eval_file)

    # Load feature importance (optional based on your earlier description)
    feature_file = os.path.join(FEATURE_PATH, f"{dataset}_model_feature_importance_stats.csv")
    feature_df = pd.read_csv(feature_file)

    return eval_df, feature_df


# Main app function
def main():
    st.title("Evaluation Results and Feature Importance")

    # Sidebar for dataset selection
    st.sidebar.title("Select Dataset")
    dataset = st.sidebar.selectbox(
        "Choose a dataset",
        ("ambari", "hadoop", "hbase", "hdds", "hdfs", "hive", "yarn")
    )

    # Load data based on user selection
    eval_df, feature_df = load_data(dataset)

    st.header(f"Evaluation Results for {dataset.upper()} (100 Iterations)")

    # Display evaluation data
    st.subheader("Evaluation Metrics (Boxplots)")
    metrics = ['precision', 'recall', 'f1', 'auc']
    selected_metric = st.selectbox("Select Metric for Boxplot", metrics)

    # Plot boxplot for the selected metric
    st.subheader(f"Boxplot of {selected_metric} by Method")
    fig = px.box(eval_df, x="method", y=selected_metric, points="all",
                 title=f"{selected_metric.capitalize()} Distribution by Method")
    st.plotly_chart(fig)

    st.subheader("Feature Importance Visualization")

    # Sort the feature importance by 'mean_importance' in descending order
    sorted_feature_df = feature_df.sort_values(by='mean_importance', ascending=False)

    # Create a bar chart using Plotly to ensure the bars are sorted by mean_importance
    fig = px.bar(sorted_feature_df, x='feature', y='mean_importance',
                 title='Feature Importance (Ordered by Mean Importance)',
                 labels={'mean_importance': 'Mean Importance', 'feature': 'Feature'})

    # Display the chart
    st.plotly_chart(fig)
# Run the app
if __name__ == "__main__":
    main()