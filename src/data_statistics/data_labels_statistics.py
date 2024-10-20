import pandas as pd
import math
import os
from scipy.stats import norm


class ProjectAnalyzer:
    def __init__(self, projects, base_path, z_score, margin_of_error):
        self.projects = projects
        self.base_path = base_path
        self.z_score = z_score
        self.margin_of_error = margin_of_error
        self.results = []

    def calculate_sample_size(self, proportion):
        return (self.z_score ** 2 * proportion * (1 - proportion)) / self.margin_of_error ** 2

    def analyze_project(self, project_name):
        # Construct the file path
        file_path = f"{self.base_path}{project_name}_data_with_features.csv"

        try:
            # Read the CSV file
            data = pd.read_csv(file_path)

            # Calculate the number of rows of data
            num_rows = len(data)

            # Calculate the number of rows where 'label' == 1
            num_label_1 = len(data[data['label'] == 1])

            # Calculate the ratio of positive samples
            ratio_positive = num_label_1 / num_rows if num_rows > 0 else 0

            # Calculate the required sample sizes
            sample_size_total = math.ceil(self.calculate_sample_size(0.5))  # Using 0.5 for maximum variability
            sample_size_positive = math.ceil(self.calculate_sample_size(ratio_positive))

            # Append the results to the list
            self.results.append(
                [project_name.upper(), num_rows, num_label_1, ratio_positive, sample_size_total, sample_size_positive])
        except FileNotFoundError:
            print(f"File not found for project: {project_name}")
        except Exception as e:
            print(f"An error occurred for project: {project_name}: {e}")

    def analyze_projects(self):
        for project_name in self.projects:
            self.analyze_project(project_name)

    def get_results(self):
        # Create a DataFrame from the results
        results_df = pd.DataFrame(self.results, columns=['Project Name', 'Num of Samples', 'Num of Positive',
                                                         'Ratio of Positive', 'Required Sample Size (Total)',
                                                         'Required Sample Size (Positive)'])
        return results_df


class ProjectDataSampling:
    def __init__(self, results_df, base_path, output_path):
        self.results_df = results_df
        self.base_path = base_path
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

    def sample_overall_data(self, project_name, sample_size):
        # Construct the file path
        file_path = f"{self.base_path}{project_name.lower()}_data_with_features.csv"
        try:
            # Read the CSV file
            data = pd.read_csv(file_path)
            data = data[data['project_name'].str.upper() == project_name.upper()]
            data = data[['project_name', 'comment_id', 'issue_id', 'label']]
            data['URL'] = "https://issues.apache.org/jira/browse/" + data['project_name'] + "-" + \
                          data['issue_id'].astype(str)
            # Perform random sampling
            sampled_data = data.sample(n=sample_size, random_state=1)  # random_state for reproducibility

            # Save the sampled data to a CSV file
            output_file_path = f"{self.output_path}{project_name}_sample_overall_data.csv"
            sampled_data.to_csv(output_file_path, index=False)

            return sampled_data
        except FileNotFoundError:
            print(f"File not found for project: {project_name}")
            return pd.DataFrame()  # Return empty DataFrame on error
        except Exception as e:
            print(f"An error occurred for project: {project_name}: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error

    def sample_heuristic_positive_data(self, project_name, sample_size_positive):
        # Construct the file path
        file_path = f"{self.base_path}{project_name.lower()}_data_with_features.csv"

        try:
            # Read the CSV file
            data = pd.read_csv(file_path)

            # Filter data where 'label' == 1
            positive_data = data[data['label'] == 1]
            positive_data = positive_data[['project_name', 'comment_id', 'issue_id', 'label']]
            positive_data['URL'] = "https://issues.apache.org/jira/browse/" + positive_data['project_name'] + "-" + \
                                   positive_data['issue_id'].astype(str)
            # Perform random sampling
            sampled_data = positive_data.sample(n=sample_size_positive,
                                                random_state=1)  # random_state for reproducibility

            # Save the sampled data to a CSV file
            output_file_path = f"{self.output_path}{project_name}_sample_positive_data.csv"
            sampled_data.to_csv(output_file_path, index=False)

            return sampled_data
        except FileNotFoundError:
            print(f"File not found for project: {project_name}")
            return pd.DataFrame()  # Return empty DataFrame on error
        except Exception as e:
            print(f"An error occurred for project: {project_name}: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error

    def sample_all_projects(self, positive_or_overall="positive"):
        sampled_results = {}
        for _, row in self.results_df.iterrows():
            project_name = row['Project Name']
            if positive_or_overall == "positive":
                sample_size_positive = row['Required Sample Size (Positive)']
                self.sample_heuristic_positive_data(project_name, sample_size_positive)
            elif positive_or_overall == "overall":
                sample_size_overall = row['Required Sample Size (Total)']
                self.sample_overall_data(project_name, sample_size_overall)


# List of projects
projects = ['hive', 'ambari', 'hadoop', 'hbase', 'hdds', 'hdfs', 'yarn']
projects = ['ambari', 'hadoop', 'hbase', 'hdds', 'hdfs', 'yarn']
# projects = ['hadoop']
# Base file path
base_path = "../../data/modeling_data/training/features/"

# Z-score for 95% confidence
z_score = norm.ppf(0.975)
# Margin of error
margin_of_error = 0.05

# Create an instance of ProjectAnalyzer
analyzer = ProjectAnalyzer(projects, base_path, z_score, margin_of_error)

# Analyze the projects
analyzer.analyze_projects()

# Get the results
results_df = analyzer.get_results()
# Set pandas display options to avoid truncation
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.max_rows', None)  # Display all rows
pd.set_option('display.max_colwidth', None)  # Display full column content
pd.set_option('display.width', 1000)  # Adjust the width to accommodate the table

# Print the results table
print(results_df)

output_path = "../../data/data_sampling/"
sampler = ProjectDataSampling(results_df, base_path, output_path)
sampled_results = sampler.sample_all_projects(positive_or_overall="overall")
