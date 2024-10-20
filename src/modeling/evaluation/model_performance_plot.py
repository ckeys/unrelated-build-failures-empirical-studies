import matplotlib.pyplot as plt
import pandas as pd
import os
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.interpolate import make_interp_spline


def preprocess_x_value(x_below_08):
    # Create a new list to store filtered values
    filtered_values = []

    # Initialize a variable to keep track of the previous value
    previous_value = None

    # Iterate through the x_below_08 list
    for value in x_below_08:
        # Check if there's a previous value to compare with
        if previous_value is not None:
            # Compare the difference between the current value and the previous value
            # with a threshold of 0.1
            if value - previous_value >= 0.1:
                filtered_values.append(value)
            else:
                # If the difference is less than 0.1, keep the smaller value
                filtered_values.append(min(value, previous_value))
        else:
            # For the first value, simply add it to the filtered list
            filtered_values.append(value)

        # Update the previous_value for the next iteration
        previous_value = value
    return filtered_values


def performance_plot(base_path, metric='f1'):
    # Initialize an empty DataFrame to store the data from all files
    all_data = pd.DataFrame()

    # Create a color map for projects
    projects_color_map = {}  # To store project names and their corresponding colors
    projects_minidx_map = {}
    color_palette = sns.color_palette("tab10")  # Using Seaborn's color palette

    # Iterate through the directory and find CSV files matching the pattern
    for root, _, files in os.walk(os.path.join(base_path)):
        for filename in files:
            project_name = filename.split('_')[0]
            if filename.startswith(f"{project_name}_model_performance") and filename.endswith(".csv"):
                file_path = os.path.join(root, filename)
                data = pd.read_csv(file_path)
                data['project'] = project_name
                all_data = pd.concat([all_data, data])
                if project_name not in projects_color_map:
                    color_index = len(projects_color_map) % len(color_palette)
                    projects_color_map[project_name] = color_palette[color_index]

    # Filter the data for the desired model
    pu_data = all_data[all_data['model'] == 'PULearning']

    # Set the style for the plot using Seaborn
    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 6))

    last_x_below_08 = 0
    for project_name, color in projects_color_map.items():
        project_data = pu_data[pu_data['project'] == project_name]
        project_mean = project_data.groupby('dropout')[metric].median()

        # Interpolate data points for smoother lines
        x_new = np.linspace(project_mean.index.min(), project_mean.index.max(), 300)
        spl = make_interp_spline(project_mean.index, project_mean.values, k=3)
        y_smooth = spl(x_new)

        plt.plot(x_new, y_smooth, label=project_name, color=color)

        # Find x-axis value where recall drops below 0.8
        x_below_08 = project_mean[project_mean < 0.8].index.min()
        # x_below_08 = preprocess_x_value(x_below_08)
        # Add vertical line at x_below_08 with x-value label
        # plt.axvline(x=x_below_08, color=color, linestyle='--')
        projects_minidx_map[project_name] = x_below_08
    # Set plot labels and title
    plt.xlabel('Num of Positive Samples Hidden')
    plt.ylabel(f'Mean {metric.upper() if metric != "f1" else metric.upper() + "-Score"}')
    plt.title(f'Mean {metric.upper() if metric != "f1" else metric.upper() + "-Score"} vs Num of Positive Samples Hidden')

    # Add legend with color explanations
    legend_labels = [f'{project_name.upper()} ({color_i + 1})' for project_name, color_i in
                     zip(projects_color_map.keys(), range(len(projects_color_map)))]
    plt.legend(legend_labels)
    # Add dashed line at y = 0.8
    plt.axhline(y=0.8, color=color, linestyle='--')
    for project_name in projects_minidx_map:
        color = projects_color_map[project_name]
        plt.axvline(x=projects_minidx_map[project_name], color=color, linestyle='--')
        x_below_08 = projects_minidx_map[project_name]
        if metric == 'recall' and x_below_08 != 42 and x_below_08 != 59 and x_below_08 != 67:
            plt.text(projects_minidx_map[project_name], 0, f'{int(projects_minidx_map[project_name])}', rotation=50,
                     va='bottom', ha='center', color=color)
        elif metric == 'f1' and (x_below_08 != 81 and x_below_08 != 84):
            plt.text(projects_minidx_map[project_name], 0, f'{int(projects_minidx_map[project_name])}', rotation=50,
                     va='bottom', ha='center', color=color)
    # Show the plot
    # plt.show()
    plt.savefig(f'{base_path}/figs/{metric}_performance.pdf', format='pdf', bbox_inches="tight")


current_file_path = os.path.abspath(__file__)
base_path = os.path.join('/'.join(current_file_path.split('/')[:-4]), 'data','modeling_data','training','training_with_LOOCV_performance' )
performance_plot(base_path, metric='f1')
