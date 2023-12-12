import pandas as pd
import matplotlib.pyplot as plt
import os


current_file_path = os.path.abspath(__file__)
data_path = "/".join(current_file_path.split("/")[:-3] + ["data/document_analysis_data/document_analysis.csv"])
# Convert data to a DataFrame
df = pd.read_csv(data_path)
# Group by Theme and Project and count the occurrences
df_grouped = df.groupby(['Theme', 'Project'])['Theme'].count().reset_index(name='count')

# Pivot the table
df_pivot = df_grouped.pivot(index='Theme', columns='Project', values='count').fillna(0)
# labels = [textwrap.fill(label, 10) for label in df_pivot.index]
labels = df_pivot.index.tolist()


def generate_abbreviation(theme):
    words = theme.split()
    if len(words) > 1:
        return ''.join(word[0].upper() for word in words)
    else:
        return theme[:2].upper()


abbreviation_dict = dict(zip(labels, [generate_abbreviation(label) for label in labels]))
# Plot the stacked bar chart
df_pivot = df_pivot.rename(columns=abbreviation_dict, index=abbreviation_dict)
plt.figure(figsize=(100, 10))  # Adjust the size as needed
df_pivot.plot(kind='bar', stacked=True)
# Rotate x-axis labels
plt.xticks(rotation=45)
explanation = "\n ".join(f"{full_name} - {abbreviation}" for abbreviation, full_name in abbreviation_dict.items())
explanation = " Theme Abbreviations\n " + explanation
explanation_box = plt.text(15, 120, explanation, fontsize=8, color='black', backgroundcolor='white', verticalalignment='top')
explanation_box.set_bbox(dict(facecolor='white', alpha=0.9, edgecolor='black'))

plt.title('Stacked Bar Chart of Theme Counts by Project')
# Rotate x-axis labels
plt.xlabel('Theme')
plt.ylabel('Count')
plt.tight_layout()  # Ensure everything fits properly
save_path = "/".join(current_file_path.split("/")[:-3])+'/data/results/document_analysis_sb_chart.pdf'
plt.savefig(save_path, format='pdf', bbox_inches='tight')
plt.show()