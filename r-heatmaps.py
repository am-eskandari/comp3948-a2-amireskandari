import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from modules.path import PATH, CSV_DATA

# Reading the dataset (the dataset must have a header row, so no need to adjust the 'names' parameter)
dataset = pd.read_csv(PATH + CSV_DATA)

# Show header and first 5 rows
pd.set_option('display.max_columns', None)
# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)
print(dataset.head())
print(dataset.describe(include='all'))

# Select only numerical variables
numerical_data = dataset.select_dtypes(include=['float64', 'int64'])

# Compute the correlation matrix
corr = numerical_data.corr()

# Sort the 'Churn' column correlations in descending order while excluding the 'Churn' correlation with itself
sorted_corr = corr[['Churn']].sort_values('Churn', ascending=False)

# Plot the heatmap with the sorted correlations
plt.figure(figsize=(8, 8))  # Adjust the figure size as needed
sns.heatmap(sorted_corr, annot=True, fmt=".2f", linewidths=0.5, vmin=-1, vmax=1, cmap="YlGnBu")
plt.title('Correlation with Churn')  # You can set an appropriate title
plt.tight_layout()
plt.show()


# Select only numerical variables
numerical_columns = dataset.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Determine the layout of the subplots
n_cols = 3  # Number of columns in the subplot grid
n_rows = (len(numerical_columns) + n_cols - 1) // n_cols  # Compute the necessary number of rows

plt.figure(figsize=(5 * n_cols, 5 * n_rows))  # Adjust the overall figure size

# Create a scatter plot for each numerical variable
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(n_rows, n_cols, i)  # Create a subplot for each variable
    sns.scatterplot(data=dataset, x=column, y='Churn', hue='Churn', palette='bright', s=1)
    plt.xlabel(column)
    plt.ylabel('Churn')
    plt.title(f'{column} vs Churn')

plt.tight_layout()
plt.show()
