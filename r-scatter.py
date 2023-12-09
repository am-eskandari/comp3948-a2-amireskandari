import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from modules.path import PATH, CSV_DATA

# Reading the dataset
dataset = pd.read_csv(PATH + CSV_DATA)

# Define the numerical features by excluding non-numerical columns and the 'Churn' column
numerical_features = dataset.select_dtypes(include=['float64', 'int64']).columns.tolist()
if 'Churn' in numerical_features:
    numerical_features.remove('Churn')  # Remove 'Churn' from the list if it's there

# Create a grid of subplots
n_cols = 3  # Number of columns in the subplot grid
n_rows = (len(numerical_features) + n_cols - 1) // n_cols  # Compute the necessary number of rows

# Initialize the subplot figure with a certain size
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))  # Width, height in inches
fig.suptitle('Numerical Features vs Churn with Skewness', fontsize=16)

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Create a scatter plot for each numerical feature
for i, feature in enumerate(numerical_features):
    # Calculate the skewness of the feature
    skewness = dataset[feature].skew()

    # Create the scatter plot
    sns.scatterplot(data=dataset, x=feature, y='Churn', ax=axes[i], alpha=0.5, s=10)

    # Set the title and labels
    axes[i].set_title(f'{feature}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Churn')

    # Add the skewness value as text to the subplot
    axes[i].text(0.95, 0.02, f'Skewness: {skewness:.2f}',
                 verticalalignment='bottom', horizontalalignment='right',
                 transform=axes[i].transAxes, color='red', fontsize=15)

# Hide any empty subplots
for ax in axes[len(numerical_features):]:
    ax.axis('off')

# Adjust layout for a clean look
plt.tight_layout()
plt.subplots_adjust(top=0.95)  # Adjust the top spacing to fit the main title
plt.show()
