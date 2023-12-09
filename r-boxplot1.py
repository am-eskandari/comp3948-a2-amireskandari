import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming PATH and CSV_DATA are defined elsewhere in your 'modules.path' module
from modules.path import PATH, CSV_DATA

# Reading the dataset
dataset = pd.read_csv(PATH + CSV_DATA)


# Select only the numerical features from the dataset, excluding 'Churn' if it's numerical
numerical_features = dataset.select_dtypes(include=['float64', 'int64']).columns.tolist()
if 'Churn' in numerical_features:
    numerical_features.remove('Churn')

# Melt the dataset to 'long-form' or 'tidy' representation
dataset_melted = dataset.melt(id_vars='Churn', value_vars=numerical_features)

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Initialize the figure
plt.figure(figsize=(18, 8))

# Create the boxplot with 'hue' for color encoding
boxplot = sns.boxplot(data=dataset_melted, x='variable', y='value', hue='Churn', palette='Set2')

# Improve the aesthetics and the legibility
plt.title('Comparison of Numerical Features by Churn', fontsize=25)
plt.xlabel('Numerical Features', fontsize=20)
plt.ylabel('Values', fontsize=20)

# Set the x-axis labels with the names of the numerical features, rotated 90 degrees for vertical display

boxplot.set_xticklabels(boxplot.get_xticklabels(), rotation=45, fontsize=15)


plt.legend(title='Churn')

# Show the plot
plt.tight_layout()
plt.show()



# Creating a boxplot for TotalCharges
plt.figure(figsize=(10, 6))
sns.boxplot(data=dataset, y='TotalCharges')
plt.title('Boxplot of Total Charges')
plt.ylabel('Total Charges')
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming PATH and CSV_DATA are defined elsewhere in your 'modules.path' module
from modules.path import PATH, CSV_DATA

# Reading the dataset
dataset = pd.read_csv(PATH + CSV_DATA)

# Select only the numerical features from the dataset, excluding 'Churn' if it's numerical
numerical_features = dataset.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Remove 'Churn' if it's in numerical_features
if 'Churn' in numerical_features:
    numerical_features.remove('Churn')

# Additionally, remove 'TotalCharges' from the list
if 'TotalCharges' in numerical_features:
    numerical_features.remove('TotalCharges')

# Melt the dataset to 'long-form' or 'tidy' representation
dataset_melted = dataset.melt(id_vars='Churn', value_vars=numerical_features)

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Initialize the figure
plt.figure(figsize=(18, 8))

# Create the boxplot with 'hue' for color encoding
boxplot = sns.boxplot(data=dataset_melted, x='variable', y='value', hue='Churn', palette='Set2')

# Improve the aesthetics and the legibility
plt.title('Comparison of Numerical Features by Churn', fontsize=25)
plt.xlabel('Numerical Features', fontsize=20)
plt.ylabel('Values', fontsize=20)

# Set the x-axis labels with the names of the numerical features, rotated 90 degrees for vertical display
boxplot.set_xticklabels(boxplot.get_xticklabels(), rotation=45, fontsize=15)

plt.legend(title='Churn')

# Show the plot
plt.tight_layout()
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming PATH and CSV_DATA are defined elsewhere in your 'modules.path' module
from modules.path import PATH, CSV_DATA

# Reading the dataset
dataset = pd.read_csv(PATH + CSV_DATA)

# Check if 'Churn' and 'TotalCharges' exist in the dataset
if 'Churn' in dataset.columns and 'TotalCharges' in dataset.columns:
    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")

    # Initialize the figure
    plt.figure(figsize=(10, 6))

    # Create the boxplot
    sns.boxplot(data=dataset, x='Churn', y='TotalCharges', palette='Set2')

    # Improve the aesthetics and the legibility
    plt.title('Boxplot of Total Charges by Churn', fontsize=20)
    plt.xlabel('Churn', fontsize=15)
    plt.ylabel('Total Charges', fontsize=15)

    # Show the plot
    plt.tight_layout()
    plt.show()
else:
    print("The dataset does not contain 'Churn' or 'TotalCharges' columns.")
