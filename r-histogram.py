import pandas as pd
import matplotlib.pyplot as plt

# Assuming PATH and CSV_DATA are defined elsewhere in your 'modules.path' module
from modules.path import PATH, CSV_DATA

# Reading the dataset
dataset = pd.read_csv(PATH + CSV_DATA)

# Select only the numerical features
numerical_features = dataset.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Plotting histograms for each numerical feature
for feature in numerical_features:
    plt.figure(figsize=(10, 4))
    plt.hist(dataset[feature], bins=30, edgecolor='k', alpha=0.7)
    plt.title(f'Histogram of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
