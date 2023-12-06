import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from modules.best_k import best_k
from modules.chi_squares import chi_squares
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from modules.path import PATH, CSV_DATA

# DATA PROCESSING ======================================================================================================
# Reading the dataset
dataset = pd.read_csv(PATH + CSV_DATA)

# Show header and first 5 rows
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(dataset.describe())

# Creating boxplots to detect outliers
plt.figure(figsize=(15, 8))
continuous_vars = ['AccountAge', 'MonthlyCharges',
                   'TotalCharges',
                   'ViewingHoursPerWeek', 'AverageViewingDuration',
                   'ContentDownloadsPerMonth', 'UserRating', 'SupportTicketsPerMonth', 'WatchlistSize']
# Set up the matplotlib figure and axes
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))  # Adjust the grid size based on the number of variables
axes = axes.flatten()  # Flatten the axes array for easy iteration

# Creating individual boxplots for each continuous variable
for i, col in enumerate(continuous_vars):
    axes[i].boxplot(dataset[col].dropna())  # Drop NA values to avoid errors
    axes[i].set_title(col)
    axes[i].set_xticks([1])  # Set the x-ticks to show the variable name
    axes[i].set_xticklabels([col], rotation=45)

# Hide any unused axes if there are fewer variables than subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()  # Adjust the layout so labels don't overlap
plt.show()


# Creating dummy variables
categorical_vars = ['SubscriptionType', 'PaymentMethod', 'PaperlessBilling', 'ContentType',
                    'MultiDeviceAccess', 'DeviceRegistered', 'GenrePreference', 'Gender',
                    'ParentalControl', 'SubtitlesEnabled']

# For each categorical column, fill missing values with 'not exist'
for column in categorical_vars:
    if column in dataset.columns:
        dataset[column].fillna('not exist', inplace=True)
dataset = pd.get_dummies(dataset, columns=categorical_vars, drop_first=True)

# Imputing missing values
imputer = SimpleImputer(strategy='median')
dataset[['AccountAge', 'ViewingHoursPerWeek', 'AverageViewingDuration']] = imputer.fit_transform(
    dataset[['AccountAge', 'ViewingHoursPerWeek', 'AverageViewingDuration']])


# Clip 'TotalCharges' to the upper boundary.
dfAdjusted = dataset['TotalCharges'].clip(upper=2250)
dataset['TotalCharges'] = dfAdjusted

plt.figure(figsize=(8, 6))  # Set the figure size
plt.boxplot(dataset['TotalCharges'].dropna())  # Create the boxplot, dropping NA values
plt.title('Boxplot of Total Charges')  # Set the title
plt.ylabel('Total Charges')  # Set the y-axis label
plt.xticks([1], ['Total Charges'])  # Set the x-axis label
plt.show()


# Binning function
def bin_variable(data, variable, bins, labels):
    binned_col_name = f'{variable}_Bin'
    data[binned_col_name] = pd.cut(data[variable], bins=bins, labels=labels, include_lowest=True)
    print(f"Binned {variable}, new column: {binned_col_name}")  # Debugging print
    return binned_col_name


# Binning variables and adding them to the dataset
binned_columns = []
for variable, bins, labels in [
    ('AccountAge', [0, 40, 80, 120], ['Young', 'Middle-aged', 'Senior']),
    ('TotalCharges', [0, 1000, 2000, 3000], ['Low', 'Medium', 'High']),
    ('ViewingHoursPerWeek', [0, 15, 30, 45], ['Low', 'Medium', 'High']),
]:
    binned_col_name = bin_variable(dataset, variable, bins, labels)
    binned_columns.append(binned_col_name)

# Convert binned columns to dummy variables
dataset = pd.get_dummies(dataset, columns=binned_columns, drop_first=True)

# Add the new dummy variable names (generated from binned columns) to the predictorVariables list
new_binned_dummy_cols = [col for col in dataset.columns if
                         col.endswith("_High") or col.endswith("_Medium") or col.endswith("_Long") or col.endswith(
                             "_Many") or col.endswith("_Senior") or col.endswith("_Middle-aged")]

# Predictor Variables
predictorVariables = [
                         'AccountAge', 'MonthlyCharges', 'TotalCharges', 'ViewingHoursPerWeek',
                         'AverageViewingDuration', 'ContentDownloadsPerMonth', 'UserRating',
                         'SupportTicketsPerMonth', 'WatchlistSize', 'SubscriptionType_Premium',
                         'SubscriptionType_Standard', 'PaymentMethod_Credit card',
                         'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
                         'PaperlessBilling_Yes', 'ContentType_Movies', 'ContentType_TV Shows',
                         'MultiDeviceAccess_Yes', 'DeviceRegistered_Mobile', 'DeviceRegistered_TV',
                         'DeviceRegistered_Tablet', 'GenrePreference_Comedy', 'GenrePreference_Drama',
                         'GenrePreference_Fantasy', 'GenrePreference_Sci-Fi', 'Gender_Male',
                         'ParentalControl_Yes', 'SubtitlesEnabled_Yes'
                     ] + new_binned_dummy_cols

X = dataset[predictorVariables]
y = dataset['Churn']

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Feature Selection (Chi-Square, Best k) Here
# k = best_k(X_scaled, y)
# print(f"Best k value: {k}")
k = 25
selected_features = chi_squares(X, X_scaled, y, k)
print(f"Selected features: {selected_features}")

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=0)

# LOGISTIC REGRESSION ================================================================================================
logistic_model = LogisticRegression(solver='liblinear', random_state=0)

# CROSS-VALIDATION WITH SMOTE ========================================================================================
# Define cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
accuracies, precisions, recalls, f1_scores = [], [], [], []
smote = SMOTE()

# Cross-validation loop
for train_index, test_index in kf.split(X_scaled, y):
    # Splitting data into training and test sets for the current fold
    X_train_fold, X_test_fold = X_scaled[train_index], X_scaled[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]

    # Apply SMOTE to the training data
    X_train_fold_resampled, y_train_fold_resampled = smote.fit_resample(X_train_fold, y_train_fold)

    # Train the model
    logistic_model.fit(X_train_fold_resampled, y_train_fold_resampled)

    # Make predictions on the test set
    y_pred_fold = logistic_model.predict(X_test_fold)

    # Calculate metrics
    accuracies.append(accuracy_score(y_test_fold, y_pred_fold))
    precisions.append(precision_score(y_test_fold, y_pred_fold))
    recalls.append(recall_score(y_test_fold, y_pred_fold))
    f1_scores.append(f1_score(y_test_fold, y_pred_fold))

# STATISTICS ACROSS ALL FOLDS ========================================================================================
accuracies, precisions, recalls, f1_scores = map(np.array, [accuracies, precisions, recalls, f1_scores])

print(f'Average accuracy across all folds: {accuracies.mean():.4f}')
print(f'Standard deviation of accuracy across all folds: {accuracies.std():.4f}')
print(f'Average precision across all folds: {precisions.mean():.4f}')
print(f'Standard deviation of precision across all folds: {precisions.std():.4f}')
print(f'Average recall across all folds: {recalls.mean():.4f}')
print(f'Standard deviation of recall across all folds: {recalls.std():.4f}')
print(f'Average F1 score across all folds: {f1_scores.mean():.4f}')
print(f'Standard deviation of F1 score across all folds: {f1_scores.std():.4f}')

# FINAL MODEL EVALUATION =============================================================================================
# Train final model on the entire training set with SMOTE
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
logistic_model.fit(X_train_resampled, y_train_resampled)

# Evaluate on the test set
y_pred = logistic_model.predict(X_test)

# Output the evaluation results
churn = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
print('\nConfusion Matrix')
print(churn)
print(f'\nAccuracy: {accuracy_score(y_test, y_pred):.4f}')
print(f'Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}')
print(f'Recall: {recall_score(y_test, y_pred, zero_division=0):.4f}')
print(f'F1 Score: {f1_score(y_test, y_pred, zero_division=0):.4f}')
