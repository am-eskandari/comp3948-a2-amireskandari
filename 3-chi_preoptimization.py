import numpy as np
import pandas as pd
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

# Separate into x and y values.
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
]
X = dataset[predictorVariables]
y = dataset['Churn']

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=0)

# LOGISTIC REGRESSION ==================================================================================================
# Logistic Regression Model
logistic_model = LogisticRegression(solver='liblinear', random_state=0)

# CROSS-VALIDATION WITH SMOTE =========================================================================================
# Define cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# Lists to store metrics for each fold
accuracies = []
precisions = []
recalls = []
f1_scores = []
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

# STATISTICS ACROSS ALL FOLDS ==========================================================================================
accuracies, precisions, recalls, f1_scores = map(np.array, [accuracies, precisions, recalls, f1_scores])

print(f'Average accuracy across all folds: {accuracies.mean():.4f}')
print(f'Standard deviation of accuracy across all folds: {accuracies.std():.4f}')
print(f'Average precision across all folds: {precisions.mean():.4f}')
print(f'Standard deviation of precision across all folds: {precisions.std():.4f}')
print(f'Average recall across all folds: {recalls.mean():.4f}')
print(f'Standard deviation of recall across all folds: {recalls.std():.4f}')
print(f'Average F1 score across all folds: {f1_scores.mean():.4f}')
print(f'Standard deviation of F1 score across all folds: {f1_scores.std():.4f}')


# FINAL MODEL EVALUATION ===============================================================================================
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
