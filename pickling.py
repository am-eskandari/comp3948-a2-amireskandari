import pickle
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

# Reading the dataset
PATH = "/Users/test/Documents/Programming Directories/BCIT/COMP 3948/comp3948-a2-amireskandari/datasets/"
CSV_DATA = "CustomerChurn.csv"
dataset = pd.read_csv(PATH + CSV_DATA, encoding="ISO-8859-1", sep=',')

# Apply median imputation for missing values in numeric columns
dataset.fillna(dataset.median(numeric_only=True), inplace=True)

# Creating dummy variables
categorical_vars = ['SubscriptionType', 'PaymentMethod', 'PaperlessBilling', 'ContentType',
                    'MultiDeviceAccess', 'DeviceRegistered', 'GenrePreference', 'Gender',
                    'ParentalControl', 'SubtitlesEnabled']
dataset[categorical_vars].fillna('not exist', inplace=True)

dummyDf = pd.get_dummies(dataset[categorical_vars])
dataset = pd.concat([dataset, dummyDf], axis=1)

# Clip 'TotalCharges' to the upper boundary.
dfAdjusted = dataset['TotalCharges'].clip(upper=2250)
dataset['TotalCharges'] = dfAdjusted

# Binning
# dataset['AccountAgeBin'] = pd.cut(x=dataset['AccountAge'], bins=[0, 800, 1600, 2400])
dataset['TotalChargesBin'] = pd.cut(x=dataset['TotalCharges'], bins=[0, 1000, 2000, 3000])
dataset['ViewingHoursPerWeekBin'] = pd.cut(x=dataset['WatchlistSize'], bins=[0, 15, 30, 45])
additional_dummies = pd.get_dummies(dataset[[
    # 'AccountAgeBin',
    'TotalChargesBin',
    'ViewingHoursPerWeekBin'
]])
dataset = pd.concat([dataset, additional_dummies], axis=1)

predictorVariables = [
    'AccountAge',
    'MonthlyCharges',
    'TotalCharges',
    'ViewingHoursPerWeek',
    'AverageViewingDuration',
    'ContentDownloadsPerMonth',
    'UserRating',
    'SupportTicketsPerMonth',
    'WatchlistSize',
    'SubscriptionType_Premium',
    'SubscriptionType_Standard',
    'PaymentMethod_Credit card',
    'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check',
    'PaperlessBilling_Yes',
    'ContentType_Movies',
    'ContentType_TV Shows',
    'MultiDeviceAccess_Yes',
    'DeviceRegistered_Mobile',
    'DeviceRegistered_TV',
    'DeviceRegistered_Tablet',
    'GenrePreference_Comedy',
    'GenrePreference_Drama',
    'GenrePreference_Fantasy',
    'GenrePreference_Sci-Fi',
    'Gender_Male',
    'Gender_Female',
    'ParentalControl_Yes',
    'SubtitlesEnabled_Yes',
    # 'AccountAgeBin_(0, 800]',
    # 'AccountAgeBin_(800, 1600]',
    # 'AccountAgeBin_(1600, 2400]',
    'TotalChargesBin_(0, 1000]',
    'TotalChargesBin_(1000, 2000]',
    'TotalChargesBin_(2000, 3000]',
    'ViewingHoursPerWeekBin_(0, 15]',
    'ViewingHoursPerWeekBin_(15, 30]',
    'ViewingHoursPerWeekBin_(30, 45]'
]

X = dataset[predictorVariables]
y = dataset['Churn']

# Selected features
selected_features = [
    'AccountAge',
    'MonthlyCharges',
    'TotalCharges',
    'ViewingHoursPerWeek',
    'AverageViewingDuration',
    'ContentDownloadsPerMonth',
    'SupportTicketsPerMonth',
    'TotalChargesBin_(0, 1000]',
    'TotalChargesBin_(1000, 2000]'
]

X = X[selected_features]

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=0)

logistic_model = LogisticRegression(solver='liblinear', random_state=0)

# Cross-validation with SMOTE
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
accuracies, precisions, recalls, f1_scores = [], [], [], []
oversample = SMOTE()

# Cross-validation loop
for train_index, test_index in kf.split(X_scaled, y):
    # Splitting data into training and test sets for the current fold
    X_train_fold, X_test_fold = X_scaled[train_index], X_scaled[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]

    # Apply SMOTE to the training data
    X_train_fold_resampled, y_train_fold_resampled = oversample.fit_resample(X_train_fold, y_train_fold)

    # Train the model
    logistic_model.fit(X_train_fold_resampled, y_train_fold_resampled)

    # Make predictions on the test set
    y_pred_fold = logistic_model.predict(X_test_fold)

    # Calculate metrics
    accuracies.append(accuracy_score(y_test_fold, y_pred_fold))
    precisions.append(precision_score(y_test_fold, y_pred_fold))
    recalls.append(recall_score(y_test_fold, y_pred_fold))
    f1_scores.append(f1_score(y_test_fold, y_pred_fold))

# Statistics across all folds
accuracies, precisions, recalls, f1_scores = map(np.array, [accuracies, precisions, recalls, f1_scores])

print(f'Average accuracy across all folds: {accuracies.mean():.4f}')
print(f'Standard deviation of accuracy across all folds: {accuracies.std():.4f}')
print(f'Average precision across all folds: {precisions.mean():.4f}')
print(f'Standard deviation of precision across all folds: {precisions.std():.4f}')
print(f'Average recall across all folds: {recalls.mean():.4f}')
print(f'Standard deviation of recall across all folds: {recalls.std():.4f}')
print(f'Average F1 score across all folds: {f1_scores.mean():.4f}')
print(f'Standard deviation of F1 score across all folds: {f1_scores.std():.4f}')

# Train final model on the entire training set with SMOTE
X, y = oversample.fit_resample(X_train, y_train)
logistic_model.fit(X, y)

# Save the model and scaler to files
model_filename = 'logistic_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(logistic_model, file)

scaler_filename = 'scaler.pkl'
with open(scaler_filename, 'wb') as file:
    pickle.dump(scaler, file)

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
