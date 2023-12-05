import pandas as pd
from modules.cfv import crossfold_validation
from modules.chi_squares import chi_squares
from modules.path import PATH, CSV_DATA
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer

# READING DATA ========================================================================================================
# Define all column names based on the provided sample
all_column_names = [
    'AccountAge', 'MonthlyCharges', 'TotalCharges', 'SubscriptionType', 'PaymentMethod',
    'PaperlessBilling', 'ContentType', 'MultiDeviceAccess', 'DeviceRegistered',
    'ViewingHoursPerWeek', 'AverageViewingDuration', 'ContentDownloadsPerMonth',
    'GenrePreference', 'UserRating', 'SupportTicketsPerMonth', 'Gender',
    'WatchlistSize', 'ParentalControl', 'SubtitlesEnabled', 'CustomerID', 'Churn'
]

# Reading the dataset (the dataset must have a header row, so no need to adjust the 'names' parameter)
dataset = pd.read_csv(PATH + CSV_DATA)
# ======================================================================================================================

# CREATING DUMMY VARIABLES =============================================================================================
# List of categorical columns to convert to dummies
categorical_vars = [
    'SubscriptionType', 'PaymentMethod', 'PaperlessBilling',
    'ContentType', 'MultiDeviceAccess', 'DeviceRegistered',
    'GenrePreference', 'Gender', 'ParentalControl', 'SubtitlesEnabled'
]

# Convert categorical variables into dummy/indicator variables
dummies = pd.get_dummies(dataset[categorical_vars], drop_first=True)
# Drop the original categorical columns from the dataset
dataset = dataset.drop(categorical_vars, axis=1)
# Concatenate the dummy variables with the rest of the dataset
dataset = pd.concat([dataset, dummies], axis=1)
# ======================================================================================================================

# IMPUTING MISSING VALUES ==============================================================================================
imputer = SimpleImputer(strategy='median')
dataset['AccountAge'] = imputer.fit_transform(dataset[['AccountAge']])
dataset['ViewingHoursPerWeek'] = imputer.fit_transform(dataset[['ViewingHoursPerWeek']])
dataset['AverageViewingDuration'] = imputer.fit_transform(dataset[['AverageViewingDuration']])
# ======================================================================================================================

# Show all columns =====================================================================================================
pd.set_option('display.max_columns', None)
# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)
# print(dataset.head())
# print(dataset.describe(include='all'))
# ======================================================================================================================

# FEATURE SELECTION =============================================================================
# Separate into x and y values.
predictorVariables = [
    'AccountAge',  # Chi-Square Score: 112300
    'MonthlyCharges',  # Chi-Square Score: 2648
    'TotalCharges',  # Chi-Square Score: 916700
    'ViewingHoursPerWeek',  # Chi-Square Score: 14790
    'AverageViewingDuration',  # Chi-Square Score: 90500
    'ContentDownloadsPerMonth',  # Chi-Square Score: 25140
    'UserRating',  # Chi-Square Score: 38.11
    'SupportTicketsPerMonth',  # Chi-Square Score: 2235
    'WatchlistSize',  # Chi-Square Score: 390.7
    'SubscriptionType_Premium',  # Chi-Square Score: 120.5
    'SubscriptionType_Standard',  # Chi-Square Score: 1.84
    'PaymentMethod_Credit card',  # Chi-Square Score: 106.3
    'PaymentMethod_Electronic check',  # Chi-Square Score: 32.32
    'PaymentMethod_Mailed check',  # Chi-Square Score: 35.22
    'PaperlessBilling_Yes',  # Chi-Square Score: 0.0234
    'ContentType_Movies',  # Chi-Square Score: 6.165
    'ContentType_TV Shows',  # Chi-Square Score: 6.876
    'MultiDeviceAccess_Yes',  # Chi-Square Score: 0.1939
    'DeviceRegistered_Mobile',  # Chi-Square Score: 0.1322
    'DeviceRegistered_TV',  # Chi-Square Score: 2.595
    'DeviceRegistered_Tablet',  # Chi-Square Score: 0.4985
    'GenrePreference_Comedy',  # Chi-Square Score: 30.39
    'GenrePreference_Drama',  # Chi-Square Score: 1.341
    'GenrePreference_Fantasy',  # Chi-Square Score: 2.161
    'GenrePreference_Sci-Fi',  # Chi-Square Score: 21.63
    'Gender_Male',  # Chi-Square Score: 4.944
    'ParentalControl_Yes',  # Chi-Square Score: 4.092
    'SubtitlesEnabled_Yes'  # Chi-Square Score: 15.8
]
X = dataset[predictorVariables]
y = dataset['Churn']

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# print("Shape of X_scaled:", X_scaled.shape)

# k = best_k(X_scaled, y)  # Output: Best k: 15
# print("Best k: " + str(k))
crossfold_validation(X_scaled, y)
selected_features = chi_squares(X, X_scaled, y, 15)
# ======================================================================================================================

# LOGISTIC REGRESSION ==================================================================================================
# Re-assign X with significant columns only after chi-square test.
X = dataset[selected_features]

# Split data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=0)
# Build logistic regression model and make predictions.
logisticModel = LogisticRegression(fit_intercept=True, solver='liblinear',
                                   random_state=0)
logisticModel.fit(X_train, y_train)
y_pred = logisticModel.predict(X_test)
print(y_pred)
# ======================================================================================================================

# EVALUATION ===========================================================================================================
cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
print('\nAccuracy: ', metrics.accuracy_score(y_test, y_pred))
print("\nConfusion Matrix")
print(cm)

TN = cm[0][0]  # = 3 True Negative (Col 0, Row 0)
FN = cm[0][1]  # = 0 False Negative (Col 0, Row 1)
FP = cm[1][0]  # = 2 False Positive (Col 1, Row 0)
TP = cm[1][1]  # = 5 True Positive (Col 1, Row 1)
print("")
print("True Negative: " + str(TN))
print("False Negative: " + str(FN))
print("False Positive: " + str(FP))
print("True Positive: " + str(TP))
precision = (TP / (FP + TP))
print("\nPrecision: " + str(round(precision, 3)))
recall = (TP / (TP + FN))
print("Recall: " + str(round(recall, 3)))
F1 = 2 * ((precision * recall) / (precision + recall))
print("F1: " + str(round(F1, 3)))

# Define scoring metrics
scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, zero_division=0),
    'recall': make_scorer(recall_score, zero_division=0),
    'f1': make_scorer(f1_score, zero_division=0)
}

# Perform cross-validation
cv_results = cross_validate(LogisticRegression(solver='liblinear', random_state=0),
                            X_scaled, y, cv=5, scoring=scoring, return_train_score=False)

# Calculate mean and standard deviation for each metric
mean_accuracy = cv_results['test_accuracy'].mean()
std_accuracy = cv_results['test_accuracy'].std()
mean_precision = cv_results['test_precision'].mean()
std_precision = cv_results['test_precision'].std()
mean_recall = cv_results['test_recall'].mean()
std_recall = cv_results['test_recall'].std()
mean_f1 = cv_results['test_f1'].mean()
std_f1 = cv_results['test_f1'].std()

# Output the results
print(f'Average accuracy across all folds: {mean_accuracy:.4f}')
print(f'Standard deviation of accuracy across all folds: {std_accuracy:.4f}')
print(f'Average precision across all folds: {mean_precision:.4f}')
print(f'Standard deviation of precision across all folds: {std_precision:.4f}')
print(f'Average recall across all folds: {mean_recall:.4f}')
print(f'Standard deviation of recall across all folds: {std_recall:.4f}')
print(f'Average F1 score across all folds: {mean_f1:.4f}')
print(f'Standard deviation of F1 score across all folds: {std_f1:.4f}')
# ======================================================================================================================
