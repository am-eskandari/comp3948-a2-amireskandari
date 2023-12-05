import pandas as pd
from modules.path import PATH, CSV_DATA
from sklearn.impute import SimpleImputer

# from modules.chi_squares import chi_squares

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
print(dataset.head())
print(dataset.describe(include='all'))
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
