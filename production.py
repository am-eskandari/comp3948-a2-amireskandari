import pandas as pd
import pickle

# Reading the dataset
PATH = "/Users/test/Documents/Programming Directories/BCIT/COMP 3948/comp3948-a2-amireskandari/datasets/"
CSV_DATA = "CustomerChurn_Mystery.csv"

dataset = pd.read_csv(PATH + CSV_DATA)

# Apply median imputation for missing values in numeric columns
dataset.fillna(dataset.median(numeric_only=True), inplace=True)

# Creating dummy variables
categorical_vars = ['SubscriptionType', 'PaymentMethod', 'PaperlessBilling', 'ContentType',
                    'MultiDeviceAccess', 'DeviceRegistered', 'GenrePreference', 'Gender',
                    'ParentalControl', 'SubtitlesEnabled']
dataset[categorical_vars] = dataset[categorical_vars].fillna('not exist')
dummyDf = pd.get_dummies(dataset[categorical_vars])
dataset = pd.concat([dataset, dummyDf], axis=1)

# Clip 'TotalCharges' to the upper boundary and binning
dataset['TotalCharges'] = dataset['TotalCharges'].clip(upper=2250)

# Adjust binning
bins = [0, 500, 1500, 2250]
labels = ['TotalCharges_0_500', 'TotalCharges_500_1500', 'TotalCharges_1500_2250']
dataset['TotalChargesBin'] = pd.cut(dataset['TotalCharges'], bins=bins, labels=labels, right=False)
additional_dummies = pd.get_dummies(dataset['TotalChargesBin'])
dataset = pd.concat([dataset, additional_dummies], axis=1)

# Adjusted predictorVariables
predictorVariables = [
    'AccountAge', 'MonthlyCharges', 'ViewingHoursPerWeek',
    'AverageViewingDuration', 'ContentDownloadsPerMonth',
    'UserRating', 'SupportTicketsPerMonth', 'WatchlistSize',
    'SubscriptionType_Premium',  # Assuming this exists in the dummy variables
    'PaymentMethod_Mailed check',  # Replace with actual PaymentMethod dummy variables
    'GenrePreference_Comedy',      # Assuming this exists in the dummy variables
    'GenrePreference_Fantasy',     # Replace with actual GenrePreference dummy variables
    'TotalCharges_0_500', 'TotalCharges_500_1500', 'TotalCharges_1500_2250'
]

X = dataset[predictorVariables]

# Load the saved MinMaxScaler
with open("scaler.pkl", "rb") as file:
    loaded_scaler = pickle.load(file)

# Apply the loaded scaler to the data
X_scaled = loaded_scaler.transform(X)

# Load the saved Logistic Regression model
with open("logistic_model.pkl", "rb") as file:
    loaded_logistic_model = pickle.load(file)

# Make predictions with the loaded model
y_pred = loaded_logistic_model.predict(X_scaled)

# Output the predictions
predictions = pd.DataFrame(y_pred, columns=['Predicted_Churn'])
print(predictions.head())

# Optionally, save the predictions to a CSV file
predictions.to_csv('Predictions.csv', index=False)
