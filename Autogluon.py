from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

# Load your data

# Load your data
# data = pd.read_csv('your_data.csv')
df = pd.read_csv(r"C:\Users\David\OneDrive\Desktop\repos\Time_Series_Comparison\JPcashflow.csv")
df = df.T


# Remove dollar signs and commas
df[1] = df[1].replace('[\$,]', '', regex=True)

# Replace '-' or any other placeholders for missing data with NaN
df[1] = df[1].replace('-', pd.NA)

df[1] = pd.to_numeric(df[1], errors='coerce')

df[0] = pd.to_datetime(df[0], errors='coerce')

time_series_data = pd.DataFrame({
    "Dates":df[0],
    "Values":df[1]
})


#split data
# Split data for features and target
X = df.drop(columns=[df.columns[0], df.columns[1], df.columns[2]])  # Features

# Initialize the scaler
scaler = MinMaxScaler()

# Assuming 'X' is your DataFrame
for column in X.columns:
    X[column] = X[column].replace('[\$,]', '', regex=True)
    X[column] = X[column].replace('-', pd.NA)
    X[column] = pd.to_numeric(X[column], errors='coerce')
    X[column].fillna(0, inplace=True)

# Apply Min-Max scaling
# It's important to drop any NaN values before fitting the scaler as it cannot handle NaNs
X[X.columns] = scaler.fit_transform(X[X.columns])

y = df[df.columns[1]]  # Target variable, ensure this is the correct column

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_test.fillna(0, inplace=True)
y_test.fillna(0, inplace=True)

print("xtrain")
print(X_train)

print("xtest")
print(X_test)

print("ytrain")
print(y_train)

print("ytest")
print(y_test)

#start
data['date_column'] = pd.to_datetime(data['date_column'])
data.set_index('date_column', inplace=True)

# Assuming 'target' is the column you want to forecast
y = data['target']
X = data.drop(columns=['target'])

forecast_horizon = 7  # for example, forecast 7 periods ahead

split_time = '2023-01-01'  # Adjust this to your dataset's appropriate split point
train_data = data[data.index < split_time]
test_data = data[data.index >= split_time]

predictor = TabularPredictor(label='target', path='autogluon_forecasting_models').fit(train_data)
predictions = predictor.predict(test_data.drop(columns=['target']))

#eval

mae = mean_absolute_error(test_data['target'], predictions)
rmse = mean_squared_error(test_data['target'], predictions, squared=False)

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")

predictor.save('my_model_path')  # Save model
loaded_predictor = TabularPredictor.load('my_model_path')  # Load model
