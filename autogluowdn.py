import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from autogluon.tabular import TabularPredictor


# Load your data
df = pd.read_csv(r"C:\Users\David\OneDrive\Desktop\repos\Time_Series_Comparison\JPcashflow.csv")
df = df.T  # Transpose if necessary based on your data structure

# Clean and convert data
df[1] = df[1].replace('[\$,]', '', regex=True).replace('-', pd.NA)
df[1] = pd.to_numeric(df[1], errors='coerce')
df[0] = pd.to_datetime(df[0], errors='coerce')

# Prepare DataFrame for time series data
time_series_data = pd.DataFrame({
    "Dates": df[0],
    "Values": df[1]
})

# Assuming df.columns[0] is 'Dates' and df.columns[1] is 'Values' after transposing
X = df.drop(columns=[0, 1, 2])  # Adjust column indices appropriately
X = X.apply(lambda x: pd.to_numeric(x.replace('[\$,]', '', regex=True).replace('-', pd.NA), errors='coerce').fillna(0))

# Initialize and apply the scaler
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# Prepare target variable 'y' from df
y = df[1].dropna()  # Make sure to drop NaN values

# Split data into training and test sets based on the time order
split_point = int(len(y) * 0.8)
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

# Combine X and y for full training and testing datasets
full_train = pd.concat([X_train, y_train.rename('Values')], axis=1)  # Rename y_train explicitly
full_test = pd.concat([X_test, y_test.rename('Values')], axis=1)     # Rename y_test explicitly

# Ensure column names are strings and check them
full_train.columns = [str(col) for col in full_train.columns]
full_test.columns = [str(col) for col in full_test.columns]

print("Training columns:", full_train.columns)  # This will show you what the column names are

predictor = TabularPredictor(label='Values', path='autogluon_forecasting_models').fit(train_data=full_train)
predictions = predictor.predict(full_test.drop(columns=['Values']))
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")
