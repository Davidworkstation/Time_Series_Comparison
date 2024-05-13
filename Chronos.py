import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-base",
    device_map="cpu",  # use "cpu" for CPU inference and "mps" for Apple Silicon
    torch_dtype=torch.bfloat16,
)

<<<<<<< HEAD

# Load your data
# data = pd.read_csv('your_data.csv')
df = pd.read_csv(r"/Users/risaiah/Desktop/GitHub Repositories/Time_Series_Cash_Flows/Time_Series_Comparison/JPcashflow.csv")
=======
# Load your data
# data = pd.read_csv('your_data.csv')
df = pd.read_csv(r"C:\Users\David\OneDrive\Desktop\repos\Time_Series_Comparison\JPcashflow.csv")
>>>>>>> 19afc427d85ebd6ef796a4cb911821da81d290c2
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

<<<<<<< HEAD
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
indices_to_drop = df.index[:3]
X = X.drop(X.index[:3])


y = df[df.columns[1]]  # Target variable, ensure this is the correct column
y.index = df[0]
indices_to_drop = df.index[:3]
y = y.drop(y.index[:3])
y = y.iloc[::-1].reset_index(drop=False)
yindex, y = y[0], y[1]
y.index = yindex
print(y.index)


# Split data into training and test sets
split_point = int(len(y) * 0.8)
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

X_test.fillna(0, inplace=True)
y_test.fillna(0, inplace=True)
y_train.fillna(0, inplace=True)
y.fillna(0, inplace=True)

print("xtrain")
print(X_train)

print("xtest")
print(X_test)


print("ytrain")
print(y_train)

print("ytest")
print(y_test)
=======

#split data
# Split data for features and target
X = df.drop(columns=[df.columns[0], df.columns[1], df.columns[2]])  # Features
>>>>>>> 19afc427d85ebd6ef796a4cb911821da81d290c2

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

X_test.fillna(0, inplace=True)
y_test.fillna(0, inplace=True)
y_train.fillna(0, inplace=True)

print("xtest")
print(y_train)
print("Length of Time Series Training Data: ", len(y_train))

print("ytest")
print(y_test)
print("Length of Time Series Test Data: ", len(y_test))

# context must be either a 1D tensor, a list of 1D tensors,
# or a left-padded 2D tensor with batch as the first dimension
context = torch.tensor(y_train)
prediction_length = 7
forecast = pipeline.predict(
    context,
    prediction_length,
    num_samples=1000,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
) # forecast shape: [num_series, num_samples, prediction_length]

# visualize the forecast
forecast_index = np.arange(0, 7)
low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

plt.figure(figsize=(11, 8))
plt.plot(y_test, color="royalblue", label="historical data")
plt.plot(forecast_index, median, color="tomato", label="median forecast")
plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
plt.legend()
plt.grid()
plt.show()

