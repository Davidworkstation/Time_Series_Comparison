import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
import optuna
import numpy as np
from sklearn.metrics import make_scorer, mean_squared_error
import pandas as pd
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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


# Database configuration for storing study results
mysqldb = 'mysql+pymysql://root:Hawaii808!@localhost:3306/time_series_lightgbm_forecast'
def objective(trial):
    light_param = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'num_leaves': trial.suggest_int('num_leaves', 5, 100),
        'max_depth': trial.suggest_int('max_depth', 2, 50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.99),
        'n_estimators': trial.suggest_int('n_estimators', 1000, 10000),
        'subsample_for_bin': trial.suggest_int('subsample_for_bin', 1000, 10000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.25, 0.99),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 3.5, 4.5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.9, 1.0),
        'random_state': trial.suggest_categorical('random_state', [None, 42]),
        'n_jobs': -1
    }
    model = LGBMRegressor(**light_param)
    tscv = TimeSeriesSplit(n_splits=5)
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
    scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring=mse_scorer)
    return np.nanmean(scores)

# Create and optimize the study
study = optuna.create_study(direction='minimize', study_name='Time_Series_Forecasting_JPmorg', storage=mysqldb, load_if_exists=True)
study.optimize(objective, n_trials=70)

# Display best parameters
print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Retrain the model with the best parameters
best_params = trial.params
best_model = LGBMRegressor(**best_params)
best_model.fit(X_train, y_train)

# Make predictions with the test set
predictions = best_model.predict(X_test)

plt.figure(figsize=(10, 5))
plt.plot(predictions, label='Predicted Values', color='red')
plt.title('Predicted Values Distribution')
plt.legend()
plt.show()

print(y_test)
print(predictions)

mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label='Actual', color='blue')  
plt.plot(y_test.index, predictions, label='Predicted', color='red')
plt.title('Model Predictions vs Actual Data')
plt.xlabel('Date')
plt.ylabel('Target Variable')
plt.legend()
plt.show()

# Calculate errors
y_test_filtered = y_test
predictions_filtered = predictions[~pd.isna(y_test)]
errors = y_test_filtered - predictions_filtered

plt.hist(errors.dropna(), bins=30, alpha=0.5, color='g')  # Drop NaN values to avoid errors in histogram plotting
plt.title('Error Distribution')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.show()