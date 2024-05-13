from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pmdarima.arima.utils import ndiffs
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf


# Load your data
# data = pd.read_csv('your_data.csv')
df = pd.read_csv(r"/Users/risaiah/Desktop/GitHub Repositories/Time_Series_Cash_Flows/Time_Series_Comparison/JPcashflow.csv")
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


#differencing
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

# Perform ADF test
result = adfuller(y_train)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# Original Series
fig, axes = plt.subplots(3, 2, sharex='col')  # Share x-axis only within columns

# Plotting time series data with specific x-axis limits
min_date = y.index.min()
max_date = y.index.max()

# Original Series
axes[0, 0].plot(y.index, y); 
axes[0, 0].set_title('Original Series')
axes[0, 0].set_xlim(min_date, max_date)  # Set x-limits for the time series plot

plot_acf(y, ax=axes[0, 1])  # Autocorrelation plot without x-limits

# First Order Differencing
y_first_diff = y.diff().dropna()
axes[1, 0].plot(y_first_diff.index, y_first_diff); 
axes[1, 0].set_title('1st Order Differencing')
axes[1, 0].set_xlim(min_date, max_date)  # Set x-limits for the time series plot

plot_acf(y_first_diff, ax=axes[1, 1])  # Autocorrelation plot without x-limits

# Second Order Differencing
y_second_diff = y.diff().diff().dropna()
axes[2, 0].plot(y_second_diff.index, y_second_diff); 
axes[2, 0].set_title('2nd Order Differencing')
axes[2, 0].set_xlim(min_date, max_date)  # Set x-limits for the time series plot

plot_acf(y_second_diff, ax=axes[2, 1])  # Autocorrelation plot without x-limits

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

df = y

# Adf Test
ndiffs(y, test='adf')  # 2

# KPSS test
ndiffs(y, test='kpss')  # 0

# PP test:
ndiffs(y, test='pp')  # 2

# PACF plot of 1st differenced series
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})


fig, axes = plt.subplots(1, 2)
axes[0].plot(y.diff()); axes[0].set_title('1st Differencing')
axes[0].set_xlim(min_date, max_date)

plot_pacf(y.diff().dropna(), ax=axes[1])
axes[1].set_xlim(0,17.5)

plt.show()


# 1,1,2 ARIMA Model
model = ARIMA(df, order=(1,1,2))
model_fit = model.fit()
print(model_fit.summary())

# 1,1,1 ARIMA Model
model = ARIMA(df, order=(1,1,1))
model_fit = model.fit()
print(model_fit.summary())

# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

# Get predictions starting from the start of the series to the end
pred = model_fit.get_prediction(start=df.index[0], end=df.index[-1], dynamic=False)
pred_conf = pred.conf_int()

# Plot the series and the forecasted values with confidence intervals
plt.figure(figsize=(12, 6))
plt.plot(df.index, df, label='Observed', color='blue')
plt.plot(pred.predicted_mean.index, pred.predicted_mean, label='Forecast', color='red')

# Plot the confidence intervals
plt.fill_between(pred_conf.index, pred_conf.iloc[:, 0], pred_conf.iloc[:, 1], color='pink')
plt.legend()
plt.show()

# Create Training and Test
train = y_train
test = y_test

print(test.index)
print(test)

def forecast_accuracy(forecast, actual):
    mape = np.nanmean(np.abs(forecast - actual) / np.abs(actual))  # MAPE
    me = np.nanmean(forecast - actual)  # ME
    mae = np.nanmean(np.abs(forecast - actual))  # MAE
    mpe = np.nanmean((forecast - actual) / actual)  # MPE
    rmse = np.sqrt(np.mean((forecast - actual)**2))  # RMSE
    corr = np.corrcoef(forecast, actual)[0, 1]  # Correlation
    mins = np.minimum(forecast, actual)
    maxs = np.maximum(forecast, actual)
    minmax = 1 - np.nanmean(mins / maxs)  # Min-Max Error
    acf1 = np.corrcoef(forecast[:-1], forecast[1:])[0, 1]  # ACF1 of forecast

    return {
        'mape': mape,
        'me': me,
        'mae': mae,
        'mpe': mpe,
        'rmse': rmse,
        'acf1': acf1,
        'corr': corr,
        'minmax': minmax
    }

def evaluate_arima_model(train, test, arima_order, forecast_steps, fig=False):
    model = ARIMA(train, order=arima_order)
    fitted = model.fit()
    
    min_date_new = train.index.min()
    max_date_new = test.index.max()
    
    forecast_result = fitted.get_forecast(steps=forecast_steps)
    fc_series = forecast_result.predicted_mean
    conf = forecast_result.conf_int()
    
    lower_series = pd.Series(conf.iloc[:, 0], index=test.index)
    upper_series = pd.Series(conf.iloc[:, 1], index=test.index)
    
    figure = plt.figure(figsize=(12, 5), dpi=100)
    sns.lineplot(x=train.index,y=train.values, label='Training', linestyle='--')
    sns.lineplot(x=test.index,y=test.values, label='Actual', marker='o') 
    sns.lineplot(fc_series, label='Forecast', color='red')
    plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
    plt.title(f'Forecast vs Actuals for ARIMA Order {arima_order}')
    plt.legend(loc='upper left', fontsize=8)
    plt.xlim(min_date_new, max_date_new) 
    plt.show()
    
    # Ensure test and forecast series are aligned and have no NaN values
    test_adjusted = test.replace(0, np.finfo(float).eps).ffill()
    fc_series = fc_series.ffill()
    
    return forecast_accuracy(fc_series, test_adjusted)


def evaluate_arima_model_subplots(train, test, arima_order, forecast_steps, ax=None):
    model = ARIMA(train, order=arima_order)
    fitted = model.fit()

    forecast_result = fitted.get_forecast(steps=forecast_steps)
    fc_series = forecast_result.predicted_mean
    conf = forecast_result.conf_int()

    lower_series = pd.Series(conf.iloc[:, 0], index=test.index)
    upper_series = pd.Series(conf.iloc[:, 1], index=test.index)
    
    if ax is None:
        ax = plt.gca()  # Gets the current active axis

    sns.lineplot(x=train.index, y=train.values, label='Training', linestyle='--', ax=ax)
    sns.lineplot(x=test.index, y=test.values, label='Actual', marker='o', ax=ax)
    sns.lineplot(x=fc_series.index, y=fc_series.values, label='Forecast', color='red', ax=ax)
    ax.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
    ax.set_title(f'Forecast vs Actuals for ARIMA Order {arima_order}')
    ax.set_ylim(-1000, 25000)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlim(train.index.min(), test.index.max())

    # Return the axis if needed outside the function
    return ax
    
# Run evaluation for multiple configurations
orders = [
    (0, 0, 0),  # Naive model, mean of the series
    (1, 0, 0),  # AR model with one lag
    (0, 0, 1),  # MA model with one lag
    (1, 1, 0),  # First-order differencing with one AR term
    (0, 1, 1),  # First-order differencing with one MA term
    (1, 1, 1),  # First-order differencing with one AR and one MA term
    (0, 2, 1),  # Second-order differencing with one MA term
    (1, 2, 0),  # AR model with second-order differencing
    (1, 2, 1),  # Both AR and MA with second-order differencing
    (2, 1, 0),  # Two AR terms with one differencing
    (0, 1, 2),  # Two MA terms with one differencing
    (2, 1, 2),  # More complex model with two AR and two MA terms
    (3, 1, 0),  # Three AR terms
    (0, 1, 3),  # Three MA terms
    (3, 1, 1),  # Three AR terms and one MA term
    (1, 1, 3),  # One AR term and three MA terms
    (3, 1, 2),  # Three AR terms and two MA terms
    (3, 2, 1)   # Three AR terms, two differencing steps, and one MA term
]

results = {}
for order in orders:
    results[order] = evaluate_arima_model(train, test, order, len(test))

# Print the results
for order, accuracy in results.items():
    print(f"ARIMA Order {order}:")
    print(accuracy)
    
fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(30, 20))  # Adjust the size and layout as needed
axes = axes.flatten()  # Flatten the array of axes if needed

# Assuming you have multiple sets of train, test data or orders
for i, order in enumerate(orders):
    ax = axes[i]
    evaluate_arima_model_subplots(train, test, order, forecast_steps = len(test), ax=ax)

plt.tight_layout()
plt.show()
        
    


#references
#https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/