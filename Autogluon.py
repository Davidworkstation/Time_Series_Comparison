from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

# Load your data
data = pd.read_csv('path_to_your_data.csv')
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
