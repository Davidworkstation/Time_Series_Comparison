Introduction to Time Series Forecasting
Time series forecasting involves using historical data points, indexed in time order, to make predictions about future values. This analytical approach is pivotal across various domains, such as finance, where it aids in predicting trends like cash flows or stock prices. The essence of time series analysis lies in identifying the patterns in time-stamped data and extrapolating these observations to forecast future events. The process can be complex due to the inherent noise, seasonality, trends, and other external influences that affect time series data.

Forecasting Techniques and Models
Several methodologies are employed for time series forecasting, ranging from traditional statistical models to advanced machine learning algorithms. ARIMA (AutoRegressive Integrated Moving Average) and VAR (Vector Autoregression) are traditional models that focus on linear relationships among sequential data points and are widely used for their simplicity and effectiveness in handling various time series quirks like seasonality and trend. On the other hand, machine learning approaches like LightGBM and AutoGluon offer robust alternatives that cater to nonlinear complexities in data which traditional models might fail to capture. LightGBM is a gradient boosting framework that uses tree-based learning algorithms, while AutoGluon automates machine learning tasks, allowing for effective model selection and hyperparameter tuning.

Data Collection and Project Application
The project utilizes web scraping techniques to automatically gather quarterly cash flow data from J.P. Morgan, facilitating real-time data analysis and enhancing the accuracy of forecasts. This automated data collection is crucial for maintaining an up-to-date dataset, which is essential for the accuracy of any forecasting model. By comparing the performance of traditional models like ARIMA and VAR with advanced techniques such as LightGBM and AutoGluon, as well as exploring unsupervised methods in Chronos, this project aims to identify the most efficient and accurate models for forecasting financial time series. This comparative analysis will help in understanding the strengths and limitations of each approach, guiding financial analysts in choosing the appropriate forecasting model for their specific needs.

Files:
    1. Webscrape.py - generates a csv files of JP Morgan's quarterly cash flow summaries from 2009 to 2023. Uses Selenium to open a dummy webpage to load necessary html and javascript, and performs simple browser interactions to capture the entirety of the data in jqx table format. 
    2. VAR.py - 
    3. ARIMA.py - 
    4. LightGBM_Regressor.py
    5. Autogluon.py
    6. 