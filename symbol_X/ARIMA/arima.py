#create arima mlodel for time series forecasting importing prices from yahoo finance
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import matplotlib.dates as mdates


def get_data(symbol):
    start = dt.datetime(2020,6,1)
    end = dt.datetime(2024,8,1)

    data = yf.download(symbol, start=start, end=end)
    data = data['Close'] #.diff().dropna()
    return data


data = get_data('AAPL')

#---------------plot the values of data to see the trend----------
# plt.plot(data)
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.title('AAPL Stock Price')
# plt.show()

#-----------------Check ADF (check stationary)-----------------
# adf_test = adfuller(data)
# print('ADF Statistic: %f' % adf_test[0])
# print('p-value: %f' % adf_test[1]) #if p-value < 0.05, the data is stationary


#-----------------Check ACF (differencing order=2)-----------------

# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# import matplotlib.pyplot as plt
# plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})


# # Original Series
# fig, axes = plt.subplots(4, 2, sharex=False)
# plt.subplots_adjust(hspace=0.5)
# axes[0, 0].plot(data); axes[0, 0].set_title('Original Series')
# plot_acf(data, ax=axes[0, 1])

# #ajoute de l'espace pour que les graphiques ne se touchent pas


# # 1st Differencing
# axes[1, 0].plot(data.diff()); axes[1, 0].set_title('1st Order Differencing')
# plot_acf(data.diff().dropna(), ax=axes[1, 1])

# # 2nd Differencing
# axes[2, 0].plot(data.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
# plot_acf(data.diff().diff().dropna(), ax=axes[2, 1])

# # 3rd Differencing
# axes[3, 0].plot(data.diff().diff().diff()); axes[3, 0].set_title('3rd Order Differencing')
# plot_acf(data.diff().diff().dropna(), ax=axes[3, 1])

# plt.show()

#-----------------Check PACF (check p order=1)-----------------
# # PACF plot of 1st differenced series
# plt.rcParams.update({'figure.figsize': (10, 8), 'figure.dpi': 120})

# # Create the subplots: 2 rows, 2 columns
# fig, axes = plt.subplots(2, 2, sharex=False)

# # First differencing
# axes[0, 0].plot(data.diff())
# axes[0, 0].set_title('1st Differencing')
# plot_pacf(data.diff().dropna(), ax=axes[0, 1])

# # Second differencing
# axes[1, 0].plot(data.diff().diff())
# axes[1, 0].set_title('2nd Differencing')
# plot_pacf(data.diff().diff().dropna(), ax=axes[1, 1])

# # Set y-limits for PACF plots
# axes[1, 1].set_ylim(-1, 2)
# axes[0, 1].set_ylim(-1, 2)


# # Setting custom tick positions and formatting for the x-axis on the first differencing plot
# tick_positions = pd.date_range(start='2023-06-01', end='2024-08-01', freq='M')  # Monthly ticks
# axes[0, 0].set_xticks(tick_positions)
# axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%m'))

# # Apply the same formatting to the second differencing plot, if desired
# axes[1, 0].set_xticks(tick_positions)
# axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%m'))

# plt.tight_layout()
# plt.show()

#-----------------Check q order=1-----------------

# plt.rcParams.update({'figure.figsize': (9, 6), 'figure.dpi': 120})

# Create the subplots: 2 rows, 2 columns
# fig, axes = plt.subplots(2, 2, sharex=False)

# First differencing
# axes[0, 0].plot(data.diff())
# axes[0, 0].set_title('1st Differencing')
# plot_acf(data.diff().dropna(), ax=axes[0, 1])
# axes[0, 1].set_ylim(-1, 2)

# Second differencing
# axes[1, 0].plot(data.diff().diff())
# axes[1, 0].set_title('2nd Differencing')
# plot_acf(data.diff().diff().dropna(), ax=axes[1, 1])
# axes[1, 1].set_ylim(-1, 2)

# plt.show()

#-----------ARIMA MODEL-----------------------

# # 1,2,1 ARIMA Model
# model = ARIMA(data, order=(1,2,1))
# model_fit = model.fit()
#print(model_fit.summary())

#------Plot residual errors----------
# residuals = pd.DataFrame(model_fit.resid)
# fig, ax = plt.subplots(1,2)
# residuals.plot(title="Residuals", ax=ax[0])
# ax[0].set_ylim(-25, 25)
# residuals.plot(kind='kde', title='Density', ax=ax[1])
# ax[1].set_xlim(-50,50)

# plt.show()

#-----------Actual vs Fitted-------------
# from statsmodels.graphics.tsaplots import plot_predict

# #fitted values
# plot_predict(model_fit, dynamic=False)

# #actual data
# plt.plot(data, label='actual', color='red')
# plt.legend()

# plt.ylim(160,245)
# plt.ylabel('Price')

# plt.show()

#-----------------FORECASTING-----------------

# from statsmodels.tsa.stattools import acf

# Create Training and Test
# train = data[:85]
# test = data[85:]


# # Build Model
# model = ARIMA(train, order=(1, 2, 1))  
# fitted = model.fit()  

# # Forecast
# #fc, se, conf = fitted.forecast(119, alpha=0.05)  # 95% conf

# forecast = fitted.get_forecast(steps=119)
# forecast_mean = forecast.predicted_mean
# forecast_conf_int = forecast.conf_int()

# y = pd.Series(data, index=data.index)

# forecast_index = pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), periods=len(forecast_mean), freq='D')

# # Plot
# plt.figure(figsize=(12,5), dpi=100)
# plt.plot(train, label='training')
# plt.plot(test, label='actual')

# # Plot forecasted values
# plt.plot(forecast_index, forecast_mean, color='red', label='Forecast')

# # Plot confidence intervals
# plt.fill_between(forecast_index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='red', alpha=0.15, label='95% Confidence Interval')

# plt.title('Forecast vs Actuals')
# plt.legend(loc='upper left', fontsize=8)
# plt.show()

#---UP TO DATE----------
# from sklearn.metrics import mean_squared_error


# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

train = train.diff().diff().dropna()
test = test.diff().diff().dropna()

# # Fit an ARIMA model
# model = ARIMA(train, order=(1, 2, 0))  # Example: ARIMA(5,1,0)
# model_fit = model.fit()

# # Make predictions
# predictions = model_fit.forecast(steps=len(test))

# # Plot the predictions against the actual values
# plt.figure(figsize=(10, 4))
# plt.plot(train.index, train, color='blue', label='Training Data')
# plt.plot(test.index, test, color='green', label='Actual Data')
# plt.plot(test.index, predictions, color='red', linestyle='dashed', label='Predicted Data')
# plt.title('ARIMA Model - Predictions vs Actual')
# plt.legend()
# plt.show()

# # Calculate the Mean Squared Error
# mse = mean_squared_error(test, predictions)
# print(f'Mean Squared Error: {mse}')


#-----------------FORECASTING-----------------
# import pmdarima as pm
# auto_arima = pm.auto_arima(data, stepwise=False, seasonal=False)
# print(auto_arima.summary(), "\n") #

#----------------Seasonal Decomposition-----------------

# from statsmodels.tsa.seasonal import seasonal_decompose 

# # Convert the index to a DatetimeIndex if it's not already
# data.index = pd.to_datetime(data.index)

# # Specify the period manually (e.g., 12 for monthly data assuming yearly seasonality)
# decompose_data = seasonal_decompose(data, model="additive", period=12)
# decompose_data.plot()


#----------------SARIMA Model-----------------

# #Build the SARIMA model
# model = SARIMAX(train, order=(3, 1, 0), seasonal_order=(2, 2, 1, 30))
# fitted = model.fit()
# print(fitted.summary())

# # Forecast
# fc = fitted.get_forecast(steps=len(test))

# # Manually create a date range for the forecast to align with the test data
# fc_index = pd.date_range(start=test.index[0], periods=len(test), freq=test.index.freq)
# fc_series = pd.Series(fc.predicted_mean.values, index=fc_index)

# # Get confidence intervals and align them with the correct index
# conf = fc.conf_int(alpha=0.05)
# lower_series = pd.Series(conf.iloc[:, 0], index=fc_index)
# upper_series = pd.Series(conf.iloc[:, 1], index=fc_index)

# # Plot the results
# plt.figure(figsize=(12, 5), dpi=100)
# plt.plot(train, label='Training Data')
# plt.plot(test, label='Actual Data')
# plt.plot(fc_series, label='Forecast')
# plt.fill_between(lower_series.index, lower_series, upper_series,
#                     color='k', alpha=.15)
# plt.title('Forecast vs Actuals')
# plt.legend(loc='upper left', fontsize=8)
# plt.show()




#-------------ARIMA Model-----------------

# Build the ARIMA model
model = ARIMA(train, order=(2, 2, 1))  
fitted = model.fit()  
print(fitted.summary())

# Forecast
fc = fitted.get_forecast(steps=len(test))


# Manually create a date range for the forecast to align with the test data
fc_index = pd.date_range(start=test.index[0], periods=len(test), freq=test.index.freq)
fc_series = pd.Series(fc.predicted_mean.values, index=fc_index)


# Get confidence intervals and align them with the correct index
conf = fc.conf_int(alpha=0.05)
lower_series = pd.Series(conf.iloc[:, 0], index=fc_index)
upper_series = pd.Series(conf.iloc[:, 1], index=fc_index)

# Plot the results
plt.figure(figsize=(12, 5), dpi=100)
plt.plot(train, label='Training Data')
plt.plot(test, label='Actual Data')
plt.plot(fc_series, label='Forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.xlim('2023-06-01', '2024-08-01')
plt.show()

from sklearn.metrics import mean_squared_error
# Calculate the Mean Squared Error
mse = mean_squared_error(test, fc_series)
print(f'Mean Squared Error: {mse}')

