import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Get the data for the stock AAPL
data = yf.download('AAPL','2016-01-01','2019-08-01')

# Import the library
from statsmodels.tsa.arima.model import ARIMA

# Fit the model
model = ARIMA(data['Close'], order=(1,1,1))
model_fit = model.fit()

# Summary of the model
print(model_fit.summary())

# Plot the residuals
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()

# Density plot of the residuals
residuals.plot(kind='kde')
plt.show()

