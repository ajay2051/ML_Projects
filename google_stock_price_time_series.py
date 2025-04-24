import pandas as pd
import yfinance as yf
import datetime
from datetime import date, timedelta
today = date.today()

d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = date.today() - timedelta(days=730)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2

data = yf.download('GOOG',
                      start=start_date,
                      end=end_date,
                      progress=False)
data["Date"] = data.index
data = data[["Date", "Open", "High", "Low", "Close" , "Volume"]]
data.reset_index(drop=True, inplace=True)
print(data.tail())

data = data[["Date", "Close"]]
print(data.head())

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.figure(figsize=(15, 10))
plt.plot(data["Date"], data["Close"])

from statsmodels.tsa.seasonal import seasonal_decompose
decompose = seasonal_decompose(data["Close"], period=7)
decompose.plot()

from statsmodels.tsa.arima.model import ARIMA

p, d, q = 5, 1, 2
model = ARIMA(data["Close"], order=(p, d, q))
fitted = model.fit()
print(fitted.summary())

import statsmodels.api as sm
import warnings
model=sm.tsa.statespace.SARIMAX(data['Close'],
                                order=(p, d, q),
                                seasonal_order=(p, d, q, 12))
model=model.fit()
print(model.summary())

predictions = model.predict(len(data), len(data)+10)
print(predictions)

data["Close"].plot(legend=True, label="Training Data", figsize=(15, 10))
predictions.plot(legend=True, label="Predictions")