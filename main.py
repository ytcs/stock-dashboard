import pandas as pd
import numpy as np

import prophet as ph
from wallstreet import Stock

import streamlit as st

ticker=st.text_input('Ticker',value='AAPL')
s = Stock(ticker) # stock ticker
time_horizon = 90 # number of days to look ahead


# Prophet
df = s.historical(days_back=600, frequency='d')
df['ds'] = pd.to_datetime(df['Date'])
df['volume'] = np.log(df.Volume)
df['y'] = df['Adj Close']
df = df[['ds','y','volume']]

m = ph.Prophet(weekly_seasonality=False)
m.fit(df)
future = m.make_future_dataframe(periods=time_horizon)
forecast = m.predict(future)
fig=m.plot(forecast)
st.pyplot(fig)