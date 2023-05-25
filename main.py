import pandas as pd
import numpy as np

import prophet as ph
from wallstreet import Stock

import streamlit as st

st.set_page_config(page_title="Stock Trend Visualizer", page_icon='📈', layout="centered", initial_sidebar_state="auto", menu_items=None)

ticker=st.text_input('Ticker',value='AAPL')
time_horizon=st.text_input('Time Horizon (Days)',value='60')
days_back=st.text_input('Time Lookback (Days)',value='600')
s = Stock(ticker) # stock ticker
time_horizon = int(time_horizon) # number of days to look ahead


# Prophet
df = s.historical(days_back=int(days_back), frequency='d')
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

st.dataframe(forecast.tail(time_horizon))