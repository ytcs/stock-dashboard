import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from joblib import Parallel,delayed
import plotly.express as px
from functools import partial

import prophet as ph
from wallstreet import Stock, Call, Put

import streamlit as st

st.set_page_config(page_title="Stock Trend Visualizer", page_icon='📈', layout="centered", initial_sidebar_state="auto", menu_items=None)

c1,c2,c3 = st.columns(3)
with c1:
    ticker=st.text_input('Ticker',value='AAPL')
with c2:
    time_horizon=st.text_input('Time Horizon (Days)',value='60')
with c3:
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

# st.dataframe(forecast.tail(time_horizon))

# Option expectation structure
expirations = Call(ticker).expirations
def getprice(strike,expiration):
  c = Call(ticker,strike=strike, d=expiration.day, m=expiration.month, y=expiration.year)
  p = Put(ticker,strike=strike, d=expiration.day, m=expiration.month, y=expiration.year)
  return {'Strike':strike,'CallPrice':(c.ask+c.bid)/2,'PutPrice':(p.ask+p.bid)/2}

def getimpliedprice(ticker,expiration):
  s0 = Stock(ticker).price
  c = Call(ticker, d=expiration.day, m=expiration.month, y=expiration.year)
  p = Put(ticker, d=expiration.day, m=expiration.month, y=expiration.year)
  strikes = np.asarray(list(set(c.strikes) & set(p.strikes)))
  strikes = strikes[(strikes>0.85*s0) & (strikes < 1.15*s0)]
  entries = Parallel(n_jobs=-1)(delayed(partial(getprice,expiration=expiration))(s) for s in strikes)
  df=pd.DataFrame(entries).set_index('Strike').sort_index()

  spl_callprice = UnivariateSpline(df.index,df.CallPrice,k=4)
  spl_putprice = UnivariateSpline(df.index,df.PutPrice,k=4)
  w_call=np.maximum(0,spl_callprice.derivative(2)(df.index))
  s_call=np.sum(w_call*df.index)/np.sum(w_call)
  w_put=np.maximum(0,spl_putprice.derivative(2)(df.index))
  s_put=np.sum(w_put*df.index)/np.sum(w_put)

  return {'Date':expiration,'Call Implied':s_call,'Put Implied': s_put}

df_opt = pd.DataFrame([getimpliedprice(ticker,pd.to_datetime(exp,dayfirst=True)) for exp in expirations[:8]]).set_index('Date')
fig_opt = px.line(df_opt)
st.plotly_chart(fig_opt)