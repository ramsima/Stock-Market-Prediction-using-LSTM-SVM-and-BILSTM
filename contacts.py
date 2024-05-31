
import numpy as np
from numpy import array
import pandas as pd
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#for image
import base64

#utils pkgs
import codecs

# Components pkgs
import streamlit.components.v1 as stc

# importing pages
import home,data,visualization,predict,indicators


with open('style.css') as design:
    source = design.read()

# converting img into python file
def get_img_as_base64(file):
    with open(file, "rb") as g:
        data = g.read()
    return base64.b64encode(data).decode()



def get_fill_color1(label):
    if label>=1:
        return 'rgba(0,255,0,0.6)'
    else:
       return 'rgba(255,0,0,0.6)' 




def app():
  

    

  stock_list = ['NEPSE','HYDROPOWER','MICROFINANCE','AKPL','NTC','UPPER','NABIL','MLBSL','MLBS','NICA','UNL']

  st.header('Select Desired Stock')
  stock = st.selectbox('', stock_list, placeholder='NEPSE')
  str = '.csv'
  stock = stock + str


  # Read the CSV file
  df = pd.read_csv(stock)

  #Conversion line
  hi_val = df['High'].rolling(window=9).max()
  low_val = df['Low'].rolling(window=9).min()
  df['Conversion'] = (hi_val + low_val) /2

  # Base line
  hi_val2 = df['High'].rolling(window=26).max()
  low_val2 = df['Low'].rolling(window=26).min()
  df['Baseline'] = (hi_val2 + low_val2) /2

  #leading span A
  df['SpanA'] = (df['Conversion']+ df['Baseline'])/2
  
  #leading span B
  hi_val3 = df['High'].rolling(window=52).max()
  low_val3 = df['Low'].rolling(window=52).min()
  df['SpanB'] = (hi_val3 + low_val3) /2

  #lagging span
  df['Lagging'] = df['Close'].shift(-26)
  
  #create Candlestick
  candle = go.Candlestick(x=df.index, open=df['Open'].shift(-26),high=df['High'].shift(-26),low=df['Low'].shift(-26),close=df['Close'].shift(-26), name='Candlestick')
  df1 = df

  fig = go.Figure()
  fig = make_subplots(rows=2,cols=1,shared_xaxes="columns", shared_yaxes="rows", column_width=[1], row_heights=[0.6,0.1], subplot_titles=["Candlestick", "Volume"])

  df['label'] = np.where(df['SpanA'] > df['SpanB'], 1,0)
  df['group'] = df['label'].ne(df['label'].shift()).cumsum()

  df = df.groupby('group')

  dfs = []
  for name, data in df:     
    dfs.append(data)

  for df in dfs:
    fig.add_traces(go.Scatter(x=df.index, y= df.SpanA, line=dict(color='rgba(0,250,0,0)'), name ='Cloud'))
    fig.add_traces(go.Scatter(x=df.index, y= df.SpanB, line=dict(color='rgba(250,0,0,0)'), fill='tonexty', fillcolor = get_fill_color1(df['label'].iloc[0]), name='Cloud'))

  baseline = go.Scatter(x=df1.index, y=df1['Baseline'].shift(-26), line=dict(color='red',width = 1),name="Baseline")
  conversion = go.Scatter(x=df1.index, y=df1['Conversion'].shift(-26), line=dict(color='blue',width = 1),name="Conversion")
  lagging = go.Scatter(x=df1.index, y=df1['Lagging'].shift(-26), line=dict(color='green',width = 1),name="Lagging")
  span_a = go.Scatter(x=df1.index, y=df1['SpanA'].shift(0), line=dict(color='green',width = 1),name="SpanA")
  span_b = go.Scatter(x=df1.index, y=df1['SpanB'].shift(0), line=dict(color='red',width = 1),name="SpanB")

  fig.add_trace(candle)
  fig.add_trace(baseline)
  fig.add_trace(conversion)
  fig.add_trace(lagging)
  fig.add_trace(span_a)
  fig.add_trace(span_b)

  fig.add_trace(go.Bar(x=df1.index, y=df1['Volume'], name='Volume'),col=1,row=2)
  fig.update_layout(width=1200,height=800,xaxis_rangeslider_visible=False,showlegend=False)

  fig.show()






    




    