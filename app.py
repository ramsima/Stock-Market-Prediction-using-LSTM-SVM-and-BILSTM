import numpy as np
import pandas as pd
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler



stock_list = ['NEPSE','HYDROPOWER','MICROFINANCE','RHGCL','CYC','KLBS','AKPL','GHL','NTC','UPPER','NABIL','MLBSL','MLBS','NICA','UNL']

st.header('Stock Market Solution')
stock = st.selectbox('Select desired Stock', stock_list, placeholder='NEPSE')
str = '.csv'
stock = stock + str


# Read the CSV file
data = pd.read_csv(stock)

st.subheader('Stock Data')
st.write(data)


model_list = ['LSTM','BILSTM','SVM']


model_path1 = 'D:\project\Stock-MArket-Forecasting-master\Stock-MArket-Forecasting-master\\'
model_path2 = '.keras'

st.subheader('MODEL')
model_name = st.selectbox('Select desired model', model_list, placeholder='SVM')

model_true_path = model_path1 + model_name + model_path2
model = load_model(model_true_path)


data_train = pd.DataFrame(data.Close[0: int(len(data)*0.70)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.70): len(data)])

scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)


df = data['Close']


# Calculate the EMA with a period of 20 days and a weighting factor of 0.2
df['ema_20'] = df.ewm(span=20, adjust=False, min_periods=20).mean()


st.subheader('Price vs EMA20')
fig11 = plt.figure(figsize=(8,6))
plt.plot(df['ema_20'], 'r', label='EMA_20_DAYS')
plt.plot(data.Close, 'g', label='Closing Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig11)

df1 = data['Close']
# Calculate the EMA with a period of 50 days and a weighting factor of 0.2
df1['ema_50'] = df1.ewm(span=50, adjust=False, min_periods=50).mean()

st.subheader('Price vs EMA50')
fig12 = plt.figure(figsize=(8,6))
plt.plot(df1['ema_50'], 'r', label='EMA_50_DAYS')
plt.plot(data.Close, 'g', label='Closing Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig12)


st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r', label='MA_50_DAYS')
plt.plot(data.Close, 'g', label='Closing Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig1)


st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r', label='MA_50_DAYS' )
plt.plot(ma_100_days, 'b', label='MA_100_DAYS')
plt.plot(data.Close, 'g', label='Closing Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r', label='MA_100_DAYS')
plt.plot(ma_200_days, 'b', label='MA_200_DAYS')
plt.plot(data.Close, 'g', label='Closing Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig3)


x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale
y = y * scale

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label='Predicted Price')
plt.plot(y, 'g', label = 'Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig4)
