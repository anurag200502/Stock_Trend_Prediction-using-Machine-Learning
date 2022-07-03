# py -m streamlit run app.py

import math
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt  # graph plot
from sklearn.preprocessing import MinMaxScaler  # min max
import keras
from keras.models import load_model
import streamlit as st

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
stock = yf.Ticker(user_input)
df = stock.history(period="max", auto_adjust=True)
st.title("Stock Trend Prediction")

# Describing data
st.write(df.describe())

# Visualization
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(18, 9))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(18, 9))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(18, 9))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

data_training = df['Close'][0:int(len(df) * 0.70)]
data_testing = df['Close'][int(len(df) * 0.70):int(len(df))]

scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(data_training.values.reshape(-1, 1))

# Load model
model = load_model('keras_model (2).h5')

# Testing part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df.values.reshape(-1, 1))

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1 / scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# final graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(20, 10))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig2)
