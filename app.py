import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
import datetime

# start='2010-01-01'
# end='2019-12-31'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
if "start_date" not in st.session_state:
    st.session_state.start_date = datetime.date(2010, 1, 1)

if "end_date" not in st.session_state:
    st.session_state.end_date = datetime.date(2019, 12, 31)

# Let users select a date
start_date = st.date_input(
    "Select a start date",
    value=st.session_state.start_date,
    min_value=datetime.date(1900, 1, 1),
    max_value=datetime.date(2030, 12, 31)
)

end_date = st.date_input(
    "Select an end date",
    value=st.session_state.end_date,
    min_value=datetime.date(1900, 1, 1),
    max_value=datetime.date(2030, 12, 31)
)

# Update session state with selected values
st.session_state.start_date = start_date
st.session_state.end_date = end_date

if st.button("Predict Stock Trends"):
    df=yf.download(user_input,start=start_date,end=end_date)

    # Describing the data
    st.subheader(f'Data from {start_date}  ---  {end_date}')
    st.write(df.describe())

    # Visualizations
    st.subheader('Closing Price vs Time chart')
    fig = plt.figure(figsize=(12,6))
    plt.plot(df['Close'])
    st.pyplot(fig)

    # Moving Average 100
    st.subheader('Closing Price vs Time chart with 100MA')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize=(12,6))
    plt.plot(ma100,'r')
    plt.plot(df.Close,'b')
    st.pyplot(fig)

    # Moving Average 200
    st.subheader('Closing Price vs Time chart with 100MA & 200MA')
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize=(12,6))
    plt.plot(ma100,'r')
    plt.plot(ma200,'g')
    plt.plot(df.Close,'b')
    st.pyplot(fig)

    # Slitting data into Training and Testing
    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))
    data_training_array = scaler.fit_transform(data_training)



    # Load the model
    model = load_model('keras_model.h5')

    past_100_days = data_training.tail(100)
    ## final_df = past_100_days.append(data_testing, ignore_index=True)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

    input_data=scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100 : i])
        y_test.append(input_data[i,0])
    x_test, y_test= np.array(x_test), np.array(y_test)

    y_pred = model.predict(x_test)

    
    # scale_factor = 1/0.02134523
    # y_predicted = y_predicted * scale_factor
    # y_test = y_test * scale_factor
    scaler = scaler.scale_
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(data_training)
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Actual vs Predicted graph
    st.subheader('Pedicted vs Actual Prices')
    fig2 = plt.figure(figsize=(12,6))
    plt.plot(y_test,'b',label='Orginal Price')
    plt.plot(y_pred,'r',label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mse=mean_squared_error(y_test,y_pred)
    print(mse)
    r2 = r2_score(y_test, y_pred)
    print(r2)
    non_zero_indices = y_test != 0  # Filter out zero values
    mape = np.mean(np.abs((y_test[non_zero_indices] - y_pred[non_zero_indices]) / y_test[non_zero_indices])) * 100
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    st.write("Mean Absolute Percentage Error (MAPE): ",mape)