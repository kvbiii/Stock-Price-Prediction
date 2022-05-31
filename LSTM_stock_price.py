#Importing the libraries
import datetime as dt
import math
import pandas_datareader as web
import numpy as np
import pandas as pd    
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

#Our picked company
company = 'AAPL'

#Starting and ending dates of the data which will be imported
start = dt.datetime(2000, 1, 1)
end = dt.datetime.now()

#Importing data
df = web.DataReader(company, 'yahoo', start, end)

prediction_days = 60

#Creating a new data frame with only 'Close' prices
data = df.filter(['Close'])

#Converting data frame to array
dataset = data.values

#Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset.reshape(-1, 1))
training_data_len = math.ceil(len(dataset) * 0.8)

#Creating scaled training data
train_data = scaled_data[0:training_data_len , :]

#Spliting the data into x_train and y_train
x_train = []
y_train = []
for i in range(prediction_days, len(train_data)):
    x_train.append(train_data[i-prediction_days:i, 0])
    y_train.append(train_data[i, 0])

#Converting x_train and y_train to arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshaping x_train
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Creating the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#Compiling the model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Training the model
model.fit(x_train, y_train, batch_size = 1, epochs = 1)

#Creating testing data
test_data = scaled_data[training_data_len - prediction_days: , :]

#Creating x_test and y_test
x_test = []
y_test = dataset[training_data_len:]
for i in range(prediction_days, len(test_data)):
    x_test.append(test_data[i-prediction_days:i, 0])

#Coverting x_test to array
x_test = np.array(x_test)

#Reshaping x_test
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#Getting the model predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#Plotting the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

#Creating new data frame to get what price the model predicted for chosen day
test_start = dt.datetime(2015, 1, 1)
test_end = dt.datetime.now()
new_df = web.DataReader(company, 'yahoo', test_start, test_end)
new_data = new_df.filter(['Close'])

#Getting the last 60 days and coverting that to new data frame
last_60_days = new_data[-prediction_days:].values

#Scaling the data
last_60_days_scaled = scaler.transform(last_60_days)

#New X_test
X_test = []
X_test.append(last_60_days_scaled)

#Coverting X_test to array
X_test = np.array(X_test)

#Reshaping X_test
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#Visualizing the data
plt.figure(figsize=(16, 8))
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Price ($)', fontsize = 18)
plt.legend(loc="best")
plt.title('LSTM model price prediction for {}'.format(company))
plt.show()

#Getting the root mean squared error (RMSE) which is the standard deviation of the prediction errors
print('Mean absolute sqaured error is: {}'.format(np.sqrt(np.mean(predictions - y_test) ** 2)))

#Creating the value which is a correlation between real price and our model predicted price
corr_price, p = stats.pearsonr(valid['Close'].values, valid['Predictions'].values)
print("Correlation between actual price and predicted by our model price is: {}".format(corr_price))

#Recreating my arrays to create summarizing data frame
y_test = y_test.flatten().tolist()
predictions = predictions.flatten().tolist()
columns =  ['Close_actual', 'Close_pred']
df_pred_LSTM = pd.DataFrame(list(zip(y_test, predictions)), columns = columns)
df_pred_LSTM['diff'] = df_pred_LSTM.Close_actual.values - df_pred_LSTM.Close_pred.values
df_pred_LSTM['perc_diff'] = ((df_pred_LSTM['diff']) / (df_pred_LSTM['Close_pred']))*100
print(df_pred_LSTM)

#Final result summary
print('Model predicted for today close: {}, current price is: {}'.format(round(df_pred_LSTM['Close_pred'].iat[-1], 2), round(df_pred_LSTM['Close_actual'].iat[-1], 2)))