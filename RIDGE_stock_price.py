#Importing the libraries
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import pandas_datareader as web
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy import stats

#Our picked company
company = 'AAPL'

#Starting and ending dates of the data which will be imported
start = dt.datetime(2000, 1, 1)
end = dt.datetime.now()

#Importing data
df = yf.download(company, start, end)

#Removing NaN values
df.dropna(axis=0, inplace=True)
df['Date'] = df.index

#Sorting data
data = df.sort_index(ascending=True, axis=0)


#Creating a separate dataset
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])

for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Adj Close'][i]

#We will create a number of features on the Dates
new_data['year'] = new_data['Date'].map(lambda x : x.year)
new_data['month'] = new_data['Date'].map(lambda x : x.month)
new_data['day_week'] = new_data['Date'].map(lambda x : x.dayofweek)
new_data['quarter'] = new_data['Date'].map(lambda x : x.quarter)
new_data['week'] = new_data['Date'].map(lambda x : x.week)
new_data['quarter_start'] = new_data['Date'].map(lambda x : x.is_quarter_start)
new_data['quarter_end'] = new_data['Date'].map(lambda x : x.is_quarter_end)
new_data['month_start'] = new_data['Date'].map(lambda x : x.is_month_start)
new_data['month_end'] = new_data['Date'].map(lambda x : x.is_month_end)
new_data['year_start'] = new_data['Date'].map(lambda x : x.is_year_start)
new_data['year_end'] = new_data['Date'].map(lambda x : x.is_year_end)
new_data['week_year'] = new_data['Date'].map(lambda x : x.weekofyear)
new_data['quarter_start'] = new_data['quarter_start'].map(lambda x: 0 if x is False else 1)
new_data['quarter_end'] = new_data['quarter_end'].map(lambda x: 0 if x is False else 1)
new_data['month_start'] = new_data['month_start'].map(lambda x: 0 if x is False else 1)
new_data['month_end'] = new_data['month_end'].map(lambda x: 0 if x is False else 1)
new_data['year_start'] = new_data['year_start'].map(lambda x: 0 if x is False else 1)
new_data['year_end'] = new_data['year_end'].map(lambda x: 0 if x is False else 1)
new_data['day_month'] = new_data['Date'].map(lambda x: x.daysinmonth)

#Creating a feature which could be important - Markets are only open between Monday and Friday.
mon_fri_list = [0,4]
new_data['mon_fri'] = new_data['day_week'].map(lambda x: 1 if x in mon_fri_list  else 0)

#Changing index in data
new_data.index = new_data['Date']
new_data.drop('Date', inplace=True, axis=1)

#Creating 'lags' which will define the auto-correlaton effect between past observations
for i in range(1, 22):
    new_data["lag_{}".format(i)] = new_data.Close.shift(i)

#Creating dummies for chosen features
cols = ['year', 'month', 'day_week', 'quarter', 'week', 
        'quarter_start', 'quarter_end', 'week_year', 'mon_fri', 'year_start', 'year_end',
       'month_start', 'month_end', 'day_month']
for i in cols:
    new_data = pd.concat([new_data.drop([i], axis=1), 
        pd.get_dummies(new_data[i], prefix=i)
    ], axis=1)

#Removing NaN values
new_data = new_data.dropna()
new_data = new_data.reset_index(drop=True)

#Creating splitting index
test_index = int(len(new_data) * (1 - 0.77))

#Splitting dataset on train and test
X_train = new_data.loc[:test_index-1].drop(['Close'], axis=1)
y_train = new_data.loc[:test_index-1]["Close"]
X_test = new_data.loc[test_index+1:].drop(["Close"], axis=1)
y_test = new_data.loc[test_index+1:]["Close"]  

#Scaling the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Creating Ridge model
ridge = Ridge(max_iter=10000, random_state=17)
ridge.fit(X_train_scaled, y_train)
y_pred = ridge.predict(X_test_scaled)

#Creating a data frame
columns = ['Close_actual', 'Close_pred']
df_pred_ridge = pd.DataFrame(columns = columns)
df_pred_ridge.Close_actual = y_test
df_pred_ridge.Close_pred = y_pred

#Visualizing the data
plt.figure(figsize=(16, 8))
plt.plot(df_pred_ridge.Close_pred, linewidth = 1.0)
plt.plot(df_pred_ridge.Close_actual, linewidth = 1.0)
plt.xlabel('Days', fontsize=18)
plt.ylabel('Price ($)', fontsize = 18)
plt.legend(['Actual',  'Predictions'], loc='lower right')
plt.title('Ridge model price prediction for {}'.format(company))
plt.show()

#Getting the root mean squared error (RMSE) which is the standard deviation of the prediction errors
print('Mean absolute sqaured error is: {}'.format(np.sqrt(np.mean(df_pred_ridge.Close_actual - df_pred_ridge.Close_pred)) ** 2))

#Creating the value which is a correlation between real price and our model predicted price
corr_price, p = stats.pearsonr(df_pred_ridge.Close_actual.values, df_pred_ridge.Close_pred.values)
print("Correlation between actual price and predicted by our model price is: {}".format(corr_price))

#Creating summarizing data frame
df_pred_ridge['diff'] = df_pred_ridge.Close_actual - df_pred_ridge.Close_pred
df_pred_ridge['perc_diff'] = ((df_pred_ridge['diff']) / (df_pred_ridge['Close_pred']))*100
print(df_pred_ridge)

#Final result summary
print('Model predicted for today close: {}, current price is: {}'.format(round(df_pred_ridge['Close_pred'].iat[-1], 2), round(df_pred_ridge['Close_actual'].iat[-1], 2)))