import pandas as pd
import statsmodels.api as sm
from datetime import datetime
import matplotlib.pyplot as plt 
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import warnings
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from datetime import timedelta
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
warnings.filterwarnings('ignore')
# from pandas.tseries.offsets import DateOffset

def predict(dataset):
    model=sm.tsa.statespace.SARIMAX(dataset['capacity'],order=(3,1,0),seasonal_order=(1,0,1,12))
    results=model.fit()

    '''generate date range'''
    future_dates = pd.date_range(start='20220329',end='20220413')
    future_datest_df=pd.DataFrame(index=future_dates[0:],columns=dataset.columns)

    future_datest_df['forecast'] = results.predict(start='20220329',end='20220413', dynamic= True)
    future_datest_df['forecast'] = round(future_datest_df['forecast'],0)

    return future_datest_df['forecast']
        

# You can write code above the if-main block.
if __name__ == '__main__':

    # # You should not modify this part, but additional arguments are allowed.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()


    '''load data'''
    data = pd.read_csv(args.training)
    
    data = pd.DataFrame(data)
    spring_data = data[-120:]
    train_spring = spring_data
    train_spring.set_index('date',inplace=True)


    '''show decompose image'''
    # decomp = seasonal_decompose(train_spring)
    # fig = decomp.plot()
    # fig.set_size_inches(15,8)
    # plt.show()

    '''show acf and pacf'''
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(train_spring['capacity'],lags=40,ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(train_spring['capacity'],lags=40,ax=ax2)

    '''predict'''
    result = predict(train_spring)

    '''Output'''
    result_predict = result[-15:]
    df = pd.DataFrame( result_predict)
 
    df.index = df.index.strftime('%Y%m%d')
    df.index.name = 'date'
    df.columns = ['operating_reserve(MW)']
    df.to_csv('submission.csv')

