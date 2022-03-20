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

def predict(dataset):
    predictions =[]

    #forecast 
    for t in range(len(test)):
        someday = train[-1:].index.values[0]
        someday= pd.to_datetime(someday,format = '%Y-%m-%d %H:%M:%S')
        # calculating end date by adding 1 day
        Enddate = someday + timedelta(days=1)
        
        model = ARIMA(dataset,order=(2,1,3))
        model_fit = model.fit(disp=False)
        yhat = model_fit.forecast()[0]
        
        predictions.append(yhat)
        train.loc[Enddate] = [round(yhat[0],1)]

    return train,predictions
    

# You can write code above the if-main block.
if __name__ == '__main__':

    # # You should not modify this part, but additional arguments are allowed.
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--training',
    #                    default='training_data.csv',
    #                    help='input training data file name')

    # parser.add_argument('--output',
    #                     default='submission.csv',
    #                     help='output file name')
    # args = parser.parse_args()


    '''load data'''
    
    path_file = '台灣電力公司_過去電力供需資訊2021.csv'
    data = pd.read_csv(path_file)
    data = pd.DataFrame(data)
    data = data.loc[:,['日期','備轉容量(MW)']]
    data.columns=['date','capacity']

    data['date']= pd.to_datetime(data['date'])
    today_file =  '台灣電力公司_本年度每日尖峰備轉容量率.csv'
    today_data = pd.read_csv(today_file)
    today_data = pd.DataFrame(today_data)
    today_data = today_data.loc[:,['date','capacity']]
    today_data['date'] = pd.to_datetime(today_data['date'])

    '''change datetime format'''
    for index in range(len(data['date'])):
        index_time = str(data['date'][index]).split('.')[-1][1:]
        # print(index_time)
        time_change = datetime.strptime(index_time,"%Y%m%d")
        data['date'][index] = time_change
        data['capacity'][index] /= 10 

    '''concat two files'''
    data_cat = pd.concat([data[:365],today_data])

    '''split training set and testing set'''
    train = data_cat[:-15]
    test = data_cat[-15:]

    train.set_index('date',inplace=True)
    test.set_index('date',inplace=True)

    result,prediction = predict(train)
    rmse = sqrt(mean_squared_error(test,prediction))

    print('Test RMSE: %.3f'%rmse)

    result_predict = result[-15:]
    result_predict.index = result_predict.index.strftime('%Y%m%d')
    result_predict.index.name = 'date'
    result_predict.columns = ['operating_reserve(MW)']
    result_predict.to_csv( 'submission.csv')

