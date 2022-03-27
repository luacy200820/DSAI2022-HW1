# DSAI-HW-2022  

## Dataset  
1.台灣電力公司_過去電力供需資訊2021  
2.台灣電力公司_本年度每日尖峰備轉容量率  

## Method  
使用**SARIMA**做預測  
資料蒐集為台電2021年度加上今年度的備轉容量  

經過不同時段預測  
選擇距離3/29前120天當作訓練集    

將數據decompose，呈現下圖走勢，分別為原數據、Trend、Seasonality 和 Stationary:  
![](https://i.imgur.com/PHzMbGq.png)    

針對ACF(Autocorrelation Function)和PACF(Partial Autocorrelation Function)圖可幫助我們判斷模型SARIMA(p, d, q)參數的選擇  

![](https://i.imgur.com/KNyLsfo.png)    


經過多次測量參數，最後參數挑選為order = (3,1,0)、seasonal_order=(1,0,1,12)  

最後預測3/30~4/13的值，輸出在submission.csv  


