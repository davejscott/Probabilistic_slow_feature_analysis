import numpy as np
import pandas as pd
from pandas import read_csv
from matplotlib import pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss

### load data in

train = pd.read_csv('5P14clean.csv')
train.dropna()
train.index = train.Timestamp

#print(train.head())
train['Frequency'].plot()

def adf_test(timeseries):
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag ='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic',
                         'p-value','#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical value (%s)'%key] = value
    for key, value in dftest[4].items():
        if value < dftest[0]:
            dfoutput['stationary(%s)'%key] = value
            #return True
        elif value > dftest[0]:
            dfoutput['non-stationary(%s)'%key] = value
            #return False
    print (dfoutput)

def slow_feature_analysis(A, B, d):
    #A: temporal structure matrix
    #B: covariance matrix
    [u1, s1, v1] = np.linalg.svd(B, int)
    v = np.sqrt(v1)
    x,resid,rank,s = np.linalg.lstsq(v,u1.T)
    A1 = x * (A * u1/v)
    W, D, E = np.linalg.svd(A1)
    Lambda = []
    print(E.shape)
    for i in range(d):
        q = E.copy()
        Lambda.append(q[i,i])
    print(Lambda)
    return Lambda
   # W = np.linalg.lstsq(u1,v) * W
    
    
        
    
#adf_test(train['Frequency'])
# q, number of features
# d, number of delays
def dynamic_slow_feature_analysis(stat_timeseries, q, d):
     Z = np.array(stat_timeseries)
     l = np.array(Z.shape)
     X = []
     for ite in range(d):
         y = Z.copy()
         X.append(y[ite:-1+ite-d])
     #print(X)
     X = np.array(X)
     ones = np.ones((X.shape), int)
     errorX = X[:,:-1] - X[:,1:]
     meanX = np.mean(X)
     X = X - ones * meanX
     errorX = X[:,:-1] - X[:,1:]
     A = np.dot(errorX, errorX.T) / (l - 1)
     B = np.dot(X, X.T) / l
     slow_feature_analysis(A,B, d)
     ### Slow feature analysis time!


dynamic_slow_feature_analysis(train['Motor Temperature'], 5, 1)    
    

  