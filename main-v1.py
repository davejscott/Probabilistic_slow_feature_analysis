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
    #print(E.shape)
    for i in range(d):
        q = E.copy()
        Lambda.append(q[i,i])
    #print(Lambda)
    return Lambda, W
   # W = np.linalg.lstsq(u1,v) * W
    
   #ERRORS:
   # Double check why the Lambda value being extracted it one
   # Make the W area!
    
        
    
#adf_test(train['Frequency'])
# q, number of features
# d, number of delays
def dynamic_slow_feature_analysis(stat_timeseries, q, d, m):
     Z = np.array(stat_timeseries)
     l = np.array(Z.shape)
     print(l)
     X = []
     for ite in range(d):
         y = Z.copy()
         X.append(y[ite:-1+ite-d])
     X = np.array(X)
     ones = np.ones((X.shape), int)
     errorX = X[:,:-1] - X[:,1:]
     meanX = np.mean(X)
     X = X - ones * meanX
     errorX = X[:,:-1] - X[:,1:]
     A = np.dot(errorX, errorX.T) / (l - 1)
     B = np.dot(X, X.T) / l
     Lambda, W = slow_feature_analysis(A,B, d)
     ### Slow feature analysis time!
     ## Still some issues with SFA
     Feature = X * W
     R = np.linalg.inv(W).T
     C = R[-1 -m +1: -1,:-1-q + 1:-1]
     Lambdafeat = []
     Lambda = np.array(Lambda)
     t = Lambda.shape
     # FIX THIS SECTION FOR ISSUES WITH
     # The iterative portion of Lambda
     #for ite in range(d):
     #    E = Lambda.copy()
     #    Lambdafeat.append(E[:-1-d+ite])
     print(Lambdafeat)
     Reconstruction = Feature[:,-1-q+1:-1] * R[:,-1-q+1:-1].T
     ReconstructionError = X[:,:m] - Reconstruction[:,:m]
     Sigma = np.var(ReconstructionError)
     Lam = 1 - Lambda / 2
     Lambda = []
     p = 0.01
     for n in Lam:
         if n > 0.01:
             Lambda.append(n)
         else:
             Lambda.append(p)
     #    Lambda.append(max(1 - Lambdafeat(ite)/2)
     print(Lambda) 
     print(Sigma)
     print(R)
     print(C)
     return Sigma, Lambda, C
        # if Lambda < 0
         #    Lambda = 0.01
        
     


dynamic_slow_feature_analysis(train['Motor Temperature'], 5, 1, 1)    