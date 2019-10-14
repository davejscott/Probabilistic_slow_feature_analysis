import numpy as np
import pandas as pd
from pandas import read_csv
from matplotlib import pyplot
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss

### load data in


df=pd.read_csv('5P14Clean.csv', sep=',')
df=df.dropna()
df.values
df.plot()
pyplot.show()

### set variables for ADF test

alpha = 0.05
lags = 1
normalized = 0

#series = read_csv('P14train.csv', header=0, index_col=0)
#X = series.values
def stationarity_test(timeseries):
    columns = list(timeseries)
    for ite in columns:
        result = adfuller(df[ite])
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
        if result[0] < 0.05:
            print('this data is stationary')
        else:
            print('this data is non-stationary')

stationarity_test(df)
        

cols_to_transform = ['Price','Km_From_CBD','Local_Median_Price', 'AreaSize']

for x in cols_to_transform:
    df[x]

    newcolname = ('T1_Pre_'+ x)
    df.newcolname = df[x] + 1 
    


data = df

def DSFA(data, qs, d):
    m = data.shape
    Z = m * d
    #for ite in range(d-1):
    #    df = df([[data[ite:([-1] - d + ite), :]]
    
    #A = [[0 for col in range(Z)] 0 for row in range(Z)]
    #B = [[0 for col in range(Z)] 0 for row in range(Z)]
    #meandf = np.mean(df, axis=0)
    #df = df - [[1 for col in range(df)] 1 for row in range(1)] * meandf
    #errordf = df()
    
         
 # ML a([1:end 1],:)

#python a[r_[:len(a),0]]

#a with copy of the first row appended to the end

d=1
q=2
DSFA(data,q,d)       
    
    

#... ADFTEST

#...
## Set system up for analysis

#iteNum = 75
#iterations = 2

#qstat=1
#d = 1

#qNS = 4
#d2 = 1

#for i in range:
#    q = qstat + qNS * d2
#    dataNS = mcat([NSdata])
#    dataS = mcat([Sdata])
#    data = mcat([dataNS, dataS])
#    numXn2 = mcat([numXn])
#    numXs2 = mcat([numXn])
    # build DSFA algorithm