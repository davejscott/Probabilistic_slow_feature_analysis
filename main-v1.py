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

print(train.head())
train['Frequency'].plot()

def adf_test(timeseries):
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag ='AIC')
    output = []
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic',
                         'p-value','#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical value (%s)'%key] = value
    for key, value in dftest[4].items():
        if value < dftest[0]:
            dfoutput['stationary(%s)'%key] = value
            output.append()
            return True
        elif value > dftest[0]:
            dfoutput['non-stationary(%s)'%key] = value
            return False
    print (dfoutput)

        
    
adf_test(train['Frequency'])


#df=pd.read_csv('5P14Clean.csv', sep=',')
#df=df.dropna()
#df.plot()
#pyplot.show()


#series = read_csv('P14train.csv', header=0, index_col=0)
#X = series.values

### set variables for ADF test

#alpha = 0.05
#lags = 1
#normalized = 0

#def check_for_stationarity(X, cutoff=0.01):
    # H_0 in adfuller is unit root exists (non-stationary)
    # We must observe significant p-value to convince ourselves that the series is stationary
#    pvalue = adfuller(X)[1]
#    if pvalue < cutoff:
#        print("p-value = " + str(pvalue) + ' The series is likely stationary.')
#        return True
#    else:
#        print('p-value = ' + str(pvalue) + ' The series is likely non-stationary.')
#        return False 


#def stationarity_test(df):
#    columns = []
#    for c in df.columns[1:]:
        #if (not df[c].isnull().all()) and df[c].var() !=0:
        #    columns.append(c)
#        df = df[columns]
#        print(df[c])
#        z = adfuller(df.c)
#        print('ADF Statistic: %f' % z)
#        if len(columns) == 0: return []
#        for i, col in enumerate(df.columns):
#            serie = df[col].dropna()
#            result = adfuller(serie)
#            print('ADF Statistic: %f' % result[0])
#            print('p-value: %f' % result[1])
#            print('Critical Values:')
#            for key, value in result[4].items():
#                print('\t%s: %.3f' % (key, value))
#                if result[0] < 0.05:
#                    print('this data is stationary')
#                else:
#                    print('this data is non-stationary')
            

            #for reg in p_values:
             #   v = adfuller(serie, regression=reg)[1]
             #   if math.isnan(v): #uncertain
             #       p_values[reg].append(-0.1)
              #  else:
              #      p_values[reg].append(v)
         #columns = list(timeseries)
         #  result = adfuller(print(columns[ite]))
       
      #print('ADF Statistic: %f' % result[0])
      #  print('p-value: %f' % result[1])
       #print('Critical Values:')
        #for key, value in result[4].items():
         #   print('\t%s: %.3f' % (key, value))
        #if result[0] < 0.05:
        #    print('this data is stationary')
        #else:
        #    print('this data is non-stationary')

#stationarity_test(df)
        
#def do_adfuller(path, srv, p_values):
#    filename = os.path.join(path, srv["filename"])
#    df = load_timeseries(filename, srv)
#    columns = []
#    for c in df.columns:
#        if (not df[c].isnull().all()) and df[c].var() != 0:
#            columns.append(c)
#    df = df[columns]
#    if len(columns) == 0: return []
#    for i, col in enumerate(df.columns):
#        serie = df[col].dropna()
#        if is_monotonic(serie):
#            serie = serie.diff()[1:]

#        for reg in p_values:
#            v = adfuller(serie, regression=reg)[1]
#            if math.isnan(v): # uncertain
#                p_values[reg].append(-0.1)
#            else:
#                p_values[reg].append(v)

#    return p_values 
    
#for column in df.columns[::-1]:
 #   print(df[column])
    

#data = df

#def DSFA(data, qs, d):
#    m = data.shape
#    Z = m * d
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

#d=1
#q=2
#DSFA(data,q,d)       
    
    

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