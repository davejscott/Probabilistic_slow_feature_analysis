import numpy as np
### load data in

### set variables for ADF test

alpha = 0.05
lags = 1
normalized = 0

data = X

def DSFA(data=none, q=none, d=none):
    m = size(data, 2)
    for ite in range(0:d-1):
        X = mcat([X, data(mslice[ite:(end - d + ite)], mslice[:])
    A = [0 for col in range(m * d) 0 for row in range(m *d)]
    B = [0 for col in range(m * d) 0 for row in range(m *d)]
    meanX = np.mean(X, axis=0)
    
         
         
        
    
    

... ADFTEST

...
## Set system up for analysis

iteNum = 75
iterations = 2

qstat=1
d = 1

qNS = 4
d2 = 1

for i in range:
    q = qstat + qNS * d2
    dataNS = mcat([NSdata])
    dataS = mcat([Sdata])
    data = mcat([dataNS, dataS])
    numXn2 = mcat([numXn])
    numXs2 = mcat([numXn])
    # build DSFA algorithm
    