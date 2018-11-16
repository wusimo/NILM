from __future__ import division
import heapq
import math
import numpy as np
import scipy.stats
import scipy.signal
import scipy.linalg

def preprocess(data):
    if data[0]==0:
        data[0]=np.mean(data[1:10])
    
    for i in range(1,len(data)):
        ratio=abs(data[i]-data[i-1])/data[i-1]
        if data[i]==0 or ratio>=2:
            data[i]=data[i-1]
    return data

########################################################
'''Savitzky-Golay filter'''
def s_filter(data,wspan, polyorder):
    y_filtered=scipy.signal.savgol_filter(data, wspan, polyorder, mode='nearest')
    return y_filtered

########################################################
'''rlowess smooth
Robust version of local weighted regression, Insensitive to outliers'''
def rlowess(y, fraction, iter=3):
    n = len(y)
    x = np.linspace(0,100,n)
    r = int(math.ceil(n*fraction))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:,None] - x[None,:]) / h), 0.0, 1.0)
    w = (1 - w**3)**3
    y_est = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:,i]
            b = np.array([np.sum(weights*y), np.sum(weights*y*x)])
            A = np.array([[np.sum(weights), np.sum(weights*x)],
                   [np.sum(weights*x), np.sum(weights*x*x)]])
            if np.linalg.det(A) < 1.e-16:
                A = np.array([[1.e-16, np.sum(weights*x)],[np.sum(weights*x), 1.e-16]])
            beta = scipy.linalg.solve(A, b)
            y_est[i] = beta[0] + beta[1]*x[i]
    return y_est

########################################################
'''errors to rlowess smooth(for data_daily)'''
def errlist(value_day,tm_yday,tm_wday,i_wday,fraction):
    idx=[]
    err=[]
    for index, item in enumerate(tm_wday):
        if item==i_wday:
            idx.append(index)
    value_sub=np.array([value_day[i] for i in idx])
    yday_sub=np.array([tm_yday[i] for i in idx])

    value_sub_est=rlowess(value_sub, fraction)
    
    for i in range(len(value_sub)):
        error=value_sub[i]-value_sub_est[i]
        err.append(error)
    label=yday_sub
    # err : error list for each Mon-Sun
    # label: correspoding tm_yday index
    return (err,label)

########################################################
'''errors to Moving Average'''
def errMA(value_day,tm_yday,tm_wday,i_wday,MA):
    idx=[]
    err=[]
    for index, item in enumerate(tm_wday):
        if item==i_wday:
            idx.append(index)
    value_sub=[value_day[i] for i in idx]
    yday_sub=[tm_yday[i] for i in idx]
    '''for i in range(MA):
        error=value_sub[i]-sum(value_sub[0:i+1])/(i+1)
        err.append(error)'''
    for i in range(MA,len(value_sub)):
        error=value_sub[i]-sum(value_sub[(i-MA):i])/MA
        err.append(error)
    label=yday_sub[MA:]
    # err : error list for each Mon-Sun
    # label: correspoding tm_yday index
    return (err,label)

########################################################
'''Find outlier from zscore---large number of data points(min)'''
def z_outlier(err, thresh):
    if len(err) == 1:
        print 'Warning: only one data point'
    z_score=scipy.stats.mstats.zscore(err)
    return abs(z_score) > thresh

########################################################
'''Find outlier from LOF---small number of data points(Daily)'''
def dist(x1,x2): #Compute distance between any two point
    dist=abs(x1-x2)
    return dist

def Nkdist(test,train,k):
    distlist=[]
    for i in range(len(train)): #built distance list
        distlist.append(dist(test,train[i]))    
    index=[sorted(distlist).index(i) for i in distlist]
    index=index[:k]
    return index

def knnd(test,train,k):
    distlist=[]
    for item in train: #built distance list
        distlist.append(dist(test,item))    
    mindist=heapq.nsmallest(k,distlist)
    mindist.sort()
    Mind=mindist[-1]
    return Mind

def transpose(data):
    dataT=map(list, zip(*data))
    return dataT

def LOF(data,label,Minpts,nstd):
    N=len(data)
    LOFlist=[]
    for index, p in enumerate(data):
        test=p
        #p: test instance
        train=data[:index]+data[index+1:]
        Nindex_p=Nkdist(test,train,Minpts)
        #Nindex: Index of k-distance neighborhoods of p

        rdist=0
        #rdist: reachability distance of p
        for i in Nindex_p:
            test=data[i]
            train=data[:i]+data[i+1:]
            rdist+=max(knnd(test,train,Minpts),dist(p,test))
        LRD_p=Minpts/rdist
        #LRD_p: local reachability density of p

        LRD_q=0
        for idx in Nindex_p:
            test=data[idx]
            train=data[:idx]+data[idx+1:]
            Nindex_q=Nkdist(test,train,Minpts)

            rdist=0
            for i in Nindex_q:
                test2=data[i]
                train2=data[:i]+data[i+1:]
                rdist+=max(knnd(test2,train2,Minpts),dist(test,test2))
            LRD_q+=Minpts/rdist
        LOF=LRD_q/LRD_p/Minpts
        LOFlist.append([LOF,label[index]])

    LOFlist.sort(reverse=-1)
    LOF_t=transpose(LOFlist)
    mean=sum(LOF_t[0])/N
    std=math.sqrt(sum((x-mean)**2 for x in LOF_t[0])/(N-1))

    for j, x in enumerate(LOF_t[0]):
        if x<mean+std*nstd:#Threshold std*n
            break
    LOFlist=LOFlist[:j]
    return LOFlist
########################################################
