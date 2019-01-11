
# coding: utf-8

# # Table of Contents
# * <a href='#1'>1. make disaggregate algorithm faster</a>
# 
#     * <a href='#1.1'>1.1 previous bayesian algorithm wrote in the last week</a>
# 
#     * <a href='#1.2'>1.2 make code more robust</a>
# 
#     * <a href='#1.3'>1.3 make the code faster</a>
# 
# * <a href='#2'>2. Cluster and mode of power changes around change point</a>
# 
#     * <a href='#2.1'>2.1 Align power usage after change points, and normalization</a>
# 
#     * <a href='#2.2'>2.2 disaggregate data from a few days, and align</a>
# 
#     * <a href='#2.3'>2.3 Align all data from April 2016</a>
# 
#     * <a href='#2.4'>2.4 load more data, from 2016/3/22 ~ 2016/7/31</a>
# 
#     * <a href='#2.5'>2.5 Align all data from four months in 2016</a>
# 
#     * <a href='#2.6'>2.6 cluster</a>
# 
# * <a href='#3'>3. revisit disaggregation algorithm using known patterns</a>
# 
#     * <a href='#3.1'>3.1 basic algorithm that utilize the shapes of each cluster</a>

# In[1]:

# load a few necessary packages

get_ipython().magic(u'matplotlib inline')

import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import time
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_context("poster")


# <a id='1'></a>
# # 1. make disaggregate algorithm faster

# <a id='1.1'></a>
# ## 1.1 previous bayesian algorithm wrote in the last week

# In[79]:

# prepare example data
from bayesian_cp_detect import file_readef

def load_dat(date, Hstart, Hend, folder='data/IHG/'):
    (t,y)=np.array(file_readef.readfile(date,Hstart,Hend,folder))
    return t, y


# In[11]:

# example data, should be easy to disaggregate
# plot the example data

t_raw_4_16_19_23, y_raw_4_16_19_23 = load_dat('4-16', 19, 23, 'data/IHG/')

plt.figure(figsize=[18,3])
plt.plot(t_raw_4_16_19_23, y_raw_4_16_19_23, 'k.-')
plt.xlabel('t (15s sampling rate)')


# In[13]:

from bayesian_cp_detect import cp_detect
from bayesian_cp_detect import outlierdef


# In[20]:

get_ipython().run_cell_magic(u'time', u'', u"\ny_raw = y_raw_4_16_19_23\ny = outlierdef.preprocess(y_raw)\n\nprint 'size of input: %d' % len(y)\n[mu_list_list, sigma_list_list, prob_r_list_list] = cp_detect.bayesian_change_point(y)\nchangepoint, changepoint_p = cp_detect.get_change_point(prob_r_list_list)")


# In[30]:

# write a function overlay change points to raw data
def plot_with_cp(y, cp_list=[]):
    plt.figure(figsize=(18,3))
    plt.plot(y, 'k-', linewidth=2)
    for cp in cp_list:
        plt.plot([cp,cp], [0, 500], 'k--', linewidth=1)
    plt.xlabel('t')
    plt.ylabel('power')


# In[28]:

plot_with_cp(y, changepoint)


# <a id='1.2'></a>
# ## 1.2 make code more robust

# In[32]:

# load a example of "difficult" data
t_raw_4_3_7_18, y_raw_4_3_7_18 = load_dat('4-3', 7, 18, 'data/IHG/')
plot_with_cp(y_raw_4_3_7_18)


# In[33]:

get_ipython().run_cell_magic(u'time', u'', u"\n# segment using previous version\n\ny_raw = y_raw_4_3_7_18\ny = outlierdef.preprocess(y_raw)\n\nprint 'size of input: %d' % len(y)\n[mu_list_list, sigma_list_list, prob_r_list_list] = cp_detect.bayesian_change_point(y)\nchangepoint, changepoint_p = cp_detect.get_change_point(prob_r_list_list)")


# In[34]:

plot_with_cp(y, changepoint)


# #### Note that there are 2 problems
# 
# * it is slower than before, takes ~ 5 seconds
# * disaggregate ~2200 is bad - a change point was detected due to bad data
# 
# 

# In[35]:

plot_with_cp(y, changepoint)
plt.xlim([2200, 2300])


# To solve the bad point problem, a filter based on relative change is introduced. The idea is that the relative to the closest point (either left or right, whichever closer) should not be too large. To not include points during rapid change, this method only filter points that are either higher or lower than both near points

# In[36]:

def rel_change(y):
    return np.min([np.abs(y[1] - y[0]), np.abs(y[1] - y[2])]) / float(y[1])

def rel_change_filter(t, data_input, thre=.2):
    id_filter = [i for i in range(1, len(data_input)-1) 
     if (data_input[i]>data_input[i-1] and data_input[i]>data_input[i+1] and rel_change(data_input[i-1:i+2])>thre) or
                 (data_input[i]<data_input[i-1] and data_input[i]<data_input[i+1] and rel_change(data_input[i-1:i+2])>thre/(1-thre))
    ]
    id_filter2 = np.setdiff1d(range(len(data_input)), id_filter)
    t_2 = [t[i] for i in id_filter2]
    data_input_2 = [data_input[i] for i in id_filter2]
    return t_2, data_input_2


# In[38]:

get_ipython().run_cell_magic(u'time', u'', u"\n# segment using previous version\nt_raw = t_raw_4_3_7_18\ny_raw = y_raw_4_3_7_18\nt_2, y_2 = rel_change_filter(t_raw, y_raw,thre=.2)\n\nprint 'size of input: %d' % len(y)\n[mu_list_list, sigma_list_list, prob_r_list_list] = cp_detect.bayesian_change_point(y_2)\nchangepoint, changepoint_p = cp_detect.get_change_point(prob_r_list_list)")


# In[40]:

plot_with_cp(y_2, changepoint)
plt.xlim([2200, 2300])


# <a id='1.3'></a>
# ## 1.3 make the code faster

# ## 1.3.1 the problem
# 
# We will look at the "good data" again, to see can we make it even faster, and then apply to a more difficult situation

# In[53]:

get_ipython().run_cell_magic(u'time', u'', u"\nt_raw = t_raw_4_16_19_23\ny_raw = y_raw_4_16_19_23\nt_2, y_2 = rel_change_filter(t_raw, y_raw,thre=.2)\n\nprint 'size of input: %d' % len(y)\n[mu_list_list, sigma_list_list, prob_r_list_list] = cp_detect.bayesian_change_point(y_2)\nchangepoint, changepoint_p = cp_detect.get_change_point(prob_r_list_list)")


# In[54]:

plot_with_cp(y_2, changepoint)


# #### the problem is during the region that there is not too much changes

# In[55]:

plt.plot([len(x) for x in prob_r_list_list],'k-')
plt.xlabel('t')
plt.ylabel('calculation amount')


# ## 1.3.2 filtering probability
# 
# **the code below uses a new version of bayesian_change_point**
# 
# bayesian_change_point_3 did one step of filtering of list of r (r is the distance to the last change point). If the prob of certain r is lower than a threshold, it is filtered
# 
# ** note that the speed using bayesian_change_point_3 is faster then bayesian_change_point **

# In[49]:

get_ipython().run_cell_magic(u'time', u'', u"\nt_raw = t_raw_4_16_19_23\ny_raw = y_raw_4_16_19_23\nt_2, y_2 = rel_change_filter(t_raw, y_raw,thre=.2)\n\nprint 'size of input: %d' % len(y)\n[mu_list_list, sigma_list_list, prob_r_list_list, r_list_list] = cp_detect.bayesian_change_point_3(y_2)\nchangepoint, changepoint_p = cp_detect.get_change_point(prob_r_list_list)")


# In[50]:

plot_with_cp(y_2, changepoint)


# In[52]:

plt.plot([len(x) for x in prob_r_list_list],'k-')
plt.xlabel('t')
plt.ylabel('calculation amount')


# In[57]:

get_ipython().run_cell_magic(u'time', u'', u'# try a more "difficult" data\n\nt_raw = t_raw_4_3_7_18\ny_raw = y_raw_4_3_7_18\nt_2, y_2 = rel_change_filter(t_raw, y_raw,thre=.2)\n\nprint \'size of input: %d\' % len(y)\n[mu_list_list, sigma_list_list, prob_r_list_list, r_list_list] = cp_detect.bayesian_change_point_3(y_2)\nchangepoint, changepoint_p = cp_detect.get_change_point(prob_r_list_list)')


# In[58]:

plot_with_cp(y_2, changepoint)


# In[59]:

plt.plot([len(x) for x in prob_r_list_list],'k-')
plt.xlabel('t')
plt.ylabel('calculation amount')


# ## 1.3.3 More filtering
# 
# in bayesian_change_point_4, we will also filter r if r is larger than a certain threshold and is not close to the largest r. This is the situation where power does not really change much.
# 
# In the example below, the performance decreases, but not much. This is due to the extra calculation done. But I expect the performance is much better for even worse data

# In[60]:

get_ipython().run_cell_magic(u'time', u'', u'# try a more "difficult" data\n\nt_raw = t_raw_4_3_7_18\ny_raw = y_raw_4_3_7_18\nt_2, y_2 = rel_change_filter(t_raw, y_raw,thre=.2)\n\nprint \'size of input: %d\' % len(y)\n[mu_list_list, sigma_list_list, prob_r_list_list, r_list_list] = cp_detect.bayesian_change_point_4(y_2)\nchangepoint, changepoint_p = cp_detect.get_change_point(prob_r_list_list)')


# In[61]:

plot_with_cp(y_2, changepoint)


# In[62]:

plt.plot([len(x) for x in prob_r_list_list],'k-')
plt.xlabel('t')
plt.ylabel('calculation amount')


# In[65]:

# load even worst data
t_raw_4_4_7_18, y_raw_4_4_7_18 = load_dat('4-4', 7, 18, 'data/IHG/')


# In[69]:

get_ipython().run_cell_magic(u'time', u'', u"\nt_raw = t_raw_4_4_7_18\ny_raw = y_raw_4_4_7_18\nt_2, y_2 = rel_change_filter(t_raw, y_raw,thre=.2)\n\nprint 'size of input: %d' % len(y)\n[mu_list_list, sigma_list_list, prob_r_list_list, r_list_list] = cp_detect.bayesian_change_point_3(y_2)\nchangepoint, changepoint_p = cp_detect.get_change_point(prob_r_list_list)")


# In[70]:

plot_with_cp(y_2, changepoint)


# In[71]:

plt.plot([len(x) for x in prob_r_list_list],'k-')
plt.xlabel('t')
plt.ylabel('calculation amount')


# In[66]:

get_ipython().run_cell_magic(u'time', u'', u"\nt_raw = t_raw_4_4_7_18\ny_raw = y_raw_4_4_7_18\nt_2, y_2 = rel_change_filter(t_raw, y_raw,thre=.2)\n\nprint 'size of input: %d' % len(y)\n[mu_list_list, sigma_list_list, prob_r_list_list, r_list_list] = cp_detect.bayesian_change_point_4(y_2)\nchangepoint, changepoint_p = cp_detect.get_change_point(prob_r_list_list)")


# In[67]:

plot_with_cp(y_2, changepoint)


# In[68]:

plt.plot([len(x) for x in prob_r_list_list],'k-')
plt.xlabel('t')
plt.ylabel('calculation amount')


# ## 1.3.4 summary
# 
# Nothing new. Just going to run on good data again just to summarize where we are in terms of speed and performance

# In[72]:

get_ipython().run_cell_magic(u'time', u'', u"\nt_raw = t_raw_4_16_19_23\ny_raw = y_raw_4_16_19_23\nt_2, y_2 = rel_change_filter(t_raw, y_raw,thre=.2)\n\nprint 'size of input: %d' % len(y)\n[mu_list_list, sigma_list_list, prob_r_list_list, r_list_list] = cp_detect.bayesian_change_point_4(y_2)\nchangepoint, changepoint_p = cp_detect.get_change_point(prob_r_list_list)")


# In[73]:

plot_with_cp(y_2, changepoint)


# In[74]:

plt.plot([len(x) for x in prob_r_list_list],'k-')
plt.xlabel('t')
plt.ylabel('calculation amount')


# In[78]:

plt.plot(r_list_list[-1], prob_r_list_list[-1], 'ko-')
plt.xlabel('distance to the last changepoint')
plt.ylabel('prob, log scale')
plt.title('prob. of distance to the last point at the last data point')


# <a id='2'></a>
# # 2. Cluster and mode of power changes around change point

# <a id='2.1'></a>
# ## 2.1 Align power usage after change points, and normalization

# two helper functions: filter_dat will filter data, disaggregate will align all change points

# In[85]:

def filter_dat(t, y):
    t_2, y_2 = rel_change_filter(t, y, thre=.2)
    return t_2, y_2

def disaggregate(dat):
    mu_list_list, sigma_list_list, prob_r_list_list, r_list_list = cp_detect.bayesian_change_point_4(dat, r_blur=30)
    changepoint, changepoint_p = cp_detect.get_change_point(prob_r_list_list)
    dat_seg = [dat[cp_1:cp_2] for cp_1, cp_2 in zip(changepoint[:-1], changepoint[1:])]
    return dat_seg, changepoint


# In[111]:

get_ipython().run_cell_magic(u'time', u'', u"\nt_raw = t_raw_4_16_19_23\ny_raw = y_raw_4_16_19_23\nt_2, y_2 = filter_dat(t_raw, y_raw)\ndat_seg, changepoint = disaggregate(y_2)\n# print 'size of input: %d' % len(y)\n# [mu_list_list, sigma_list_list, prob_r_list_list, r_list_list] = cp_detect.bayesian_change_point_4(y_2)\n# changepoint, changepoint_p = cp_detect.get_change_point(prob_r_list_list)")


# In[112]:

print 'the number of seg:', len(dat_seg)
for seg in dat_seg:
    plt.plot(seg, 'k-')


# **let us avoid the long non-changing segment by focusing on the first 100 points**

# In[113]:

print 'the number of seg:', len(dat_seg)
for seg in dat_seg:
    plt.plot(seg, 'k-')
plt.xlim([0,100])


# The function below will take the aligned segments, and use the last data point from the last segment to calculate the 
# change in power usage

# In[114]:

def normalize_by_last_point(dat_seg):
    last_p = 0
    new_dat_seg = []
    for seg in dat_seg:
        new_dat_seg.append([x - last_p for x in seg])
        last_p = seg[-1]
    new_dat_seg.pop(0)
    return new_dat_seg


# In[115]:

dat_seg_new = normalize_by_last_point(dat_seg)
print 'the number of seg:', len(dat_seg_new)
for seg in dat_seg_new:
    plt.plot(seg, 'k-')
plt.xlim([0,100])


# **maybe the last point includes too much noise, the function below normalize by the last n points**
# 
# Although it does not look very different, the code below will use this function by default

# In[116]:

def normalize_by_last_n_point(dat_seg, n=3):
    last_p = 0
    new_dat_seg = []
    for seg in dat_seg:
        new_dat_seg.append([x - last_p for x in seg])
        last_p = np.mean(seg[-n:])
    new_dat_seg.pop(0)
    return new_dat_seg


# In[117]:

dat_seg_new = normalize_by_last_n_point(dat_seg)
print 'the number of seg:', len(dat_seg_new)
for seg in dat_seg_new:
    plt.plot(seg, 'k-')
plt.xlim([0,100])


# <a id='2.2'></a>
# ## 2.2 disaggregate data from a few days, and align

# wrote a function that takes date, start and end time. plot the aligned the results
# 
# ** importantly, note the similarity between different days**

# In[149]:

def align_cp_plot_2(date, Hstart, Hend):
    t, y = load_dat(date, Hstart, Hend)
    t, y = filter_dat(t, y)
    [dat_seg, changepoint] = disaggregate(y)
    dat_seg_new = normalize_by_last_n_point(dat_seg)
    for seg in dat_seg_new:
        plt.plot(seg, 'k-')
    plt.xlim([0, 100])
    plt.xlabel('time after change point')
    plt.ylabel('power change since the last change point')
    plt.title(date+' ('+str(Hstart)+'-'+str(Hend)+'h)')
#     plt.title(date, Hstart, Hend)


# In[145]:

get_ipython().run_cell_magic(u'time', u'', u"align_cp_plot_2('4-16', 19, 23) # good data again")


# In[146]:

get_ipython().run_cell_magic(u'time', u'', u"align_cp_plot_2('4-1', 7, 18)")


# In[147]:

get_ipython().run_cell_magic(u'time', u'', u"align_cp_plot_2('4-16', 7, 18)")


# In[148]:

get_ipython().run_cell_magic(u'time', u'', u"align_cp_plot_2('4-3', 7, 18)")


# <a id='2.3'></a>
# ## 2.3 Align all data from April 2016

# There are two variables used to record disaggregated signals, all_seg is list, each element is all segments within the same day; all_seg_2 is the flatten version of all_seg

# In[186]:

def dissaggregate_align_pool(date_start, date_end, Hstart, Hend):

    date_current = date_start

    all_seg = []
    all_seg_2 = []

    while date_current < date_end:
        date = str(date_current.month) + '-' + str(date_current.day)
        print 'reading', date, Hstart, Hend
        
        t, y = load_dat(date, Hstart, Hend)
        t, y = filter_dat(t, y)
        [dat_seg, changepoint] = disaggregate(y)
        dat_seg_new = normalize_by_last_n_point(dat_seg)
        
        all_seg.append(dat_seg_new)
        all_seg_2.extend(dat_seg_new)
        
        date_current += datetime.timedelta(1)
    return all_seg, all_seg_2


# In[187]:

import datetime


# In[188]:

date_start = datetime.date(2016,4,1)
date_end = datetime.date(2016,5,1)

Hstart=7
Hend=18

all_seg_april_last, all_seg_2_april_last = dissaggregate_align_pool(date_start
                                                                   , date_end
                                                                   , Hstart
                                                                   , Hend)


# ** Plot all segments in April **

# In[189]:

for seg in all_seg_2_april_last:
    plt.plot(seg, 'k-', linewidth=1, alpha=.1)
plt.plot([0,100], [0,0], 'k-', linewidth=1)
plt.xlim([0,100])
plt.xlabel('t (sampling rate = 15s)')
plt.ylabel('power change after change point')
plt.title('April, 2016')


# <a id='2.4'></a>
# ## 2.4 load more data, from 2016/3/22 ~ 2016/7/31

# In[167]:

# run downloader.py
# python downloader.py > downloader.py.log


# ** make sure that data looks just as before **
# 
# *Note that somehow data is shifted......*

# In[183]:

from os import path
def read_dat(date, h_start, h_end, folder_path):
    dat = []
    for h in range(h_start, h_end):
        try:
            file_name = '%d-%d-%d.csv' % (date.month, date.day, h)
            file_path = path.join(folder_path, file_name)
            tmp = pd.read_csv( file_path )
            dat.extend( [t[1]/3 for t in tmp.values] )
        except:
            print '%s failed' % file_path
    return dat


# In[184]:

dat = read_dat(datetime.datetime(2016, 4, 16), 19, 23, folder_path='new_data/IHG')
y = range(len(dat))

plot_with_cp(dat)


# In[185]:

plot_with_cp(y_raw_4_16_19_23)


# <a id='2.5'></a>
# ## 2.5 Align all data from four months in 2016

# In[201]:

def dissaggregate_align_pool(date_start, date_end, Hstart, Hend):

    date_current = date_start

    all_seg = []
    all_seg_2 = []
    all_seg_time = []

    while date_current < date_end:
        date = str(date_current.month) + '-' + str(date_current.day)
        print 'reading', date, Hstart, Hend
        dat = read_dat(date_current, Hstart, Hend, folder_path='new_data/IHG')
        t = range(len(dat)) # fake time
        try:
            t, y = filter_dat(t, dat)
            [dat_seg, changepoint] = disaggregate(y)
            dat_seg_new = normalize_by_last_n_point(dat_seg)
        except:
            dat_seg_new = []

#         [dat_seg, changepoint] = disaggregate(y)
#         dat_seg_new = normalize_by_last_point(dat_seg)
        
        all_seg.append(dat_seg_new)
        all_seg_2.extend(dat_seg_new)
        all_seg_time.append((date_current, Hstart, Hend))

        date_current += datetime.timedelta(1)
    return all_seg, all_seg_2, all_seg_time


# In[202]:

date_start = datetime.date(2016,4,1)
date_end = datetime.date(2016,8,1)

Hstart=7
Hend=18

all_seg_4_month, all_seg_2_4_month, all_seg_time_4_month = dissaggregate_align_pool(date_start
                                                                                    , date_end
                                                                                    , Hstart
                                                                                    , Hend)


# ** let us make a copy of current data, as it takes a long time to load **

# In[220]:

import json

all_seg_time_4_month_compress = [ (x[0].year, x[0].month, x[0].day) for x in all_seg_time_4_month]

with open('metadata/four month data.json', 'w') as fid:
    json.dump([all_seg_4_month, all_seg_2_4_month, all_seg_time_4_month_compress], fid)


# ** plotting!!! **

# In[221]:

for seg in all_seg_2_4_month:
    plt.plot(seg, 'k-', linewidth=1, alpha=.1)
plt.plot([0,100], [0,0], 'k-', linewidth=1)
plt.xlim([0,100])
plt.xlabel('t (sampling rate = 15s)')
plt.ylabel('power change after change point')
plt.title('April, 2016')


# ** even smaller alpha (transparency) **

# In[222]:

for seg in all_seg_2_4_month:
    plt.plot(seg, 'k-', linewidth=1, alpha=.01)
plt.plot([0,100], [0,0], 'k-', linewidth=1)
plt.xlim([0,100])
plt.xlabel('t (sampling rate = 15s)')
plt.ylabel('power change after change point')
plt.title('April, 2016')


# ** plot by month **
# 
# Note the vast difference between different months.
# Reason? season effect? or reading problems? or initial disaggregate problem?
# 
# ** important todo **
# 
# check out disaggregate in June and July

# In[241]:

_, ax = plt.subplots(2,2,figsize=[12,8])
ax = ax.flatten()

for i_seg_list, seg_list in enumerate(all_seg_4_month):
    cur_ax = ax[all_seg_time_4_month[i_seg_list][0].month-4]
    for seg in seg_list:
        cur_ax.plot(seg, 'k-', linewidth=1, alpha=.1)

for i in range(4):
    ax[i].set_xlim([0, 100])
    ax[i].set_ylim([-300, 400])
    ax[i].grid()
    if i == 0:
        ax[i].set_title('April')
    elif i == 1:
        ax[i].set_title('May')
    elif i == 2:
        ax[i].set_title('June')
    elif i == 3:
        ax[i].set_title('July')


# <a id='2.6'></a>
# ## 2.6 cluster

# In[280]:

# define color
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_context("poster")
    sns.palplot(sns.color_palette("hls", 8))
    color_list = sns.color_palette("hls", 8)
except:
    color_list = ['k', 'r', 'b', 'g', 'y', 'm', 'c', 'k', 'r']


# In[263]:

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


# In[264]:

n_clusters=6

X = [t[:50] for t in all_seg_2_april_last if len(t)>50]
y_pred = KMeans(n_clusters=6).fit_predict(X)


# In[290]:

# collect all clusters
cluster_mean = []
cluster_std = []
n_cluster = max(y_pred)+1
for i_cluster in range(n_clusters):
    seg_list = [x for i,x in enumerate(X) if y_pred[i] == i_cluster]
    cluster_mean.append(np.mean(seg_list, axis=0))
    cluster_std.append(np.std(seg_list, axis=0))


# In[307]:

cluster_rank = np.argsort([np.mean(t) for t in cluster_mean])[::-1]

rank_old_to_new = {t:i for i, t in enumerate(cluster_rank)}
print rank_old_to_new

y_pred_sort = [rank_old_to_new[x] for x in y_pred]
cluster_mean_sort = [cluster_mean[x] for x in cluster_rank]
cluster_std_sort = [cluster_std[x] for x in cluster_rank]


# In[316]:

fig, ax = plt.subplots(nrows=2,ncols=3,figsize=[16,8]);
ax = ax.flatten()

for i_cluster in range(n_clusters):
    ax_current = ax[i_cluster]
    
    seg_list = [x for i,x in enumerate(X) if y_pred_sort[i] == i_cluster]
    for seg in seg_list:
        ax_current.plot(seg, '-', linewidth=1, alpha=.3, color=color_list[i_cluster])
        ax_current.set_xlim([0, 50])
        ax_current.set_ylim([-200, 400])
        ax_current.plot([0,50], [0,0], 'k--')
    
    ax_current.plot(cluster_mean_sort[i_cluster], color=color_list[i_cluster])


# In[ ]:

import json

with open('metadata/april data.json', 'w') as fid:
    json.dump([all_seg_april_last
               , all_seg_2_april_last
               , [list(x) for x in cluster_mean_sort]
               , [list(x) for x in cluster_std_sort]
              ], fid)


# <a id='3'></a>
# # 3. revisit disaggregation algorithm using known patterns

# <a id='3.1'></a>
# ## 3.1 basic algorithm that utilize the shapes of each cluster

# In[364]:

def extract_first_n(data_list, n):
    return [t[:n] for t in data_list if len(t)>=n]

def evolve_prob(prior_prob, r_list, retain_prob):
    new_prob = []
    for t in prior_prob:
        new_prob.append([retain_prob[r]+p for p, r in zip(t, r_list)])
    return new_prob

# print evolve_prob([[-1, -2],[-3, -4]], [2,4], [1,2,3,4,5])

def sum_prob(prob):
    return sp.misc.logsumexp(prob)
# print sum_prob([[1,2], [3,4]]), sum_prob([1,2,3,4])
# print sum_prob([1,2]), np.log(np.sum(np.exp(1)+np.exp(2)))

def add_to_all(list_of_list, val):
    res = []
    for l in list_of_list:
        res.append([x+val for x in l])
    return res
# print add_to_all([[1,2],[3,4,5]],2)

def add_to_front(new_front, list_of_list):
    return [[new_front]+t for t in list_of_list]
# print add_to_front(1, [[2],[3,4],[]])

# np.argmax(prior_list), prior_list, r_list
# a = [[1,2,33],[4,5,12]]
# np.argmax(a)
def get_max_id(list_of_list):
    n_list = len(list_of_list)
    list_size = len(list_of_list[0])
    t = np.argmax(list_of_list)+1
    i_list = np.ceil(t / float(list_size)) - 1
    pos_in_list = t - i_list * list_size - 1
#     print 'n_list', n_list
#     print 'list_size', list_size
#     print 't', t
#     print 'i_list', i_list
#     print 'pos_in_list', pos_in_list
    return int(i_list), int(pos_in_list)
# get_max_id(a)

def select_from_list(list_of_list, id):
    result = []
    for l in list_of_list:
        result.append([l[i] for i in id])
    return result
# print select_from_list([[1,2,3], [3,4,5], [2,3,4,6]], [0,2])

def create_counter(mem=100):
    return {'mem':mem, 'dat':[0 for i in range(mem)],'start':0}

def vote(obj, counter, pos, p):
    while obj['start'] < counter - obj['mem'] + 1:
        obj['dat'].pop(0)
        obj['dat'].append(0)
        obj['start'] += 1
    if pos-obj['start']>=0:
        obj['dat'][pos-obj['start']]+=p
    return obj
    
def search_call(obj, thre):
    return [pos+obj['start'] for pos, p in enumerate(obj['dat']) if p > thre]

# a = create_counter()
# print a
# a = vote(a, 1, 1, 2)
# print search_call(a, 5)
# print a

# a = vote(a, 100, 1, 2)
# print search_call(a, 5)
# print a

# a = vote(a, 100, 1, 2)
# print search_call(a, 5)
# print a

# a = vote(a, 101, 13, 20)
# print search_call(a, 5)
# print a


# In[366]:

def update_prob(datum, prior_prob, r_list, mu_list, sigma_list, shapes, mu_prior, sigma_measurement):
    n_shape = len(shapes) # the number of possible change points
    n_r = len(r_list) # the number of r(s), r is the distance to the last change point
    shape_len = len(shapes[0])
    flag_print = False
    if flag_print:
        print 'shape_len', shape_len
        print 'datum', datum
        print 'mu_prior (last point)', mu_prior
#     if len(r_list) != n_shape:
#         raise('the number of r list does not match to the number of shapes')
    if len(prior_prob) != n_shape:
        raise('the number of prior prob does not match to the number of shapes')
    for t in prior_prob:
        if len(t) != n_r:
            print len(t), n_r
            raise('number of r does not match to probs')
    for t in mu_list:
        if len(t) != n_r:
            raise('number of r does not match to mu')
    for t in sigma_list:
        if len(t) != n_r:
            raise('number of r does not match to sigma')
    
    TOL = .9999
    
    gap_prior = 100.
    min_length_prior = 5
    STORAGE_MAX = 10000 # at a cost of mem, make a look up table for log H and log 1-H
    log_H_list = [np.log(1-1/(gap_prior*100))] * min_length_prior + [np.log(1-1/gap_prior)]*(STORAGE_MAX-min_length_prior) # hazard function, log(1-H)
    log_H_2_list = [np.log(1/(gap_prior*100))] * min_length_prior + [np.log(1/gap_prior)]*(STORAGE_MAX-min_length_prior) # log(H)
    
    if flag_print:
        print 'prior_prob', prior_prob
        print 'r_list', r_list
        print 'mu_list', mu_list
        print 'sigma_list', sigma_list

    # step 1, calculate the new probabilty of prior_prob, considering part of 
    # the possibility will go to new change point
    prior_prob_plus_1 = evolve_prob(prior_prob, r_list, log_H_list)
    prob_change = np.log((1-np.exp(sum_prob(prior_prob_plus_1)))/n_shape)
    prob_update = [[prob_change]+t for t in prior_prob_plus_1]
    if flag_print:
        print 'step 1'
        print prob_update, sum_prob(prob_update)
    
    # step 2: update r_list
    r_list_update = [0] + [t+1 for t in r_list]
    if flag_print:
        print 'step 2'
        print 'r_list_update', r_list_update
    
    # step 3: update u and sigma
    mu_prior = mu_prior
    sigma_prior = 50
    mu_list_update = add_to_front(mu_prior, mu_list)
    sigma_list_update = add_to_front(sigma_prior, sigma_list)
    if flag_print:
        print 'step 3'
        print 'mu_list_update', mu_list_update
        print 'sigma_list_update', sigma_list_update
    
    # step 4: predict prob
    mu_list_post = []
    sigma_list_post = []
    prob_list_post = []
    for i_shape in range(n_shape):
        mu_list_post_tmp = []
        sigma_list_post_tmp = []
        prob_list_post_tmp = []
        for i_r in range(n_r+1): # because everything shifted by 1
            r = r_list_update[i_r]
            mu = mu_list_update[i_shape][i_r]
            sigma = sigma_list_update[i_shape][i_r]
            if r < shape_len:
                shape_value = shapes[i_shape][r]
            else:
                shape_value = shapes[i_shape][-1]
            mu_with_shift = mu + shape_value
            # update sigma and mu, note that mu is considered as shift
            mu_update_with_shift = (mu_with_shift*sigma_measurement**2+datum*sigma**2)/(sigma_measurement**2+sigma**2)
            mu_update = mu_update_with_shift-shape_value
            sigma_update = np.sqrt(sigma_measurement**2*sigma**2/(sigma_measurement**2+sigma**2))
            
            prob = prob_update[i_shape][i_r]
            predict_prob = -((datum-mu_with_shift)/sigma_measurement)**2/2.0-np.log(sigma_measurement) 
            prob_post = prob + predict_prob
            
            if flag_print:
                print i_shape, i_r, 'r:', r, 'mu', mu, 'sigma', sigma, 'mu_with_shift', mu_with_shift
                print 'datum', datum, 
                print 'mu_update_with_shift', mu_update_with_shift, 'mu_update', mu_update, 'sigma_update', sigma_update
                print 'prob', prob, 'predict_prob', predict_prob, 'prob_post', prob_post
            mu_list_post_tmp.append(mu_update)
            sigma_list_post_tmp.append(sigma_update)
            prob_list_post_tmp.append(prob_post)
        mu_list_post.append(mu_list_post_tmp)
        sigma_list_post.append(sigma_list_post_tmp)
        prob_list_post.append(prob_list_post_tmp)
    
    # truncation
    t = sum_prob(prob_list_post)
    prob_list_post = add_to_all(prob_list_post, -t)

    # test if truncation is possible
    cum_pro = np.cumsum( np.sum(np.exp(prob_list_post),axis=0) )
    i_r_max = np.min([i for i, pro in enumerate(cum_pro) if pro > TOL]);
    if flag_print:
        print 'current r_max', r_list_update[i_r_max]
    if i_r_max<10:
        if flag_print:
            print 'i_r_max too small, do not truncate'
        i_r_max = len(cum_pro)-1

    if flag_print:
        print 'cum_pro', cum_pro
        print 'mu_list_post', mu_list_post
        print 'sigma_list_post', sigma_list_post
        print 'prob_list_post', prob_list_post, sum_prob(prob_list_post)
        print 'r_list_update', r_list_update

    mu_list_post = extract_first_n(mu_list_post, i_r_max+1)
    sigma_list_post = extract_first_n(sigma_list_post, i_r_max+1)
    prob_list_post = extract_first_n(prob_list_post, i_r_max+1)
    r_list_update = [t for i, t in enumerate(r_list_update) if i <=i_r_max]
    if flag_print:
        print 'total r(s)', len(cum_pro), 'truncation', i_r_max
        
    # a second round of truncation for flat signals, truncate from the end
    if len(r_list_update) > 30:
        r_max = np.max(r_list_update)
        valid_r = [i for i,t in enumerate(r_list_update) if t < 30 or t >= r_max-30]
        mu_list_post = select_from_list(mu_list_post, valid_r)
        sigma_list_post = select_from_list(sigma_list_post, valid_r)
        prob_list_post = select_from_list(prob_list_post, valid_r)
        r_list_update = [r_list_update[i] for i in valid_r]
#         r_list_update = [t for i, t in enumerate(r_list_update) if i <=i_r_max]
        
    
    # find the highest p
    i_shape_ml, i_r_ml = get_max_id(prob_list_post)
#     print len(prob_list_post[0]), len(r_list_update), i_shape_ml, i_r_ml, np.array(prob_list_post).shape
#     print prob_list_post
    if flag_print:
        print 'best prob is shape %d and dist to the last cp is %d' % (i_shape_ml, r_list_update[i_r_ml])
        
    if flag_print:
        print 'mu_list_post', mu_list_post
        print 'sigma_list_post', sigma_list_post
        print 'prob_list_post', prob_list_post, sum_prob(prob_list_post)
        print 'r_list_update', r_list_update
    r_list_update[i_r_ml]
    return prob_list_post, mu_list_post, sigma_list_post, r_list_update, (i_shape_ml, r_list_update[i_r_ml], np.max(prob_list_post))


# In[378]:

def disaggregate_with_shape(t, y, shapes):
    sigma_init = 20

    t, y = rel_change_filter(t, y, thre=.2)
    n_shapes = len(shapes)

    r_list = [0]
    prior_list = [[np.log(1./n_shapes)]] * n_shapes
    mu_list = [[0]] * n_shapes
    sigma_list =[[sigma_init]] * n_shapes

    # print r_list
    # print prior_list
    # print mu_list
    # print sigma_list

    last_y = 100
    sigma_measurement = 20
    cp_results = []
    cp_counter = create_counter()
    all_cp = []
    r_list_len = []
    counter_max = 800
    for counter in range(len(y)):
        prior_list, mu_list, sigma_list, r_list, most_likely_cp = update_prob(y[counter], prior_list, r_list, mu_list, sigma_list, shapes, last_y, sigma_measurement)
        vote(cp_counter, counter, counter-most_likely_cp[1], np.exp(most_likely_cp[2]))
        cp = search_call(cp_counter, 1)
        all_cp.extend(cp)
        all_cp = list(set(all_cp))
        cp_results.append(most_likely_cp)
        r_list_len.append(len(r_list))

        last_y = y[counter]

    all_cp = np.sort(all_cp)
    return t, y, all_cp


# In[384]:

get_ipython().run_cell_magic(u'time', u'', u'shapes = [cluster_mean_sort[i] for i in range(6) if i != 2]\nt, y, cp = disaggregate_with_shape(t_raw_4_16_19_23, y_raw_4_16_19_23, shapes)\nplot_with_cp(y,cp)')


# In[385]:

get_ipython().run_cell_magic(u'time', u'', u'shapes = [cluster_mean_sort[i] for i in range(6) if i != 2]\nt, y, cp = disaggregate_with_shape(t_raw_4_3_7_18, y_raw_4_3_7_18, shapes)\nplot_with_cp(y,cp)')


# In[386]:

get_ipython().run_cell_magic(u'time', u'', u'shapes = [cluster_mean_sort[i] for i in range(6) if i != 2]\nt, y, cp = disaggregate_with_shape(t_raw_4_4_7_18, y_raw_4_4_7_18, shapes)\nplot_with_cp(y,cp)')


# In[ ]:



