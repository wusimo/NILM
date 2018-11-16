#shape update shape
import sys
# set the input file path, you have to change this path
mod_path = '/Users/Simo//Documents/energyanalytics/energyanalytics/disaggregation'
if not (mod_path in sys.path):
    sys.path.insert(0, mod_path)
    
mod_path = '/Users/Simo//Documents/energyanalytics/energyanalytics/disaggregation'
if not (mod_path in sys.path):
    sys.path.insert(0, mod_path)
    
from .. import bayesian_cp_3 as bcp
from .. import cp_detect
# make sure that the code is loaded to the lastest version
reload(bcp)
import json
import numpy as np
import scipy as sp
import datetime
import matplotlib.pyplot as plt
import operator
import os
import pandas as pd
import string
import copy
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
import operator
import seaborn as sns
import operator
from collections import defaultdict
#from datadef import wdayformat

def wdayformat(tm_wday):
    if tm_wday==1:
        strwday='Mon'
    elif tm_wday==2:
        strwday='Tue'
    elif tm_wday==3:
        strwday='Wed'
    elif tm_wday==4:
        strwday='Thu'
    elif tm_wday==5:
        strwday='Fri'
    elif tm_wday==6:
        strwday='Sat'
    elif tm_wday==7:
        strwday='Sun'            
    return strwday

def readfile(f,Col): #read .csv files
    data=[]
    time=[]
    #head=''
    lines = f.readlines()
    #label=[]
    #head+=lines[0]
    for line in lines[1:]:
        line=line.strip('\n')
        line=line.split(',')
        tmp_time=float(line[0])
        tmp_data=0
        for i in range(len(Col)):
            tmp_data+=float(line[Col[i]])
        data.append(tmp_data),
        time.append(tmp_time),
    f.close()
    return (time,data)

def rel_change(y):
    """
    return relative change comparing to the closer neighbouring points
    """
    return np.min([np.abs(y[1] - y[0]), np.abs(y[1] - y[2])]) / float(y[1])

def rel_change_filter_0819_3(t, data_input, thre=.2):
    """
    filter data based on relative change
    data points in data_input that below or above both neighbouring points
    and have relative change above thre will be set as the average of neighbouring data.
    """
    thre_2 = thre/(1-thre)
    id_filter = [i for i in range(1, len(data_input)-1) 
     if (data_input[i]>data_input[i-1] and data_input[i]>data_input[i+1] and rel_change(data_input[i-1:i+2])>thre) or
                 (data_input[i]<data_input[i-1] and data_input[i]<data_input[i+1] and rel_change(data_input[i-1:i+2])>thre_2)
    ]
    data_input_2 = [(data_input[i-1]+data_input[i+1])/2 if (i in id_filter) else dat for i, dat in enumerate(data_input) ]
    return t, data_input_2

def initial_disaggregate(t_all,y_all,num_day,period = 1440):
    
    #date_current = date_start
    day = 0
    all_dat_seg = []
    while day < num_day:
        #print 'reading: ', date_current

        #t, y = read_dat_0819(date_current, 0, 23, '../new_data/IHG')
        t=np.array([i+1 for i in range(period)])
        y=y_all[(day)*period:(day+1)*period]
        t_2, y_2 = rel_change_filter_0819_3(t,y)
        mu_list_list, sigma_list_list, prob_r_list_list, r_list_list = cp_detect.bayesian_change_point_4(y_2, r_blur=30)
        changepoint, changepoint_p = cp_detect.get_change_point(prob_r_list_list)
        changepoint.append(len(t_2)-1)

        if len(changepoint)>1:
            dat_seg = [[y[cp_start:cp_end], y[cp_start-3:cp_start]] for cp_start, cp_end in zip(changepoint[:-1], changepoint[1:])]
        else:
            dat_seg = []
        all_dat_seg.extend(dat_seg)

        #date_current+=datetime.timedelta(1)
        day+=1
        
    return all_dat_seg

def Computeandplotsegments(t_all,y_all,days = 500):

    all_seg_april = initial_disaggregate(t_all,y_all,500,period = 96)
    all_seg_april_normalized = [np.array(x[0])-np.mean(x[1]) for x in all_seg_april if len(x[1])==3]
    all_seg_april_normalized = [x for x in all_seg_april_normalized if len(x)>0]
    all_positive_seg_april_normalized = [x for x in all_seg_april_normalized if x.min()>0]
    plt.figure(figsize=[8, 6])

    for x in all_seg_april_normalized:
        plt.plot(x, 'k-', linewidth=.5, alpha=.1)

    plt.xlabel('time point after change point')
    plt.ylabel('relative power shift')
    plt.xlim([0, 10])
    plt.ylim([-600, 4000])

def plot_24h_data(t, raw_data,cp_list,k=360):
    fig, axes = plt.subplots(nrows=4, figsize=[18, 10])
    
    for i, ax in enumerate(axes):
        #ax.plot(t, data, 'r-', markersize=3, linewidth=1, label='smooth')
        ax.plot(t, raw_data, 'k.', markersize=3, label='raw')
        
        for cp in cp_list:
            ax.plot([t[cp], t[cp]], [0, 1000], 'k-', linewidth=1)
        ax.set_ylabel('power')
        ax.set_xlim([0+i*k,k+i*k])
    ax.set_xlabel('time/h')
    plt.legend()

def extract_first_n(data_list, n):
    return [t[:n] for t in data_list if len(t)>=n]

# integrated functions:
def integrated_clustering(t_all,y_all,num_of_days=500,period = 1440,trim=10,min_n_clusters = 4, max_n_clusters=10,hierarchical=0):
    
    
    """
    method for finding the change shape based on unsupervised learning and changepoint detection on history data 
    
    :param t_all: 1 dimension list of index of the history data used for unsupervised learning
    :type t_all: list
    :param y_all: 1 dimension list containing values of the history data used for unsupervised learning
    :type y_all: list
    :param num_of_days: length of history data used in unit of days
    :type num_of_days: int
    :param period: How many data points per day, in other words, the inverse of frequency of the given data 
    :type period: int
    :param min_n_clusters: a prior knowledge on minimum number of clusters wanted
    :type min_n_clusters: int
    :param min_n_clusters: a prior knowledge on maximum number of clusters wanted
    :type min_n_clusters: int

    Suppose you want to get the cluster of change shapes from history data

    >>> cluster_mean,cluster_std,n_clusters,all_seg_per_cluster = integrated_clustering(t_all,y_all,num_of_days=500,period = 1440,trim=10,min_n_clusters = 17, max_n_clusters=18)
    
    and then plot it 

    >>> plot_cluster_result(cluster_mean,cluster_std,n_clusters,all_seg_per_cluster)

    """



    all_seg_april = initial_disaggregate(t_all,y_all,num_of_days,period = period)
    
    ''' '''
    all_seg_april_normalized = [np.array(x[0])-np.mean(x[1]) for x in all_seg_april if len(x[1])==3]
    
    ''' filter the empty segments'''
    all_seg_april_normalized = [x for x in all_seg_april_normalized if len(x)>0]
    
    ''' clustering in different ranges will probably have a better result'''
    if hierarchical == 0:
        pass
    elif hierarchical ==1:
        all_seg_april_normalized = [x for x in all_seg_april_normalized if x.mean()>1000]
    else:
        all_seg_april_normalized = [x for x in all_seg_april_normalized if x.mean()<1000]
    
    ''' filter out the positive segments'''
    all_positive_seg_april_normalized = [x for x in all_seg_april_normalized if x.min()>0]
    
    
    all_seg_april_normalized_trim50 = extract_first_n(all_positive_seg_april_normalized, trim)
    cluster_average = []
    
    # find optimal clustering number using silhouette score
    
    optimal_dict = {}
    
    for n_clusters in range(min_n_clusters,max_n_clusters):
        
        y_pred = KMeans(n_clusters=n_clusters).fit_predict(all_seg_april_normalized_trim50)

        cluster_average = []
        for i_cluster in range(n_clusters):
            cluster_average.append(
                np.mean([np.mean(x) for i, x in enumerate(all_seg_april_normalized_trim50) if y_pred[i]==i_cluster])
            ) 

        # sihouette score
        cluster_labels = y_pred
        sample_silhouette_values = silhouette_samples(all_seg_april_normalized_trim50, cluster_labels)
        
        silhouette_avg = silhouette_score(pd.DataFrame(all_seg_april_normalized_trim50), cluster_labels)

        optimal_dict[n_clusters] = silhouette_avg +(sample_silhouette_values.min()+sample_silhouette_values.max())/2
    
    # n_clusters will give us the optimal number of clusters
    n_clusters = max(optimal_dict.iteritems(), key=operator.itemgetter(1))[0]

    #print n_clusters
    
    y_pred = KMeans(n_clusters=n_clusters).fit_predict(all_seg_april_normalized_trim50)

    cluster_average = []
    
    for i_cluster in range(n_clusters):
        cluster_average.append(
            np.mean([np.mean(x) for i, x in enumerate(all_seg_april_normalized_trim50) if y_pred[i]==i_cluster])
        ) 
    cluster_average_rank = np.argsort(cluster_average)[::-1]
    rank_map = {cluster_average_rank[i_cluster]:i_cluster for i_cluster in range(n_clusters)} # old index:new index

    y_pred_old = y_pred
    y_pred = [rank_map[x] for x in y_pred]
    all_seg_per_cluster = [[] for i in range(n_clusters) ]
    for i_seg in range(len(all_seg_april_normalized_trim50)):
        all_seg_per_cluster[y_pred[i_seg]].append(all_seg_april_normalized_trim50[i_seg])
        
    cluster_mean = [[] for i in range(n_clusters) ]
    cluster_std = [[] for i in range(n_clusters) ]
    for i_cluster in range(n_clusters):
        cluster_mean[ i_cluster ] = np.mean(np.array(all_seg_per_cluster[i_cluster]), axis=0)
        cluster_std[ i_cluster ] = np.std(np.array(all_seg_per_cluster[i_cluster]), axis=0)
    
    
    
    
    #cluster_mean_2 = cluster_mean[5:6]
    
    return cluster_mean,cluster_std,n_clusters,all_seg_per_cluster

def DP_state_generation(N):
    if N==1:
        return [[0],[1]]
    else:
        return_list = DP_state_generation(N-1)
        toreturn = []
        for i in return_list:
            #print i
            i.append(0)
            toreturn.append(copy.copy(i))
            i.pop()
            i.append(1)
            #print i
            toreturn.append(copy.copy(i))
                
        #print toreturn
        return toreturn

    color_list = sns.color_palette("hls", n_clusters)

    fig, ax = plt.subplots(nrows=5,ncols=4,figsize=[20,12]);
    ax = ax.flatten()

    for i_cluster in range(n_clusters):
        ax_current = ax[i_cluster]

        for seg in all_seg_per_cluster[i_cluster]:
            ax_current.plot(seg, '-', linewidth=1, alpha=.3, color=color_list[i_cluster])
        ax_current.set_xlim([0, 10])
        ax_current.set_ylim([-500, 4000])
        ax_current.plot([0,50], [0,0], 'k--')    
        ax_current.plot(cluster_mean[i_cluster], color=color_list[i_cluster])
        ax_current.fill_between(range(10)
                                , cluster_mean[i_cluster]-cluster_std[i_cluster]
                                , cluster_mean[i_cluster]+cluster_std[i_cluster]
                                , color=color_list[i_cluster], alpha=.8)

def generate_state_prob_list(N):
    toreturn = {}
    toreturnlist = DP_state_generation(N)
    k = 1/float(len(toreturnlist))
    for i in toreturnlist:
        toreturn[tuple(i)] = k
        
    return toreturn

def viterbi_for_missing_change_point(shape_prob_list, state_prob_list, boot_state_prob_list, data_seg, obs_mat, power_usage, alpha = 1):
    
    '''
    
    Explanation:

    example suppose we have three segment , so in total we have 2 changepoints
    then the shape_prob_list
    
    

    Parameters
    ----------
    shape_prob_list: the probability of each shape at each changepoint

    state_prob_list: initial state probability 
    
    boot_state_prob_list:

    '''
    # originally shape means the 'change', state means the actual usage...

    # compute the number of segments
    n_seg = len(data_seg)

    # the list to store the result
    state_prob_list_list = [state_prob_list]
    state_memory_list_list = []
    shape_memory_list_list = []
    
    
    # the main loop to loop through all changepoint
    for i_seg in range(n_seg):
        # first to compute the mean of the current segment
        seg_mean = np.mean(data_seg[i_seg])
        # initialize the probability for next state
        next_state_prob_list = {t:0 for t in state_prob_list.keys()}
        state_memory_list = {t:0 for t in state_prob_list.keys()} #
        shape_memory_list = {t:0 for t in state_prob_list.keys()} #

        for next_state, next_state_prob in next_state_prob_list.items():

            max_prob = -float('Inf')
            max_past_state = tuple()
            max_shape = ()
            #print max(shape_prob_list[i_seg].iteritems(),key = operator.itemgetter(1))[0]
            
            # loop through all possible changes...
            for shape_code, shape_prob in shape_prob_list[i_seg].items():
                #print obs_mat,shape_code
                # compute the corresponding changes
                change_state = np.dot(obs_mat, shape_code) # if the obs_mat is identity matrix then the change_state = shape_code
                past_state = tuple(np.subtract(next_state, change_state))  # find out the corresponding past_state
                if past_state in state_prob_list: # the past state should be all positive
                    if state_prob_list[past_state] * shape_prob + alpha*boot_state_prob_list[i_seg+1][next_state]  > max_prob:
                        max_prob = state_prob_list[past_state] * shape_prob + alpha*boot_state_prob_list[i_seg+1][next_state]
                        max_past_state = past_state
                        max_shape = shape_code
            state_memory_list[next_state] = max_past_state  # the table 2, write down the most possible past state
            next_state_prob_list[next_state] = max_prob     
            shape_memory_list[next_state] = max_shape
        
        computed = sum(map(operator.mul,max(state_prob_list.iteritems(),key = operator.itemgetter(1))[0],power_usage))
        observed = sum(map(operator.mul,max(boot_state_prob_list[i_seg].iteritems(),key = operator.itemgetter(1))[0],power_usage))
        
        #print computed
        #print observed

        #if (float(computed)-float(observed))/float(observed)<0.3:
            
        state_prob_list = next_state_prob_list
        
        #print '##############'
        state_prob_list_list.append(next_state_prob_list)
        state_memory_list_list.append(state_memory_list)
        shape_memory_list_list.append(shape_memory_list)


    trace_list = []
    shape_list = []

    end_state = sorted(state_prob_list_list[-1].items(), key=operator.itemgetter(1))[-1][0]
    trace_list.insert(0, end_state)
    
    for i in reversed(range(n_seg)):
        max_shape = shape_memory_list_list[i][end_state]
        end_state = state_memory_list_list[i][end_state]
        trace_list.insert(0, end_state)
        shape_list.insert(0, max_shape)

    return trace_list, shape_list

def integrated_dissagregate(y_all,period,cluster_mean_2,day = 65,n_equipment_type = 4,n_equipment = [2,2,2,2],obs_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]),power_usage = [0,0,0,0]):
    
    opt = bcp.set_disaggregation_option(change_shape=cluster_mean_2, 
                               init_pos_std = np.sqrt([float(200/3), float(200/3), float(400/3), float(400/3)])
                                )
    
    t=np.array([i+1 for i in range(period)])
    data=y_all[(day)*period:(day+1)*period]
    cp_list = bcp.disaggregate(data,opt)
    data_seg, n_seg, data_seg_raw_last = bcp.segment_data(data, cp_list)
    new_data_seg, new_n_seg, new_data_seg_raw_last = bcp.segment_data_new(data, cp_list)
    shape_matched = cluster_mean_2
    n_shape_matched = len(shape_matched)
    #all_shape_code = bcp.shape_code_gen(n_shape_matched)
    all_shape_code = bcp.shape_code_gen_new(n_shape_matched,n_equipment[0])
    shape_dict = bcp.combine_shape(shape_matched, all_shape_code)
    #shape_prob_list = bcp.get_seg_prob(data_seg, shape_dict)

    new_shape_prob_list = bcp.get_seg_prob_positive(new_data_seg, shape_dict)
    
    
    
    
    
    shape_prob_list = bcp.get_seg_prob(data_seg, shape_dict)
    
    
    
    state_prob_list = generate_state_prob_list(len(cluster_mean_2))
    
    
    #obs_mat = np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])
    
    for item,keys in state_prob_list.iteritems():
        state_prob_list[item] = new_shape_prob_list[0][item]
    
    #trace_list, shape_list = bcp.viterbi(shape_prob_list, new_shape_prob_list[0], data_seg, obs_mat)
    #trace_list, shape_list = viterbi_new(shape_prob_list,state_prob_list,new_shape_prob_list,data_seg,obs_mat,power_usage)
    trace_list, shape_list = viterbi_for_missing_change_point(shape_prob_list,state_prob_list,new_shape_prob_list,data_seg,obs_mat,power_usage,alpha = 100)
    
    predicted_profile = bcp.generate_predicted_profile(cp_list, shape_matched, shape_list, data, n_equipment_type, obs_mat, trace_list)
    
    return predicted_profile

def plot_dissagregation(predicted_profile,t):
    plt.figure(figsize = [16,8])
    for tmp in predicted_profile:
        plt.plot(t,tmp,linewidth = 1)
    #plt.plot(t,data,'k.',markersize = 2)

    plt.xlim([1,100])
    plt.ylim([0,800])
    plt.xlabel('t/h')
    plt.ylabel('power')

def plot_dissagregation_2(predicted_profile,t):
    plt.figure(figsize = [16,8])
    for key,tmp in predicted_profile.iteritems():
        plt.plot(t,tmp,linewidth = 1)
    #plt.plot(t,data,'k.',markersize = 2)

    plt.xlim([1,1500])
    plt.ylim([0,3000])
    plt.xlabel('t/h')
    plt.ylabel('power')

def n_dimension_identity_matrix(cluster_mean_2):
    to_return_list  = []
    for i in range(1,len(cluster_mean_2)+1):
        to_return_list.append([0 if j!=i else 1 for j in range(1,len(cluster_mean_2)+1)])
    return to_return_list

def plot_cluster_result(cluster_mean,cluster_std,n_clusters,all_seg_per_cluster):  

    color_list = sns.color_palette("hls", n_clusters)

    fig, ax = plt.subplots(nrows=5,ncols=4,figsize=[20,12]);
    ax = ax.flatten()

    for i_cluster in range(n_clusters):
        ax_current = ax[i_cluster]

        for seg in all_seg_per_cluster[i_cluster]:
            ax_current.plot(seg, '-', linewidth=1, alpha=.3, color=color_list[i_cluster])
        ax_current.set_xlim([0, 10])
        ax_current.set_ylim([-500, 4000])
        ax_current.plot([0,50], [0,0], 'k--')    
        ax_current.plot(cluster_mean[i_cluster], color=color_list[i_cluster])
        ax_current.fill_between(range(10)
                                , cluster_mean[i_cluster]-cluster_std[i_cluster]
                                , cluster_mean[i_cluster]+cluster_std[i_cluster]
                                , color=color_list[i_cluster], alpha=.8)

def construct_obs_mat(list_of_shapes,mapping_variable,appliance_list):
    obs_mat = []
    for i in range(len(appliance_list)):
        row = [0 for jj in range(len(list_of_shapes))]
        for k in mapping_variable[i]:
            row[k-1] = 1
        obs_mat.append(row)
    return obs_mat

def wrapped_integrated_dissagregate(y_all,appliance_list,mapping_variable,power_usage,list_of_shapes,period = 1440,day = 65):
    predicted_profile = integrated_dissagregate(y_all,period,list_of_shapes,day = day,n_equipment_type = len(appliance_list),n_equipment=[2 for i in range(1,len(appliance_list)+1)],obs_mat=construct_obs_mat(list_of_shapes,mapping_variable,appliance_list),power_usage = power_usage)
    toreturn = {}
    for i in range(len(appliance_list)-1):
        toreturn[appliance_list[i]] = predicted_profile[i]
    toreturn['other'] = predicted_profile[-1]
    return toreturn

'''
# get the file input
filename='/Users/Simo/Desktop/equota/disaggrREDD/house1_output15s'
filext='.dat'
f_input = file(filename+filext,'r')
AppNo=[3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20] #Choose App#
period=1440
N=1
# the data will be t_all (time) y_all(consumption)
(t_all,y_all)=np.array(readfile(f_input,[i-2 for i in AppNo]))

# start clustering
cluster_mean,cluster_std,n_clusters,all_seg_per_cluster = integrated_clustering(t_all,y_all,num_of_days=500,period = 1440,trim=10,min_n_clusters = 17, max_n_clusters=18)
#plot_cluster_result(cluster_mean,cluster_std,n_clusters,all_seg_per_cluster)


t=np.array([i+1 for i in range(period)])
y=y_all[N*period:(N+1)*period]
day = 65
t=np.array([i+1 for i in range(period)])
y=y_all[(day)*period:(day+1)*period]
t_2, y_2 = rel_change_filter_0819_3(t,y)
mu_list_list, sigma_list_list, prob_r_list_list, r_list_list = cp_detect.bayesian_change_point_4(y_2, r_blur=30)
changepoint, changepoint_p = cp_detect.get_change_point(prob_r_list_list)
changepoint.append(len(t_2)-1)

#if you want to plot call this function   
#plot_24h_data(t_2,y_2,changepoint)


# start dissagregate
cluster_mean_2 = cluster_mean
cluster_mean = []
cluster_mean.append(cluster_mean_2[6])
cluster_mean.append(cluster_mean_2[10])
cluster_mean.append(cluster_mean_2[15])
cluster_mean.append(cluster_mean_2[16])
cluster_mean.append(cluster_mean_2[13])
cluster_mean_2 = cluster_mean

# input for the dissagregate
list_of_shapes = cluster_mean_2
appliance_list = [1,2,3,4,5]
mapping_variable = [[1],[2],[3],[4],[5]] # in this case this is  a one-to-one mapping
power_usage = [i.mean() for i in cluster_mean_2] # you need to have the information about the power usage for each appliances
predicted_profile = wrapped_integrated_dissagregate(y_all,appliance_list,mapping_variable,power_usage,list_of_shapes,period = 1440,day = 65)
plot_dissagregation_2(predicted_profile,np.array([i+1 for i in range(period)]))

#predicted_profile = integrated_dissagregate(y_all,1440,cluster_mean_2,day = 65,n_equipment_type = len(cluster_mean_2),n_equipment=[2 for i in range(1,len(cluster_mean_2)+1)],obs_mat=n_dimension_identity_matrix(cluster_mean_2),power_usage = [i.mean() for i in cluster_mean_2])  
#print predicted_profile
#plot_dissagregation(predicted_profile,np.array([i+1 for i in range(period)]))


#TODO: missing functions to handle update clustering information


#mapping
''' 