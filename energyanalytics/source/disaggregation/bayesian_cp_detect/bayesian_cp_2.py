import numpy as np
import scipy as sp
import scipy.misc
import pandas as pd
from os import path
import datetime

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


def read_dat_0819(date, h_start, h_end, folder_path):
    t = []
    dat = []
    start_time = None
    for h in range(h_start, h_end):
        try:
            file_name = '%d-%d-%d.csv' % (date.month, date.day, h)
            file_path = path.join(folder_path, file_name)
            data_pd = pd.read_csv(file_path, names=['time', 'data'])
            time_tmp = pd.to_datetime(data_pd.time,infer_datetime_format=True)
            if not start_time:
                start_time = time_tmp[0]
            tmp = [(x - start_time).seconds/3600. for x in time_tmp]
            t.extend(tmp)
            dat.extend( [x/3 for x in data_pd.data] )
        except Exception as inst:
            print type(inst), inst.args, inst     # the exception instance
            print '%s failed' % file_path
    return t, dat


def read_dat_0910(datetime_s, datetime_e, folder_path):
    t = []
    dat = []
    start_time = None
    
    current_time = datetime_s
    
    while current_time < datetime_e:
        try:
            file_name = '%d-%d-%d.csv' % (current_time.month, current_time.day, current_time.hour)
            file_path = path.join(folder_path, file_name)
            data_pd = pd.read_csv(file_path, names=['time', 'data'])
            time_tmp = pd.to_datetime(data_pd.time,infer_datetime_format=True)
            if not start_time:
                start_time = time_tmp[0]
            tmp = [(x - start_time).days*24.+(x - start_time).seconds/3600. for x in time_tmp]
            t.extend(tmp)
            dat.extend( [x/3 for x in data_pd.data] )
        except Exception as inst:
            print type(inst), inst.args, inst     # the exception instance
            print '%s failed' % file_path
        current_time += datetime.timedelta(hours=1)
    return t, dat


def test_func():
    print 5

    
def log_norm_pdf(x
                 , mu
                 , sigma_2 # sigma^2
                ):
    return -(x-mu)**2/sigma_2 - np.log(2*np.pi*sigma_2)/2


def set_prior_6(para):
    """
    set prior before the first data came in
    doc details to be added
    """
    n_shape = para['n_shape']

    log_prob = [ [] for i_shape in range(n_shape) ]
    delta_mean = [ [] for i_shape in range(n_shape) ]
    delta_var = [ [] for i_shape in range(n_shape) ]
    time_since_last_cp = [ [] for i_shape in range(n_shape) ]
    
    return log_prob, delta_mean, delta_var, time_since_last_cp


def update_with_datum_6(datum, 
                      log_prob, 
                      delta_mean, 
                      delta_var, 
                      time_since_last_cp, 
                      last_datum, 
                      para):
    # extract parameters
    shape = para['shape']
    n_shape = para['n_shape']

    H = para['H'] # log probability that a new cp forms
    H_2_exp = 1 - np.exp(H)
    
    delta_shape = para['delta_shape'] # shape noise
    
    Q = para['Q'] # process noise
    R = para['R'] # measurement noise
    
    delta_init = para['delta_init']

    # a function that return element within the list or 
    # the last element of the list if that is not possible
    shape_helper = lambda i_shape, x: shape[i_shape][x] if x<len(shape[i_shape]) else shape[i_shape][-1]

    # step 1, grow log probability, and time since the last change point
    log_prob_grow = [ [] for _ in range(n_shape) ]
    time_since_last_cp_grow = [ [] for _ in range(n_shape)]

    # find the longest distance in time_since_last_cp
    if len(time_since_last_cp[0]) == 0: # this is the first data
        new_cp_prob = 1/float(n_shape)
        for i_shape in range(n_shape):
            log_prob_grow[i_shape] = [np.log(new_cp_prob)]
            time_since_last_cp_grow[i_shape] = [0]
    else:
        # distance from this data point to the last confirmed change point
        r_max = np.max( [t for x in time_since_last_cp for t in x] )
        
        # find probability of all shapes at r_max
        total_prob_since_last_cp = np.sum( [np.exp(t[-1]) for t in log_prob] )
        new_cp_prob = total_prob_since_last_cp * H_2_exp / n_shape

        if r_max < 5:
            new_cp_prob = np.exp(-50)
        
        for i_shape in range(n_shape):
            log_prob_grow[i_shape] = [np.log(new_cp_prob)] + log_prob[i_shape][:-1] + [ log_prob[i_shape][-1]+H ]
            time_since_last_cp_grow[i_shape] = [0] + [x+1 for x in time_since_last_cp[i_shape]]

    # step 2, update the estimation of next data
    delta_mean_grow = [ [] for _ in range(n_shape) ]
    delta_var_grow = [ [] for _ in range(n_shape) ]
    
    for i_shape in range(n_shape):
        delta_mean_grow[i_shape] = [
            shape_helper(i_shape, x)+y 
            for x, y in zip(time_since_last_cp_grow[i_shape], [last_datum]+delta_mean[i_shape])
        ]
        delta_var_grow[i_shape] = [
            delta_init[i_shape]
        ] + [
            x+Q for x in delta_var[i_shape]
        ]
    
    # estimate probability
    p_predict = [ [ ] for i_shape in range(n_shape) ]
    for i_shape in range(n_shape):
        n_tau = len(delta_mean_grow[i_shape])
        tmp = [ 0 for _ in range(n_tau) ]
        for i_tau in range(n_tau):
            tmp[i_tau] = log_norm_pdf( datum, delta_mean_grow[i_shape][i_tau], delta_var_grow[i_shape][i_tau] + R )
        p_predict[i_shape] = tmp

    # Update step
    delta_mean_posterior = [ [] for _ in range(n_shape)]
    delta_var_posterior = [ [] for _ in range(n_shape) ]
    for i_shape in range(n_shape):
        n_tau = len(delta_mean_grow[i_shape])
        delta_mean_tmp = [ [] for _ in range(n_tau) ]
        delta_var_tmp = [ [] for _ in range(n_tau) ]
        for i_tau in range(n_tau):
            K = delta_var_grow[i_shape][i_tau] / (delta_var_grow[i_shape][i_tau]+R)
            offset = datum - delta_mean_grow[i_shape][i_tau]
            delta_mean_tmp[i_tau] = delta_mean_grow[i_shape][i_tau] + K * offset
            delta_var_tmp[i_tau] = (1-K) * delta_var_grow[i_shape][i_tau]
        delta_mean_posterior[i_shape] = delta_mean_tmp
        delta_var_posterior[i_shape] = delta_var_tmp
            
    # update prob
    log_prob_posterior = [ [] for _ in range(n_shape) ]
    for i_shape in range(n_shape):
        log_prob_posterior[i_shape] = [x+y for x,y in zip(log_prob_grow[i_shape], p_predict[i_shape])]

    # normalization
    Z = sp.misc.logsumexp([x for t in log_prob_posterior for x in t])
    for i_shape in range(n_shape):
        log_prob_posterior[i_shape] = [x-Z for x in log_prob_posterior[i_shape]]
        
    # discount mean
    time_since_last_cp_posterior = time_since_last_cp_grow
    for i_shape in range(n_shape):
        delta_mean_posterior[i_shape] = [x-shape_helper(i_shape, y) for x, y in zip(delta_mean_posterior[i_shape], time_since_last_cp_posterior[i_shape])]

    return log_prob_posterior, delta_mean_posterior, delta_var_posterior, time_since_last_cp_posterior

def is_happy(prob, prob_thre=.3, len_protect = 5):
    last_cp_prob = np.sum( [np.exp(t[-1]) for t in prob] )
    return (last_cp_prob>prob_thre) or ( len(prob[0])<len_protect )

# log_prob, delta_mean, delta_var, time_since_last_cp

def trim_5(var, time_since_last_cp, time_thre=5):
    new_var = [[] for _ in range(len(var))]
    for i in range(len(var)):
        new_var[i] = [
            val 
            for pos, val in enumerate(var[i]) 
            if ((time_since_last_cp[i][pos]<time_thre) or (pos+1==len(var[i]) )) 
        ]
    return new_var


def disaggregate(data, para):

    unhappy_count_thre = para['unhappy_count_thre']
    len_protected = para['len_protected']
    
    current_data_pos = 0
    last_datum = 0

    log_prob, delta_mean, delta_var, time_since_last_cp = set_prior_6(para)
    last_cp = 0

    cp_list = [last_cp]

    unhappy_count = 0
    
    while (current_data_pos<len(data)):
        datum = data[current_data_pos]
        log_prob, delta_mean, delta_var, time_since_last_cp = update_with_datum_6(datum, log_prob, delta_mean, delta_var, time_since_last_cp, last_datum, para)
        leader_prob = np.sum( [np.exp(t[-1]) for t in log_prob] )
        leader_shape = np.argmax( [t[-1] for t in log_prob] )

        flag_happy = is_happy(log_prob)
        if current_data_pos >= 3149 and current_data_pos<3159:
            pass
            # print flag_happy, log_prob
        if flag_happy:
            
            unhappy_count = 0 # reset counter

            log_prob = trim_5(log_prob, time_since_last_cp) # trim data
            delta_mean = trim_5(delta_mean, time_since_last_cp)
            delta_var = trim_5(delta_var, time_since_last_cp)
            time_since_last_cp = trim_5(time_since_last_cp, time_since_last_cp)
            
            i = np.argmax([t[-1] for t in log_prob])
            
            if current_data_pos >= 3149 and current_data_pos<3159:
                pass
                # print current_data_pos, [t[-1] for t in log_prob]
            
        else:
            unhappy_count += 1
            if (unhappy_count == unhappy_count_thre):
                
                last_cp = current_data_pos - unhappy_count_thre
                cp_list.append(last_cp)
                
                unhappy_count = 0
                log_prob, delta_mean, delta_var, time_since_last_cp = set_prior_6(para)
                last_datum = np.mean( data[(last_cp-3):last_cp] )
                for current_data_pos_t in range(last_cp, last_cp + len_protected):
                    log_prob, delta_mean, delta_var, time_since_last_cp = update_with_datum_6(datum, 
                                                                                              log_prob, 
                                                                                              delta_mean,
                                                                                              delta_var,
                                                                                              time_since_last_cp,
                                                                                              last_datum,
                                                                                              para)
                    log_prob = [[t[-1]] for t in log_prob]
                    delta_mean = [[t[-1]] for t in delta_mean]
                    delta_var = [[t[-1]] for t in delta_var]
                    time_since_last_cp = [[t[-1]] for t in time_since_last_cp]
                
                z = np.log(np.sum([np.exp(t[-1]) for t in log_prob]))
                log_prob = [[t[-1]-z] for t in log_prob]
                
        current_data_pos += 1
        
        if current_data_pos < 3:
            last_datum = np.mean( data[0:current_data_pos] )
        else:
            last_datum = np.mean( data[(current_data_pos-3):current_data_pos] )
            
    return cp_list
