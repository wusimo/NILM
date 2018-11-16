import json
import os
import datetime
import operator

import pandas as pd
import scipy.misc
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

class instrument:
    name = ''
    instrument_max = None
    i_shape = None
    power_max = None
    power_average = None
    
    def __init__(self, name, instrument_max):
        self.name = name
        self.instrument_max = instrument_max
        
    def set_shape(self, i_shape_list):
        self.i_shape.append(i_shape_list)


def set_disaggregation_option(time_resolution = 15., 
                              change_shape = [],
                              cp_interval = 900, # in unit of seconds
                              process_noise = 3.3,
                              measure_noise = 28.3,
                              init_pos_std = 8.16
                             ):
    """
    a dict that support other functions (similar to class member variables); 
    
    <key, value> description:

        time_resolution: time resolution in units of seconds, default 15.;
    
        change_shape: list of list. Each list is change of power comparing
            to the last change point; position zero is the first point after "change"
    
        cp_interval: expected interval of change point
        
        process_noise: at each step the variance of mean will increase by 
            process_noise^2
        
        measurement_noise: float, measurement noise
        
        init_pos_std: either float or list of float. A single float will be 
            repeated for n_change_shape times. This variable sets up the initial
            std of the location of each shape.
    
    automatic generated key-value pairs:
        
        n_change_shape: the number of shapes
        
        H: np.log(1-1./(cp_interval/time_resolution)), used in calculation
    
    """
    
    option = {
        'time_resolution': time_resolution, 
        'change_shape': change_shape, 
        'n_change_shape': len(change_shape),
        'cp_interval': cp_interval, 
        'H': np.log(1-1./(cp_interval/time_resolution)),
        'process_noise': process_noise, 
        'measure_noise': measure_noise,
        'init_pos_std': init_pos_std,
        'unhappy_count_thre': 5, 
        'len_protected': 5,
        'delta_shape': [float(50/3) for _ in range(len(change_shape))]
    }
    
    return option


def disaggregate(data, opt):

    unhappy_count_thre = opt['unhappy_count_thre']
    len_protected = opt['len_protected']
    
    current_data_pos = 0
    last_datum = 0

    log_prob, delta_mean, delta_var, time_since_last_cp = set_prior_7(opt)
    last_cp = 0

    cp_list = [last_cp]

    unhappy_count = 0
    
    while (current_data_pos<len(data)):
        datum = data[current_data_pos]
        log_prob, delta_mean, delta_var, time_since_last_cp = update_with_datum_7(datum, log_prob, delta_mean, delta_var, time_since_last_cp, last_datum, opt)
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
                log_prob, delta_mean, delta_var, time_since_last_cp = set_prior_7(opt)
                last_datum = np.mean( data[(last_cp-3):last_cp] )
                for current_data_pos_t in range(last_cp, last_cp + len_protected):
                    log_prob, delta_mean, delta_var, time_since_last_cp = update_with_datum_7(datum, 
                                                                                              log_prob, 
                                                                                              delta_mean,
                                                                                              delta_var,
                                                                                              time_since_last_cp,
                                                                                              last_datum,
                                                                                              opt)
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


def set_prior_7(para):
    """
    set prior before the first data came in
    doc details to be added
    """
    n_shape = para['n_change_shape']

    log_prob = [ [] for i_shape in range(n_shape) ]
    delta_mean = [ [] for i_shape in range(n_shape) ]
    delta_var = [ [] for i_shape in range(n_shape) ]
    time_since_last_cp = [ [] for i_shape in range(n_shape) ]
    
    return log_prob, delta_mean, delta_var, time_since_last_cp


def update_with_datum_7(datum, 
                      log_prob, 
                      delta_mean, 
                      delta_var, 
                      time_since_last_cp, 
                      last_datum, 
                      para):
    # extract parameters
    shape = para['change_shape']
    n_shape = para['n_change_shape']

    H = para['H'] # log probability that a new cp forms
    H_2_exp = 1 - np.exp(H)

    H_small_factor = np.exp(10)
    H_small = np.log(1 - H_2_exp/H_small_factor); # in case that the last change point is too close, use a small H_small
    
    delta_shape = para['delta_shape'] # shape noise
    
    Q = para['process_noise']**2 # process noise
    R = para['measure_noise']**2 # measurement noise
    
    delta_init = [float(t)**2 for t in para['delta_shape']]

    # a function that return element within the list or 
    # the last element of the list if that is not possible
    shape_helper = lambda i_shape, x: shape[i_shape][x] if x<len(shape[i_shape]) else shape[i_shape][-1]

    # step 1, grow log probability, and time since the last change point
    log_prob_grow = [ [] for _ in range(n_shape) ]
    time_since_last_cp_grow = [ [] for _ in range(n_shape)]

    # determine the probability of each (shape, \tau) at the current datum point
    # find the longest distance in time_since_last_cp
    if len(time_since_last_cp[0]) == 0: # this is the first data
        new_cp_prob = 1/float(n_shape)  # each shape is equally likely
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
            new_cp_prob = new_cp_prob / H_small_factor;
        
        for i_shape in range(n_shape):
            # log_prob_grow[i_shape] = [np.log(new_cp_prob)] + log_prob[i_shape][:-1] + [ log_prob[i_shape][-1]+H ]
            if r_max < 5:
                log_prob_grow[i_shape] = [np.log(new_cp_prob)] + log_prob[i_shape][:-1] + [ log_prob[i_shape][-1]+H_small ]
            else:
                log_prob_grow[i_shape] = [np.log(new_cp_prob)] + log_prob[i_shape][:-1] + [ log_prob[i_shape][-1]+H ]
            time_since_last_cp_grow[i_shape] = [0] + [x+1 for x in time_since_last_cp[i_shape]]

    # step 2, update the estimation of next data
    delta_mean_grow = [ [] for _ in range(n_shape) ]
    delta_var_grow = [ [] for _ in range(n_shape) ]
    
    for i_shape in range(n_shape):
        delta_mean_grow[i_shape] = [
            shape_helper(i_shape, x)+y 
            for x, y in zip( 
                    time_since_last_cp_grow[i_shape], 
                    [last_datum]+delta_mean[i_shape]
                    )
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


def log_norm_pdf(x
                 , mu
                 , sigma_2 # sigma^2
                ):
    return -(x-mu)**2/sigma_2 - np.log(2*np.pi*sigma_2)/2

def is_happy(prob, prob_thre=.3, len_protect = 5):
    last_cp_prob = np.sum( [np.exp(t[-1]) for t in prob] )
    return (last_cp_prob>prob_thre) or ( len(prob[0])<len_protect )

def trim_5(var, time_since_last_cp, time_thre=5):
    new_var = [[] for _ in range(len(var))]
    for i in range(len(var)):
        new_var[i] = [
            val 
            for pos, val in enumerate(var[i]) 
            if ((time_since_last_cp[i][pos]<time_thre) or (pos+1==len(var[i]) )) 
        ]
    return new_var


def segment_data(data, cp_list):
    data_seg = []
    cp_list_2 = cp_list + [len(data)-1]
    for i in range(1, len(cp_list_2)-1):
        cp_s = cp_list_2[i]
        cp_e = cp_list_2[i+1]
        if cp_e - cp_s > 50:
            cp_e = cp_s+50
        last_datum = np.mean( data[cp_s-3:cp_s] )
        data_seg.append([t-last_datum for t in data[cp_s:cp_e]])
    n_seg = len(data_seg)
    return data_seg, n_seg


def shape_code_gen(n):
    if (n==1):
        return [(0,), (1,)]
    else:
        result = []
        last_result = shape_code_gen(n-1)
        return [(0,)+t for t in last_result] + [(1,)+t for t in last_result]

def combine_shape(shape_matched, all_shape_code):
    shape_dict = {}
    n_shape_matched = len(shape_matched)
    for shape_code in all_shape_code:
        t = []
        for i_shape, flag in enumerate(shape_code):
            if flag:
                t.append(shape_matched[i_shape])
        shape_dict[tuple(shape_code)] = np.sum(np.array(t), axis=0)
    shape_dict[tuple(0 for _ in range(n_shape_matched))] = np.zeros(50)
    return shape_dict

def l2_distance(list_1, list_2, last_point_w = 50, n=2):
    dis = 0
    tmp = [(x-y)**n for x,y in zip(list_1, list_2)]
    dis = np.sum( tmp )
    if len(list_1) >= len(list_2):
        dis+=last_point_w*(list_1[-1] - list_2[-1])**n
    return dis / (len(tmp)+last_point_w)


def get_seg_prob( data_seg, shape_dict ):
    shape_prob_list = []
    var_measurement = 800

    for i_seg, seg in enumerate(data_seg):
        distance_list = []
        distance_dict = {}
        seg_mean = np.mean(seg)

        for shape_code, shape in shape_dict.items():
            if seg_mean > 0:
                distance_dict[shape_code] = np.exp( -l2_distance(seg, shape) / var_measurement )
                distance_dict[tuple(-t for t in shape_code)] = 0            
            else:
                distance_dict[tuple(-t for t in shape_code)] = np.exp( -(shape[-1] - (-seg_mean))**2 / var_measurement )
                distance_dict[shape_code] = 0

        z = np.sum(distance_dict.values())
        distance_dict = {k:v/z for k,v in distance_dict.items()}

        shape_prob_list.append(distance_dict)
    return shape_prob_list


def viterbi(shape_prob_list, state_prob_list, data_seg, obs_mat ):
    n_seg = len(data_seg)

    state_prob_list_list = [state_prob_list]
    state_memory_list_list = []
    shape_memory_list_list = []
    
    for i_seg in range(n_seg):
        seg_mean = np.mean(data_seg[i_seg])

        next_state_prob_list = {t:0 for t in state_prob_list.keys()}
        state_memory_list = {t:0 for t in state_prob_list.keys()}
        shape_memory_list = {t:0 for t in state_prob_list.keys()}

        for next_state, next_state_prob in next_state_prob_list.items():

            max_prob = -float('Inf')
            max_past_state = tuple()
            max_shape = ()
            for shape_code, shape_prob in shape_prob_list[i_seg].items():
                change_state = np.dot(obs_mat, shape_code)
                past_state = tuple(np.subtract(next_state, change_state))
                if past_state in state_prob_list:
                    if state_prob_list[past_state] * shape_prob > max_prob:
                        max_prob = state_prob_list[past_state] * shape_prob
                        max_past_state = past_state
                        max_shape = shape_code
            state_memory_list[next_state] = max_past_state
            next_state_prob_list[next_state] = max_prob
            shape_memory_list[next_state] = max_shape

        state_prob_list = next_state_prob_list

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


# Dissaggregation

def generate_predicted_profile(cp_list, shape_matched, shape_list, raw_data, n_equipment_type, obs_mat, trace_list):

    predicted_profile = [ [] for _ in range(n_equipment_type+1) ]
    
    predicted_profile[n_equipment_type].extend( [raw_data[0] for _ in range(len(raw_data))] )
    
    for i_equipment in range(n_equipment_type):
        for i_cp in range(len(cp_list)):
            t_start = cp_list[i_cp]
            if i_cp == len(cp_list)-1:
                t_end = len(raw_data)
            else:
                t_end = cp_list[i_cp+1]
            if trace_list[i_cp][i_equipment] == 0:
                predicted_profile[i_equipment].extend([0 for _ in range(t_end-t_start)])
            else:
                if i_cp == 0 or (trace_list[i_cp][i_equipment] == trace_list[i_cp-1][i_equipment]):
                    if i_cp == 0:
                        last_datum = 0
                    else:
                        last_datum = predicted_profile[i_equipment][-1]
                    predicted_profile[i_equipment].extend([last_datum for _ in range(t_end-t_start)])
                else:
                    change_profile = []
                    for i_shape in range(len(shape_list[0])):
                        if shape_list[i_cp-1][i_shape] > 0 and obs_mat[i_equipment][i_shape] > 0:
                            change_profile.append(shape_matched[i_shape])
                    if len(change_profile) > 1:
                        change_profile = np.sum(change_profile, axis=0)
                    change_profile = change_profile[0]
                    if (t_end-t_start) > len( shape_matched[i_shape] ):
                        predicted_profile[i_equipment].extend( list(change_profile) )
                        predicted_profile[i_equipment].extend( [change_profile[-1] for _ in range(t_end-t_start-len( shape_matched[i_shape] ))] )
                    else:
                        predicted_profile[i_equipment].extend( change_profile[:t_end-t_start] )
                   
    power_sum = np.sum(predicted_profile, axis=0)
    predicted_profile_2 = [np.multiply(raw_data, np.divide(t, power_sum)) for t in predicted_profile]
    
    return predicted_profile_2    


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

def rel_change(y):
    """
    return relative change comparing to the closer neighbouring points
    """
    return np.min([np.abs(y[1] - y[0]), np.abs(y[1] - y[2])]) / float(y[1])

def power_disaggregate(t, raw_data,
                       change_shape,
                       init_pos_std, 
                       shape_matched, 
                       state_prob_list, n_equipment_type, n_equipment, obs_mat):

    opt = set_disaggregation_option(change_shape=change_shape, 
                               init_pos_std = init_pos_std
                                )
    
    _, data = rel_change_filter_0819_3(range(len(raw_data)), raw_data, thre=.1)
    t_data = t
    
    cp_list = disaggregate( data, opt )
    opt = set_disaggregation_option(change_shape=change_shape, 
                                        init_pos_std = init_pos_std
                                       )
    data_seg, n_seg = segment_data(data, cp_list)

    n_shape_matched = len(shape_matched)
    all_shape_code = shape_code_gen(n_shape_matched)
    shape_dict = combine_shape(shape_matched, all_shape_code)

    shape_prob_list = get_seg_prob(data_seg, shape_dict)
    
    trace_list, shape_list = viterbi(shape_prob_list, state_prob_list, data_seg, obs_mat)

    predicted_profile = generate_predicted_profile(cp_list, shape_matched, shape_list, raw_data, n_equipment_type, obs_mat, 
                                                       trace_list)

    return predicted_profile
    

def construct_equipment_to_shape_map( equipment, shape_2_equip_map ):
    equip_2_shape_map = { i['id']:[] for i in equipment }
    for m in shape_2_equip_map.items():
        for e in m[1]:
            equip_2_shape_map[e].append(m[0])
    return equip_2_shape_map


def complete_shapes( equip_2_shape_map, shape_2_equip_map, shape_dict, equipment, SHAPE_LEN = 50 ):
    """
        for any equipment with no mapped shapes, add a shape based on its power 
        parameter, and update equip_2_shape_map, shape_2_equip_map, shape_dict
    """
    
    # find out ones with no mapping
    equip_no_shape = [x for x, y in equip_2_shape_map.items() if len(y) == 0]
    
    for e in equip_no_shape:
        # find that equipment from equipment
        t = [x for x in equipment if x['id'] == e ]
        t = t[0]
        
        i_shape_to_be_added = len(shape_dict)
        shape_dict[i_shape_to_be_added] = [ t['power'] ] * SHAPE_LEN
        shape_2_equip_map[i_shape_to_be_added] = e
        equip_2_shape_map[ e ].append( i_shape_to_be_added )
    
    return equip_2_shape_map, shape_2_equip_map, shape_dict

def viterbi_2(data_seg, equip_2_shape_map, shape_dict, equipment, init_state = (0,0,0,0,0,0), init_state_conf = 0.9):
    """
    apply viterbi algorithm to segmented data series
    data_seg: segmented time series, list of list of floats
    equip_2_shape_map: map from int to list of shape id(s)
    equipment: list of equipment specs, including at least an id, and number as the number of equipment
    """
    # viterbi here

    n_seg = len(data_seg)
    n_equipment = len(equipment)

    if len(init_state) != n_equipment:
		raise()
		
    all_state_list = all_possible_state_helper( equipment )  # all possible next state
    state_prob_list = {
        x:( init_state_conf if x == init_state else (1.-init_state_conf)/(len(all_state_list)-1) ) 
        for x in all_state_list
    }

    best_state_transtion_recorder = []  # more description here
    state_prob_recorder = []  # more description here
    past_state_best_path_recorder = []  # more description here

    for i_seg in range(n_seg):

        # print i_seg

        seg = data_seg[i_seg]  # segment series

        # print "state_prob_list: "
        # for k, v in state_prob_list.items():
        #     print k, v
        # print "size of state_prob_list: ", len(state_prob_list.keys())

        next_state_list = all_possible_state_helper( equipment )  # all possible next state
        # print "next_state_list: ", next_state_list
        # print "size of next_state_list:, ", len(next_state_list)
        next_state_prob_list = { x:0 for x in next_state_list }
        past_state_best_list = {  }
        past_state_best_path_list = {}
        for next_state in next_state_prob_list.keys():  # for each of the all possible equipment condition combinations
            past_state_prob_recorder = {}  # record the probability
            past_state_prob_path_recorder = {}  # record path to it
            tmp = gen_previous_state( next_state, max_change = 2, constraint=[e['number'] for e in equipment ])
            # print tmp
            # raise()
            for past_state in gen_previous_state( next_state, max_change = 2, constraint=[e['number'] for e in equipment ]):
                # note that for each past_state -> next_state, there can be multiple shapes that make this happen,
                # the function "get_prob" is to look up the most probable one
                # print next_state, past_state
                transition_prob, max_prob_path = get_prob( past_state, next_state, seg, equip_2_shape_map, shape_dict )
                past_state_prob_recorder[ past_state ] = transition_prob * state_prob_list[ past_state ]  # new probability = probability to reach last state * transition probability
                past_state_prob_path_recorder[ past_state ] = max_prob_path
                
            # looking for the best path to this state
            past_state_best = -1
            past_state_best_path = -1
            past_state_best_prob = -np.inf
            for k,v in past_state_prob_recorder.items():
                if v > past_state_best_prob:
                    past_state_best = k
                    past_state_best_prob = v
                    past_state_best_path = past_state_prob_path_recorder[past_state_best]

            # print "for the next state, ", next_state, " the best past state is ", past_state_best, " using path ", past_state_best_path, " with prob ", past_state_best_prob
            next_state_prob_list[next_state] = past_state_best_prob
            past_state_best_list[next_state] = past_state_best
            past_state_best_path_list[next_state] = past_state_best_path

        state_prob_list = next_state_prob_list
        best_state_transtion_recorder.append( past_state_best_list )
        state_prob_recorder.append(next_state_prob_list)
        past_state_best_path_recorder.append(past_state_best_path_list)

        # print "state_prob_list (new after iteration %d): " % (i_seg)
        # print "highest proba so far: ", np.max( state_prob_list.values() )
        # for k, v in state_prob_list.items():
        #     print k, v

        # if (i_seg == 0):
        #     raise()
    
    return state_prob_list, best_state_transtion_recorder, past_state_best_path_recorder


def all_possible_state_helper( equipment ):
    """
    create a list of tuples to represent all possible equipment combinations
    equipment is a list of dictionary that contain at least a key called number
    """
    result = []
    for i, e in enumerate( equipment ):
        if i == 0:
            for j in range(e['number']+1):  # add one to include maximal number of equipment
                result.append( (j, ) )
        else:
            new_result = []
            for k in result:
                for j in range(e['number']+1):
                    new_result.append( tuple([t for t in k] + [j,]) )
            result = new_result
    return result



def back_tracking(state_prob_list, best_state_transtion_recorder, past_state_best_path_recorder, shape_dict):
    # for k,v in past_state_best_path_recorder[0].items():
        # print k, v
    trace_list = []
    shape_list = []
    
    n_shape = len(shape_dict)
    
    current_state = -1
    t = -np.inf
    for k, v in state_prob_list.items():
        if v > t:
            t = v
            current_state = k
    for i in reversed(range(len(best_state_transtion_recorder))):
        best_path = past_state_best_path_recorder[i][current_state]
        path_list = [0] * n_shape
        
        best_path = best_path
        # print best_path
        # print i, current_state, best_path, past_state_best_path_recorder[0][(0, 1, 0, 1, 1, 0)]
        for x in best_path:
            x = x[0]
            if x[0] == '+':
                path_list[ int(x[1:]) ] = 1
            else:
                try:
                    path_list[ int(x[1:]) ] = -1
                except:
                    print path_list, x
                    path_list[ int(x[1:]) ] = -1
        
        trace_list.insert(0, current_state)
        shape_list.insert(0, path_list)
        
        current_state = best_state_transtion_recorder[i][current_state]
    trace_list.insert(0, current_state)
    return trace_list, shape_list

def generate_predicted_profile_2(cp_list, shape_matched, shape_list, raw_data, n_equipment_type, equip_2_shape_map, trace_list, equipment):

    predicted_profile = [ [] for _ in range(n_equipment_type+1) ]
    
    predicted_profile[n_equipment_type].extend( [0 for _ in range(len(raw_data))] )
    
    for i_equipment in range(n_equipment_type):
        for i_cp in range(len(cp_list)):
            t_start = cp_list[i_cp]
            if i_cp == len(cp_list)-1:
                t_end = len(raw_data)
            else:
                t_end = cp_list[i_cp+1]
            if trace_list[i_cp][i_equipment] == 0:
                predicted_profile[i_equipment].extend([0 for _ in range(t_end-t_start)])
            else:
                if i_cp == 0 or (trace_list[i_cp][i_equipment] == trace_list[i_cp-1][i_equipment]):
                    if i_cp == 0:
                        last_datum = equipment[i_equipment]['power']
                    else:
                        last_datum = predicted_profile[i_equipment][-1]
                    predicted_profile[i_equipment].extend([last_datum for _ in range(t_end-t_start)])
                else:
                    change_profile = []
                    for i_shape in range(len(shape_list[0])):
                        if shape_list[i_cp-1][i_shape] > 0 and (i_shape in equip_2_shape_map[i_equipment]) > 0:
                            change_profile.append(shape_matched[i_shape])
                    if len(change_profile) > 1:
                        change_profile = np.sum(change_profile, axis=0)
                    change_profile = change_profile[0]
                    if (t_end-t_start) > len( shape_matched[i_shape] ):
                        predicted_profile[i_equipment].extend( list(change_profile) )
                        predicted_profile[i_equipment].extend( [change_profile[-1] for _ in range(t_end-t_start-len( shape_matched[i_shape] ))] )
                    else:
                        predicted_profile[i_equipment].extend( change_profile[:t_end-t_start] )
                   
    power_sum = np.sum(predicted_profile, axis=0)
    predicted_profile_2 = [np.multiply(raw_data, np.divide(t, power_sum)) for t in predicted_profile]
    
    return predicted_profile_2 

def get_prob( past_state, next_state, seg, equip_2_shape_map, shape_dict, SHAPE_LEN = 50, var_measurement = 800 ):
    all_possible_shape = [[0] * SHAPE_LEN]
    all_possible_shape_path = [ [] ]
    
    flag = 0
    for i in range(len(past_state)):  # do not allow shut down and turn on together, this can be relaxed if needed
        if flag == 0 and past_state[i] != next_state[i]:
            flag = past_state[i] - next_state[i]
            continue
        if flag != 0 and past_state[i] != next_state[i]:
            if flag != (past_state[i] - next_state[i]):
                return -np.inf, []
    
    for pos in range(len(past_state)):
        if (past_state[pos] == next_state[pos]):  # this equipment is not changed
            continue
        else:
            for i in range( abs(past_state[pos] - next_state[pos]) ): # for each difference, add all combinations
                all_possible_shape_new = []
                all_possible_shape_path_new = []
                for base_shape, path in zip(all_possible_shape, all_possible_shape_path):
                    for i_shape in equip_2_shape_map[pos]:
                        tmp = [x for x in path]
                        if next_state[pos] == 0: # shut down
                            add_value = [-1 * shape_dict[i_shape][-1]] * SHAPE_LEN
                            if (len(tmp)==pos+1):
                                # tmp[pos] = tmp[pos] + '-'+str(i_shape)
                                tmp[pos] = (tmp[pos][0] + '-'+str(i_shape) ,)
                            else:
                                tmp.append( ('-'+str(i_shape),) )
                        else: # turn on
                            add_value = shape_dict[i_shape]
                            if (len(tmp) == pos+1):
                                tmp[pos] = (tmp[pos][0] + '+'+str(i_shape), )
                            else:
                                tmp.append( ('+'+str(i_shape),) )
                        all_possible_shape_new.append( [ x+y for x, y in zip(base_shape, add_value) ] )
                        all_possible_shape_path_new.append(tmp)
                all_possible_shape = all_possible_shape_new
                all_possible_shape_path = all_possible_shape_path_new

    # print past_state, next_state
    # print all_possible_shape, all_possible_shape_path
    # if (past_state!=next_state):
    #     raise()

    # look up most probable path
    max_prob = -np.inf
    max_prob_path = -1
    for t, t2 in zip(all_possible_shape, all_possible_shape_path):
        prob = np.exp( -l2_distance(seg, t) / var_measurement )
        if prob > max_prob:
            max_prob = prob
            max_prob_path = t2
    return max_prob, max_prob_path

def gen_previous_state(next_state, max_change = 2, allow_pos = 0, constraint = None):

    n_equip = len(next_state)
    if not constraint:
        constraint = [1] * n_equip

    # print next_state
    
    if allow_pos == n_equip and max_change > 0:
        return [tuple(next_state)]
    if max_change == 0:
        return [tuple(next_state)]
    result = [tuple(next_state)];
    
    for i in range( allow_pos, n_equip ):
        for j in range(-max_change, max_change+1):
            if j == 0:
                continue
            new_next_state = [t for t in next_state]  # copy next_state
            new_next_state[i] = new_next_state[i] + j
            if (new_next_state[i] < 0 or new_next_state[i]>constraint[i]):
                continue
            else:
                t = gen_previous_state(new_next_state, max_change = max_change-abs(j), allow_pos = i+1, constraint = constraint)
                result.extend( t )
    return result


def all_bin_state_helper(n):
    result = []
    for i in range( 2**n ):
        result.append( int2bin(i, n) )
    return result

def int2bin(i, n):
    result = []
    while n > 0:
        result.append( i % 2 )
        i /= 2
        n -= 1
    return tuple(result)

def read_dat_0819(date, h_start, h_end, folder_path):
    t = []
    dat = []
    t_utc = []
    start_time = None
    for h in range(h_start, h_end):
        file_name = '%d-%d-%d.csv' % (date.month, date.day, h)
        file_path = os.path.join(folder_path, file_name)
        try:
            data_pd = pd.read_csv(file_path, names=['time', 'data'])
            time_tmp = pd.to_datetime(data_pd.time,infer_datetime_format=True)
            if not start_time:
                start_time = time_tmp[0]
            tmp = [(x - start_time).seconds/3600. for x in time_tmp]
            t.extend(tmp)
            t_utc.extend(data_pd.time)
            dat.extend( [x/3 for x in data_pd.data] )
        except Exception as inst:
            print type(inst), inst.args, inst     # the exception instance
            print '%s failed' % file_path
#     t_utc = [x.to_datetime() for x in t_utc]
    return t, dat, t_utc