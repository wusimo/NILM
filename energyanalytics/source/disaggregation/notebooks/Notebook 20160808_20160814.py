
# coding: utf-8

# ## goal:
# 
# * improve performance and change point detection: reduce false detection and improve performance for difficult data
# * use cluster result to assigna probabilities to segments of power usage.
# 

# In[4]:

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

import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")
    


# In[5]:

# repeat calculation in 3.3 in the last notebook

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


# In[6]:

# prepare example data
from bayesian_cp_detect import file_readef

def load_dat(date, Hstart, Hend, folder='data/IHG/'):
    (t,y)=np.array(file_readef.readfile(date,Hstart,Hend,folder))
    return t, y


# In[7]:

# example data, should be easy to disaggregate
# plot the example data

t_raw_4_16_19_23, y_raw_4_16_19_23 = load_dat('4-16', 19, 23, 'data/IHG/')

plt.figure(figsize=[18,3])
plt.plot(t_raw_4_16_19_23, y_raw_4_16_19_23, 'k.-')
plt.xlabel('t (15s sampling rate)')


# In[40]:

# write a function overlay change points to raw data
def plot_with_cp(y, cp_list=[]):
    plt.figure(figsize=(18,3))
    plt.plot(y, 'k-', linewidth=2)
    for cp in cp_list:
        plt.plot([cp,cp], [0, 500], 'k--', linewidth=1)
    plt.xlabel('t')
    plt.ylabel('power')


# In[42]:

get_ipython().run_cell_magic(u'time', u'', u'shapes = [cluster_mean_sort[i] for i in range(6) if i != 2]\nt, y, cp = disaggregate_with_shape(t_raw_4_16_19_23, y_raw_4_16_19_23, shapes)\nplot_with_cp(y,cp)')


# In[34]:

with open('metadata/april data.json', 'r') as fid:
    var_t = json.load(fid)
    cluster_mean_sort = var_t[2]
    cluster_std_sort = var_t[3]


# In[ ]:



