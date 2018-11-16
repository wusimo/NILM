import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from os import path
import datetime




def bayesian_change_point(data_input
                          , sigma_measurement = 20 # error for single measurement; parameter to tune
                          , TOL = 0.9999 # truncate tails of prob after this value
                          , mu_prior = None
                          , sigma_prior = None
                          , min_length_prior = 5
                          , gap_prior = 100.
                          , SIGMA_LOW = 10. # lowest Sigma of average level
                         ):
    '''method from Bayesian Online Changepoint Detection'''

    if not mu_prior: # set mu's prior if not specified
        mu_prior = np.mean(data_input)
    if not sigma_prior:
        sigma_prior = np.std(data_input)
    
    STORAGE_MAX = 10000 # at a cost of mem, make a look up table for log H and log 1-H
    R_MIN = 10 # min of length of 
    log_H_list = [np.log(1-1/(gap_prior*100))] * min_length_prior + [np.log(1-1/gap_prior)]*(STORAGE_MAX-min_length_prior) # hazard function, log(1-H)
    log_H_2_list = [np.log(1/(gap_prior*100))] * min_length_prior + [np.log(1/gap_prior)]*(STORAGE_MAX-min_length_prior) # log(H)

    mu_list = [mu_prior] # refresh at each measurement, prior mean of mu
    sigma_list = [sigma_prior] # prior std of mu
    prob_r_list = [0] # probability of each r

    prob_r_list_list = [prob_r_list] # history record
    mu_list_list = [mu_list]
    sigma_list_list = [sigma_list]

    prob_r_list_mod = [0]

    for datum in data_input: # for each new observation
        predictive_prob = [ # /pi_r
            -((datum-mu)/sigma_measurement)**2/2.0-np.log(sigma_measurement) 
            for mu, sigma in zip(mu_list, sigma_list)
        ]
        growth_prob = [ # prior * /pi_r * (1-H)
            p1 + p2 + log_H_list[i] 
            for i, (p1, p2) in enumerate(zip(predictive_prob, prob_r_list))
        ]

        try:
            change_prob = sp.misc.logsumexp([ # change point prob
                p1 + p2 + log_H_2_list[i] 
                for i, (p1, p2) in enumerate(zip(predictive_prob, prob_r_list))
            ])
        except: # current implement allows len(prob_r_list) to only reach STORAGE_MAX
            raise('power not change for a long time, reached max 1-H list')
        prob_r_list_update = [change_prob] # posterior
        prob_r_list_update.extend(growth_prob)
        evidence = sp.misc.logsumexp(prob_r_list_update)
        prob_r_list_update = [t-evidence for t in prob_r_list_update] # normalization

        mu_list_update = [mu_prior]
        sigma_list_update = [sigma_prior]

        tmp = [ # update mu and sigma of mu
            ((mu*sigma_measurement**2+datum*sigma**2)/(sigma_measurement**2+sigma**2), 
             np.sqrt(sigma_measurement**2*sigma**2/(sigma_measurement**2+sigma_prior**2)) ) 
            for mu,sigma in zip(mu_list, sigma_list)
        ]

        mu_list_update.extend([t[0] for t in tmp])
        sigma_list_update.extend([t[1] for t in tmp])
        sigma_list_update = [ # set lower bound of sigma to be sigma_low
            t if t > SIGMA_LOW else SIGMA_LOW 
            for t in sigma_list_update
        ]

        mu_list = mu_list_update
        sigma_list = sigma_list_update
        prob_r_list = prob_r_list_update
        if TOL: # truncation
            r_max = int( 
                np.min(
                    np.where(np.cumsum([np.exp(t) for t in prob_r_list]) > TOL)
                )
            )
            if r_max < R_MIN and len(prob_r_list)>R_MIN:
                r_max = R_MIN
            mu_list = [mu_list[i] for i in range(r_max+1)]
            sigma_list = [sigma_list[i] for i in range(r_max+1)]
            prob_r_list = [prob_r_list[i] for i in range(r_max+1)]
        mu_list_list.append(mu_list) # record
        sigma_list_list.append(sigma_list)
        prob_r_list_list.append(prob_r_list)
    mu_list_list.pop()
    sigma_list_list.pop()
    prob_r_list_list.pop()
    return mu_list_list, sigma_list_list, prob_r_list_list
    
def get_change_point(prob_r_list_list
                    , min_length_prior = 3 # minimal length
                    , p_thre = 0.07 # thre to call a change point
                    ):
    filtered_change_point_prob = [ # for example, for pos 10, 
        (i-min_length_prior+1, np.exp(j[min_length_prior-1])) 
        if len(j)>min_length_prior 
        else (i-min_length_prior+1, 0) 
        for i,j in enumerate(prob_r_list_list)
    ]
    tmp2 = [x[1] for x in filtered_change_point_prob]
    tmp3 = []
    for i in range(min_length_prior, len(filtered_change_point_prob)-min_length_prior):
        if np.argmax(tmp2[(i-min_length_prior+1):i+min_length_prior])+i-min_length_prior+1 == i:
            tmp3.append(filtered_change_point_prob[i])
    tmp3 = [x for x in tmp3 if x[1]>p_thre] # filter

    changepoint = [0]
    changepoint.extend([x[0] for x in tmp3])
    changepoint.extend([len(prob_r_list_list)-1])
    changepoint_p = [1]
    changepoint_p.extend([x[1] for x in tmp3])
    changepoint_p.extend([1])

    return changepoint, changepoint_p

def plot_change_point(t, data_input, changepoint):
    plt.ion()
    plt.show()
    fig = plt.figure(figsize=[np.ceil(len(data_input))/40,4])
    plt.plot(t, data_input, 'ko', markersize=2.5)
    data_max = np.max(data_input)
    for x in changepoint:
        plt.plot([t[x],t[x]], [0,data_max], 'k--', linewidth=1)
    
    plt.plot(t, get_dist_to_last_cp(len(data_input), changepoint)
             , 'k-')
    plt.xlabel('time')
    plt.ylabel('power')
    plt.xlim([0, len(data_input)])
    plt.ylim([0, data_max])
    plt.pause(.0001)
    return fig

def get_dist_to_last_cp(n # size of the signal array
                        , changepoint):
    last_cp = 0
    dist_to_last_cp = []
    for i in range(n):
        if changepoint[last_cp+1] == i:
            last_cp+=1
            dist_to_last_cp.append(i-changepoint[last_cp])
        else:
            dist_to_last_cp.append(i-changepoint[last_cp])
    return dist_to_last_cp

def get_posterior(data_input, changepoint
                  , sigma_measurement = 20 # error for single measurement; parameter to tune
                  , TOL = 0.9999 # truncate tails of prob after this value
                  , mu_prior = None
                  , sigma_prior = None
                  , SIGMA_LOW = 10 # lowest Sigma of average level
                 ):
    if not mu_prior:
        mu_prior = np.mean(data_input)
    if not sigma_prior:
        sigma_prior = np.std(data_input)

    mu = mu_prior
    sigma = sigma_prior
    mu_list = []
    sigma_list = []
    
    cp_current_index = 0
    cp_current = changepoint[cp_current_index]
    for i in range(len(data_input)):
        if i == changepoint[cp_current_index+1]:
            mu = mu_prior
            sigma = sigma_prior

            cp_current_index += 1
            cp_current = changepoint[cp_current_index]
        else:
            mu = (mu*sigma_measurement**2+data_input[i]*sigma**2)/(sigma_measurement**2+sigma**2)
            sigma = np.sqrt(sigma_measurement**2*sigma**2/(sigma_measurement**2+sigma**2))
        if sigma<SIGMA_LOW:
            sigma = SIGMA_LOW
        mu_list.append(mu)
        sigma_list.append(sigma)
    return mu_list, sigma_list

def bayesian_change_point_2(data_input
                          , sigma_measurement = 20 # error for single measurement; parameter to tune
                          , TOL = 0.9999 # truncate tails of prob after this value
                          , mu_prior = None
                          , sigma_prior = None
                          , min_length_prior = 5
                          , gap_prior = 100.
                          , SIGMA_LOW = 10. # lowest Sigma of average level
                         ):
    '''method from Bayesian Online Changepoint Detection
    this is a test version to remove all all-time storage to see if it speeds up the code'''

    if not mu_prior: # set mu's prior if not specified
        mu_prior = np.mean(data_input)
    if not sigma_prior:
        sigma_prior = np.std(data_input)
    
    STORAGE_MAX = 10000 # at a cost of mem, make a look up table for log H and log 1-H
    R_MIN = 10 # min of length of 
    log_H_list = [np.log(1-1/(gap_prior*100))] * min_length_prior + [np.log(1-1/gap_prior)]*(STORAGE_MAX-min_length_prior) # hazard function, log(1-H)
    log_H_2_list = [np.log(1/(gap_prior*100))] * min_length_prior + [np.log(1/gap_prior)]*(STORAGE_MAX-min_length_prior) # log(H)

    mu_list = [mu_prior] # refresh at each measurement, prior mean of mu
    sigma_list = [sigma_prior] # prior std of mu
    prob_r_list = [0] # probability of each r

    prob_r_list_list = [prob_r_list] # history record
    mu_list_list = [mu_list]
    sigma_list_list = [sigma_list]

    prob_r_list_mod = [0]

    for datum in data_input: # for each new observation
        predictive_prob = [ # /pi_r
            -((datum-mu)/sigma_measurement)**2/2.0-np.log(sigma_measurement) 
            for mu, sigma in zip(mu_list, sigma_list)
        ]
        growth_prob = [ # prior * /pi_r * (1-H)
            p1 + p2 + log_H_list[i] 
            for i, (p1, p2) in enumerate(zip(predictive_prob, prob_r_list))
        ]

        try:
            change_prob = sp.misc.logsumexp([ # change point prob
                p1 + p2 + log_H_2_list[i] 
                for i, (p1, p2) in enumerate(zip(predictive_prob, prob_r_list))
            ])
        except: # current implement allows len(prob_r_list) to only reach STORAGE_MAX
            raise('power not change for a long time, reached max 1-H list')
        prob_r_list_update = [change_prob] # posterior
        prob_r_list_update.extend(growth_prob)
        evidence = sp.misc.logsumexp(prob_r_list_update)
        prob_r_list_update = [t-evidence for t in prob_r_list_update] # normalization

        mu_list_update = [mu_prior]
        sigma_list_update = [sigma_prior]

        tmp = [ # update mu and sigma of mu
            ((mu*sigma_measurement**2+datum*sigma**2)/(sigma_measurement**2+sigma**2), 
             np.sqrt(sigma_measurement**2*sigma**2/(sigma_measurement**2+sigma_prior**2)) ) 
            for mu,sigma in zip(mu_list, sigma_list)
        ]

        mu_list_update.extend([t[0] for t in tmp])
        sigma_list_update.extend([t[1] for t in tmp])
        sigma_list_update = [ # set lower bound of sigma to be sigma_low
            t if t > SIGMA_LOW else SIGMA_LOW 
            for t in sigma_list_update
        ]

        mu_list = mu_list_update
        sigma_list = sigma_list_update
        prob_r_list = prob_r_list_update
        if TOL: # truncation
            r_max = int( 
                np.min(
                    np.where(np.cumsum([np.exp(t) for t in prob_r_list]) > TOL)
                )
            )
            if r_max < R_MIN and len(prob_r_list)>R_MIN:
                r_max = R_MIN
            mu_list = [mu_list[i] for i in range(r_max+1)]
            sigma_list = [sigma_list[i] for i in range(r_max+1)]
            prob_r_list = [prob_r_list[i] for i in range(r_max+1)]
#         mu_list_list.append(mu_list) # record
#         sigma_list_list.append(sigma_list)
#         prob_r_list_list.append(prob_r_list)
    mu_list_list.pop()
    sigma_list_list.pop()
    prob_r_list_list.pop()
    return mu_list_list, sigma_list_list, prob_r_list_list

def bayesian_change_point_3(data_input
                          , sigma_measurement = 20 # error for single measurement; parameter to tune
                          , TOL = 0.9999 # truncate tails of prob after this value
                          , mu_prior = None
                          , sigma_prior = None
                          , min_length_prior = 5
                          , gap_prior = 100.
                          , SIGMA_LOW = 10. # lowest Sigma of average level
                          , prob_r_truncation = -10.
                         ):
    '''method from Bayesian Online Changepoint Detection'''

    if not mu_prior: # set mu's prior if not specified
        mu_prior = np.mean(data_input)
    if not sigma_prior:
        sigma_prior = np.std(data_input)
    
    STORAGE_MAX = 10000 # at a cost of mem, make a look up table for log H and log 1-H
    R_MIN = 10 # min of length of 
    log_H_list = [np.log(1-1/(gap_prior*100))] * min_length_prior + [np.log(1-1/gap_prior)]*(STORAGE_MAX-min_length_prior) # hazard function, log(1-H)
    log_H_2_list = [np.log(1/(gap_prior*100))] * min_length_prior + [np.log(1/gap_prior)]*(STORAGE_MAX-min_length_prior) # log(H)

    r_list = [0]
    mu_list = [mu_prior] # refresh at each measurement, prior mean of mu
    sigma_list = [sigma_prior] # prior std of mu
    prob_r_list = [0] # probability of each r

    r_list_list = [r_list]
    prob_r_list_list = [prob_r_list] # history record
    mu_list_list = [mu_list]
    sigma_list_list = [sigma_list]

    prob_r_list_mod = [0]

    counter = 0
    
    for datum in data_input: # for each new observation
        predictive_prob = [ # /pi_r
            -((datum-mu)/sigma_measurement)**2/2.0-np.log(sigma_measurement) 
            for mu, sigma in zip(mu_list, sigma_list)
        ]
        growth_prob = [ # prior * /pi_r * (1-H)
            p1 + p2 + log_H_list[i] 
            for i, (p1, p2) in enumerate(zip(predictive_prob, prob_r_list))
        ]

        try:
            change_prob = sp.misc.logsumexp([ # change point prob
                p1 + p2 + log_H_2_list[i] 
                for i, (p1, p2) in enumerate(zip(predictive_prob, prob_r_list))
            ])
        except: # current implement allows len(prob_r_list) to only reach STORAGE_MAX
            raise('power not change for a long time, reached max 1-H list')
        prob_r_list_update = [change_prob] # posterior
        prob_r_list_update.extend(growth_prob)
        evidence = sp.misc.logsumexp(prob_r_list_update)
        prob_r_list_update = [t-evidence for t in prob_r_list_update] # normalization

        mu_list_update = [mu_prior]
        sigma_list_update = [sigma_prior]
        r_list_update = [0]
        r_list_update.extend([t+1 for t in r_list])

        tmp = [ # update mu and sigma of mu
            ((mu*sigma_measurement**2+datum*sigma**2)/(sigma_measurement**2+sigma**2), 
             np.sqrt(sigma_measurement**2*sigma**2/(sigma_measurement**2+sigma_prior**2)) ) 
            for mu,sigma in zip(mu_list, sigma_list)
        ]

        mu_list_update.extend([t[0] for t in tmp])
        sigma_list_update.extend([t[1] for t in tmp])
        sigma_list_update = [ # set lower bound of sigma to be sigma_low
            t if t > SIGMA_LOW else SIGMA_LOW 
            for t in sigma_list_update
        ]

        r_list = r_list_update
        mu_list = mu_list_update
        sigma_list = sigma_list_update
        prob_r_list = prob_r_list_update
        if TOL: # truncation
            r_max = int( 
                np.min(
                    np.where(np.cumsum([np.exp(t) for t in prob_r_list]) > TOL)
                )
            )
            if r_max < R_MIN and len(prob_r_list)>R_MIN:
                r_max = R_MIN
            mu_list = [mu_list[i] for i in range(r_max+1)]
            sigma_list = [sigma_list[i] for i in range(r_max+1)]
            prob_r_list = [prob_r_list[i] for i in range(r_max+1)]
            r_list = [r_list[i] for i in range(r_max+1)]
            
        # filter out the r(s) that has low prob
        low_r = [i for i, prob in enumerate(prob_r_list) if prob > prob_r_truncation or i<R_MIN]
        mu_list = [mu_list[i] for i in low_r]
        sigma_list = [sigma_list[i] for i in low_r]
        prob_r_list = [prob_r_list[i] for i in low_r]
        r_list = [r_list[i] for i in low_r]
        
        mu_list_list.append(mu_list) # record
        sigma_list_list.append(sigma_list)
        prob_r_list_list.append(prob_r_list)
        r_list_list.append(r_list)
        
        counter+=1
    mu_list_list.pop()
    sigma_list_list.pop()
    prob_r_list_list.pop()
    r_list_list.pop()
    return mu_list_list, sigma_list_list, prob_r_list_list, r_list_list


def bayesian_change_point_4(data_input
                          , sigma_measurement = 2 
                          , TOL = 0.9999 
                          , mu_prior = None
                          , sigma_prior = None
                          , min_length_prior = 3
                          , gap_prior = 10.
                          , SIGMA_LOW = 10.   
                          , prob_r_truncation = -10.
                          , r_blur = 100
                         ):
    '''method from Bayesian Online Changepoint Detection

        :param data_input: 1 dimension list of number for which changepoint need to be detected.
        :type data_input: list.
        :param sigma_measurement: error for single measurement; parameter to tune.
        :type  sigma_neasurement: int.
        :param TOL:  truncate tails of prob after this value.
        :type TOL: float.
        :param mu_prior: the prior mean for data_input, default set to None.
        :type mu_prior: float.
        :param sigma_prior: the prior standard deviation for data_input, default set to None.
        :type sigma_prior: float.
        :param min_length_prior: the prior of minimum duration that changepoint can not appear twice.
        :type min_length_prior: int.
        :param gap_prior: 
        :type gap_prior: int.
        :param SIGMA_LOW: lowest Sigma of average level.
        :type SIGMA_LOW: float
        :param prob_r_truncation:
        :type prob_r_truncation: float.
        :param r_blur:
        :type r_blur: int
        :returns: a list of list  -- the list of changepoints probablities for each time step
        
        Suppose you want to get all the changepoints of datalist
        
        >>> mu_list_list, sigma_list_list, prob_r_list_list, r_list_list = cp_detect.bayesian_change_point_4(data_list, r_blur=30)
        changepoint, changepoint_p = cp_detect.get_change_point(prob_r_list_list)
    
    '''

    ''' set mu's prior if not specified '''
    if not mu_prior: 
        mu_prior = np.mean(data_input)
    if not sigma_prior:
        sigma_prior = np.std(data_input)
    
    ''' use a lookup table to store the hazard function '''
    
    STORAGE_MAX = 10000 # at a cost of mem, make a look up table for log H and log 1-H
    R_MIN = 10 # min of length of 
    log_H_list = [np.log(1-1/(gap_prior*100))] * min_length_prior + [np.log(1-1/gap_prior)]*(STORAGE_MAX-min_length_prior) # hazard function, log(1-H)
    log_H_2_list = [np.log(1/(gap_prior*100))] * min_length_prior + [np.log(1/gap_prior)]*(STORAGE_MAX-min_length_prior) # log(H)

    
    # Initialize the data struction for the first point 
    r_list = [0] # the possible value for r_0 is only 0, since we assume that the first point is always a change point
    mu_list = [mu_prior] # refresh at each measurement, prior mean of mu
    sigma_list = [sigma_prior] # prior std of mu
    prob_r_list = [0] # probability of each r

    
    # the data struction holds all information 
    r_list_list = [r_list] 
    prob_r_list_list = [prob_r_list] # history record
    mu_list_list = [mu_list]
    sigma_list_list = [sigma_list]

    prob_r_list_mod = [0]

    counter = 0

    
    ''' read in each data point , the main loop '''

    for datum in data_input: 

        ''' calculate predictive probablity

        this one is actually a random variable since it is 
        conditioned on the various possible values of r at this time step.  
        so the predictive_prob is a list has the same length as the r_list at this time step.

        Each possible value of r will give us different mu and sigma, here we have the assumption of 
        the underlying data satisfies log normal distribution and the various 
        mu, sigma depends on the previous changepoints...) 

        this is exactly the step 3 in paper: Bayesian Online Changepoint Detection , Algorithm 1

        '''
        
        predictive_prob = [ # /pi_r
            -((datum-mu)/sigma_measurement)**2/2.0-np.log(sigma_measurement) 
            for mu, sigma in zip(mu_list, sigma_list)
        ]
        
        '''compute the Growth probability, using the lookup table for the hazard function of changepoint 

           this is exactly the step 4 in paper: Bayesian Online Changepoint Detection , Algorithm 1
        '''
        
        growth_prob = [ # prior * /pi_r * (1-H)
            p1 + p2 + log_H_list[i] 
            for i, (p1, p2) in enumerate(zip(predictive_prob, prob_r_list))
        ]

        try:
            '''compute the change probability, using the lookup table for the hazard function of changepoint

                this is exactly the step 5 in paper: Bayesian Online Changepoint Detection , Algorithm 1
            '''
            change_prob = sp.misc.logsumexp([ # change point prob
                p1 + p2 + log_H_2_list[i] 
                for i, (p1, p2) in enumerate(zip(predictive_prob, prob_r_list))
            ])
        except: 
            ''' current implement allows len(prob_r_list) to only reach STORAGE_MAX, the
                exception is for possible indexing reach STORAGE_MAX error
            '''

            raise('power not change for a long time, reached max 1-H list')


        prob_r_list_update = [change_prob] # posterior
        prob_r_list_update.extend(growth_prob)
        
        '''this is exactly the step 6 in paper: Bayesian Online Changepoint Detection , Algorithm 1 '''
        evidence = sp.misc.logsumexp(prob_r_list_update)
        
        '''this is exactly the step 7 in paper: Bayesian Online Changepoint Detection , Algorithm 1 '''
        prob_r_list_update = [t-evidence for t in prob_r_list_update] # normalization

        

        ''' step 8 in paper: Bayesian Online Changepoint Detection , Algorithm 1  '''
        mu_list_update = [mu_prior]
        sigma_list_update = [sigma_prior]
        r_list_update = [0]
        r_list_update.extend([t+1 for t in r_list])

        

        tmp = [ # update mu and sigma of mu
            ((mu*sigma_measurement**2+datum*sigma**2)/(sigma_measurement**2+sigma**2), 
             np.sqrt(sigma_measurement**2*sigma**2/(sigma_measurement**2+sigma_prior**2)) ) 
            for mu,sigma in zip(mu_list, sigma_list)
        ]

        mu_list_update.extend([t[0] for t in tmp])
        sigma_list_update.extend([t[1] for t in tmp])
        sigma_list_update = [ # set lower bound of sigma to be sigma_low
            t if t > SIGMA_LOW else SIGMA_LOW 
            for t in sigma_list_update
        ]

        # update the changepoint probability list, and the corresponding mean, std list for the segments.
        r_list = r_list_update
        mu_list = mu_list_update
        sigma_list = sigma_list_update
        prob_r_list = prob_r_list_update
        
        # to prevail too long r_list, truncate the low probability tails
        # this is not showed in the algorithm but was suggested in the description below Algorithm 1.
        # this is an optimization needed

        if TOL: # truncation
            r_max = int( 
                np.min(
                    np.where(np.cumsum([np.exp(t) for t in prob_r_list]) > TOL)
                )
            )
            if r_max < R_MIN and len(prob_r_list)>R_MIN:
                r_max = R_MIN
            mu_list = [mu_list[i] for i in range(r_max+1)]
            sigma_list = [sigma_list[i] for i in range(r_max+1)]
            prob_r_list = [prob_r_list[i] for i in range(r_max+1)]
            r_list = [r_list[i] for i in range(r_max+1)]
            
        # filter out the r(s) that has low prob
        # filter r(s) that larger than 100 but not too close to the max(r)
        r_max = np.max(r_list)
        low_r = [i for i, prob in enumerate(prob_r_list) if i<R_MIN or (i < r_blur and prob > prob_r_truncation) or (i > r_max-10)]
        mu_list = [mu_list[i] for i in low_r]
        sigma_list = [sigma_list[i] for i in low_r]
        prob_r_list = [prob_r_list[i] for i in low_r]
        r_list = [r_list[i] for i in low_r]
        
        # save the record
        mu_list_list.append(mu_list) 
        sigma_list_list.append(sigma_list)
        prob_r_list_list.append(prob_r_list)
        r_list_list.append(r_list)
        counter+=1
        
    mu_list_list.pop()
    sigma_list_list.pop()
    prob_r_list_list.pop()
    r_list_list.pop()
    
    return mu_list_list, sigma_list_list, prob_r_list_list, r_list_list