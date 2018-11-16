from __future__ import division

import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from os import path
import datetime

# from file_readef import *
# from outlierdef import preprocess

from bayesian_cp_detect.file_readef import *
from bayesian_cp_detect.outlierdef import preprocess
from bayesian_cp_detect.cp_detect import *
    
    
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


if __name__ == '__main__':
    date_start = datetime.date(2016,4,1)
    date_end = datetime.date(2016,5,1)
    date_current = date_start

    while date_current < date_end:
        date = str(date_current.month) + '-' + str(date_current.day)

        Hstart=7
        Hend=18

        [t,y] = np.array(readfile(date,Hstart,Hend,folder='data/IHG'))
        print date
        print len(y)

        data_input = preprocess(y)
        t_2, y_2 = rel_change_filter(t,data_input)
        y_3 = np.log(y_2)

        _, _, prob_r_list_list = bayesian_change_point(y_3, sigma_measurement=.1
                                                       , SIGMA_LOW = .1)
        changepoint, changepoint_p = get_change_point(prob_r_list_list)
        plt.close('all')
        fig = plot_change_point(t_2, y_2, changepoint)

        mu_list, sigma_list = get_posterior(y_3, changepoint, sigma_measurement=.1)
        plt.plot(mu_list, 'g-')
        plt.plot(np.array(mu_list)+np.array(sigma_list), 'g-')
        plt.plot(np.array(mu_list)-np.array(sigma_list), 'g-')

        plt.savefig(path.join('results', date + '-' + str(Hstart) + '-' + str(Hend) + '-2.jpg'))

        with open(path.join('results', date + '-' + str(Hstart) + '-' + str(Hend) + '.txt'), 'w') as fid:
            for i in changepoint:
                fid.write(str(i) + ',')
        date_current += datetime.timedelta(1)