import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt


mod_path = '/Users/bh56/Dropbox/Equota/energyanalytics/disaggregation'
if not (mod_path in sys.path):
    sys.path.insert(0, mod_path)

from bayesian_cp_detect import bayesian_cp_2 as bcp

import json

with open('../metadata/cluster result.json', 'r') as fid:
    var = json.load(fid)
    
cluster_mean = [np.array(t) for t in var]
n_clusters = len(cluster_mean)

para = {}

para['delta_init'] = [float(200/3), float(200/3), float(400/3), float(400/3)]
para['H'] = np.log(1-1./(15*4)) # 15 min per cp

para['Q'] = float(10) # process error
para['R'] = float(800) # measurement error
para['shape'] = [cluster_mean[i] for i in [0,1,4,5]]
para['n_shape'] = len(para['shape'])
para['delta_shape'] = [float(50/3) for _ in range(para['n_shape'])]

para['unhappy_count_thre'] = 5
para['len_protected'] = 5


def plot_24h_data(t, raw_data, data, cp_list, title='24 hour data'):
    fig, axes = plt.subplots(nrows=4, figsize=[18, 10])
    
    for i, ax in enumerate(axes):
        ax.plot(t, data, 'r-', markersize=3, linewidth=1, label='smooth')
        ax.plot(t, raw_data, 'k.', markersize=3, label='raw')
        
        for cp in cp_list:
            ax.plot([t[cp], t[cp]], [0, 500], 'k-', linewidth=1)
        ax.set_ylabel('power')
        ax.set_xlim([0+i*6,6+i*6])
        if i==0:
            ax.set_title(title)
    ax.set_xlabel('time/h')
    plt.legend()

    
current_time = datetime.date(2016,4,1)
while current_time<datetime.date(2016,8,1):
    try:
        figure_title = '%02d-%02d.jpg' % (current_time.month, current_time.day)

        t, raw_data = bcp.read_dat_0819(current_time, 0, 24, '../new_data/IHG')
        if t[0]==t[1]:
            t = t[::2]
            raw_data = raw_data[::2]

        _, data = bcp.rel_change_filter_0819_3(range(len(raw_data)), raw_data, thre=.1)

        cp_list = bcp.disaggregate(data, para)
        plot_24h_data(t, raw_data, data, cp_list, title=figure_title)
        
        plt.savefig('../results/sep_10_result_'+figure_title)
        plt.close()

        print figure_title
    except:
        print figure_title+' failed'

    current_time+=datetime.timedelta(1)
