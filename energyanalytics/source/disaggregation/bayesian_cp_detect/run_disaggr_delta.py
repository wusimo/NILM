#from filedef import *
from file_readef import *
import numpy as np
from os import path
from viterbi_delta import disaggregation

####################################################
#define status space/# of appliances/min# of status#
downsampling_interval=30 #whether downsampling
smooth=0 #whether smooth data

status=np.array([[0,1,2,3,4],
                 [0,1,2,3,4,5], # changeable only two status [0,1]
                 [0,1]])

power=np.array([[0,38,76,114,152],
                [0,130,190,210,285,333], #changeable [0,min,max]
                [0,28]])              


freq_conversion=np.array([0,1,0])

conditions=np.array([1,0,1])
sampling_interval=15 #datapoints per hour
num_change=1 #define max # of changing unit:1 to num_app
####################################################

#4.1 4.11 4.16
#Hstart:Hend--every 6 hours
#0-5, 6-11, 12-17, 18-23
date='4-16'
Hstart=18
Hend=23


(t,y)=np.array(readfile(date,Hstart,Hend))

if downsampling_interval>sampling_interval:
    interval=downsampling_interval/sampling_interval
    sampling_interval=downsampling_interval    
    y=np.array([y[i] for i in range(0,len(y),interval)])
    t=np.array([t[i] for i in range(0,len(y))])


picname=path.join('results', 'disaggr_2016-'+date+' '+str(Hstart)+'h to '+str(Hend)+'h')
stat_p_err, delta_event,y_filtered, idxpcp_start, idxpcp_end, value_event_mean,viterbi_status=disaggregation(picname,t,y,status,power,freq_conversion,conditions,smooth,sampling_interval,num_change)

power_all=[]
power_all.append(np.transpose(stat_p_err).tolist())
power_all=np.array(power_all)
