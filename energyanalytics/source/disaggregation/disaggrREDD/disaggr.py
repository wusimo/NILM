#from filedef import *
from filedef import *
import numpy as np
from viterbi import disaggregation

filename='house1_output15s'
filext='.dat'
f_input = file(filename+filext,'r')

####################################################
N=1 #start time
period=1440 # *15s
#period=1440*4
AppNo=[5,6,9,12,17] #Choose App#

####################################################
#define status space/# of appliances/min# of status#
power=np.array([[6,195,425],
                [0,181,1100],
                [0,81,146,340],
                [0,1600],
                [0,65,72,100]])

sampling_rate=60 #datapoints per hour
num_change=1 #define max # of changing unit:1 to num_app



####################################################
status=[]
for item in power:
    status.append([i for i in range(len(item))])
status=np.array(status)
conditions=np.zeros((len(power),), dtype=np.int)

#obtain timestamp, readings
#unit of t: min
(t_all,y_all)=np.array(readfile(f_input,[i-2 for i in AppNo]))
t=np.array([i+1 for i in range(period)])
y=y_all[N*period:(N+1)*period]
picname='test15s'
stat_p_err,t_plot,y_plot=disaggregation(picname,t,y,status,power,conditions,sampling_rate=15,num_change=1)
###################################################
