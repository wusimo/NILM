# coding=utf-8
import sys
mod_path = '/Users/Simo//Documents/energyanalytics/energyanalytics/disaggregation'
if not (mod_path in sys.path):
    sys.path.insert(0, mod_path)

import numpy as np
from util import find_nearest
from sklearn.utils.extmath import cartesian
from bayesian_cp_detect import bayesian_cp_3 as bcp
from bayesian_cp_detect import cp_detect

class CombinatorialOptimization(object):
    
    """
    Example:
    self.power_list = {
    "1": [0,20,30],
    "2": [0,100]
    }
    then self.state_combinations = [
    [0,0],
    [0,100],
    [20,0],
    [20,100],
    [30,0],
    [30,100]
    ]
    """
    def __init__(self,appliance_power_dict = {}):
        self.power_list = appliance_power_dict
        #self.state_combinations = []
        self.index_to_status = cartesian( [ i for i in range( len( self.power_list[app] ) ) ] for app in self.power_list)
        self.MODEL_NAME = "CO"
        self.compute_all_state()

    def compute_all_state(self):
        self.state_combinations = cartesian([self.power_list[appliance] for appliance in self.power_list])
    
    # given the raw data and change point list, segment the data according to change point list
    def segment_data(self, data, cp_list):
        data_seg = []
        data_seg_raw_last = []

        if len(cp_list)>0 and cp_list[-1]!=len(data)-1: 
            cp_list_2 = cp_list + [len(data)-1]
        else:
            cp_list_2 = cp_list
     
        for i in range(0, len(cp_list_2)-1):
            cp_s = cp_list_2[i]
            cp_e = cp_list_2[i+1]
            if cp_e - cp_s > 50:
                cp_e = cp_s+50
            #last_datum = np.mean( data[cp_s-3:cp_s] )
            #last_datum = np.mean(data[cp_list_2[i-1]:cp_s])
            data_seg.append([t for t in data[cp_s:cp_e]]) 
            data_seg_raw_last.append(data[cp_e])
        n_seg = len(data_seg)
        return data_seg, n_seg, data_seg_raw_last

    # TODO: decouple the following power_disaggregate
    def power_disaggregate(self,total_power_usage,r_blur = 30):
        # total_power_usage is simply a list
        n_equipment_type = len(self.power_list)

        t=np.array([i+1 for i in range(len(total_power_usage))])
        y = total_power_usage
        t_2, y_2 = bcp.rel_change_filter_0819_3(t,y)
        mu_list_list, sigma_list_list, prob_r_list_list, r_list_list = cp_detect.bayesian_change_point_4(y_2, r_blur=r_blur)
        changepoint, changepoint_p = cp_detect.get_change_point(prob_r_list_list)
        if len(changepoint)>0 and changepoint[-1]!=len(t_2)-1:
            changepoint.append(len(t_2)-1)
        cp_list = changepoint
        data_seg, n_seg, temp = self.segment_data( total_power_usage, cp_list )
        
        # compute the trace list 
        trace_list,_ = find_nearest( np.array( [ sum(s) for s in self.state_combinations ] ),np.array( [ np.mean( np.array(seg) ) for seg in data_seg ] ) )
        # generated predicted profile
        predicted_profile = [ [] for _ in range(n_equipment_type+1) ]
        
        if cp_list[-1]==len(total_power_usage)-1:
            cp_list = cp_list[:-1]

        for i_cp in range(len(trace_list)):
            t_start = cp_list[i_cp]
            if i_cp ==len(cp_list)-1:
                t_end = len(total_power_usage)
            else:
                t_end = cp_list[i_cp+1]

            for i_equipment in range(n_equipment_type):
                temp = self.state_combinations[trace_list[i_cp]][i_equipment]
                predicted_profile[i_equipment].extend([ temp for _ in range(t_end-t_start)])
        
        power_sum = np.sum(predicted_profile[:-1],axis = 0)
        others = predicted_profile[n_equipment_type]
        power_sum[ power_sum == 0 ] = 1
        print len(total_power_usage),len(power_sum)
        predicted_profile_2 = [ np.multiply( np.array(total_power_usage),np.divide(t,power_sum) ) for t in predicted_profile[:-1] ]

        return predicted_profile_2