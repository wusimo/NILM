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
import collections
from scipy.stats import norm

class DisaggSuperClass(object):

    def __init__(self):
        return

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

class BayesianNILM(object):
    def __init__(self,appliance_power_dict = {}):
        self.power_list = appliance_power_dict
        self.index_to_status = cartesian( [ i for i in range( len( self.power_list[app] ) ) ] for app in self.power_list)
        self.MODEL_NAME = "BNILM"
        self.compute_all_state()

    def compute_all_state(self):
        self.state_combinations = cartesian([self.power_list[appliance] for appliance in self.power_list])

    def train(self,all_appliance_data):

        """ 
        When training, we are computing P(Y),P(X|Y), where P(Y) is the probability of a certain state occurs
        
        all_appliance_data = {"appliance_name":appliance history power data}

        """
        self.PY = collections.defaultdict(float)
        self.PXY = collections.defaultdict(list)
        self.P = collections.defaultdict(list)
        N = len(all_appliance_data[all_appliance_data.keys()[0]])
        mem = collections.defaultdict(list)
        for ind in range(N):
            state = [find_nearest(np.array(self.power_list[app]),np.array([all_appliance_data[app][ind]]))[0][0] for app in self.power_list]
            state = " ".join([str(i) for i in state])
            self.P[state].append(sum(all_appliance_data[app][ind] for app in self.power_list))
            self.PY[state]+=1
            mem[str(state)].append(sum([all_appliance_data[app][ind] for app in self.power_list]))
        # Compute PY
        for key in self.PY:
            self.PY[key] = float(self.PY[key]/N)

        #Compute P(X|Y), Assume normal distribution use mu and sigma computed above to obtain this probability
        for state in mem.keys():
            # collect history data corresponding to this state
            self.PXY[state].append(np.mean(mem[str(state)]))
            self.PXY[state].append(max(np.std(mem[str(state)]),0.01))

    
    def get_trace_list(self,data):
        
        res = []
        P = collections.defaultdict()
        for point in data:
            for state in self.PY:
                P[state] = self.PY[state]*norm.pdf(point,self.PXY[state][0],self.PXY[state][1])
            res.append([ int(i) for i in max(P,key = P.get).strip().split(" ")]) # get the key for the max item in the dictionary...this is hacky 
        return res

    def power_disaggregate(self,total_power_usage):
        
        n_equipment_type = len(self.power_list)
        trace_list = self.get_trace_list(total_power_usage)
        predicted_profile = [ [] for _ in range(n_equipment_type+1) ]

        cp_list = [ i for i in range(len(total_power_usage)) ]
        
        if cp_list[-1]==len(total_power_usage)-1:
            cp_list = cp_list[:-1]

        for i_cp in range(len(trace_list)-1):
            t_start = cp_list[i_cp]
            if i_cp ==len(cp_list)-1:
                t_end = len(total_power_usage)
            else:
                t_end = cp_list[i_cp+1]

            for i_equipment in range(n_equipment_type):
                temp = self.power_list[self.power_list.keys()[i_equipment]][trace_list[i_cp][i_equipment]]
                predicted_profile[i_equipment].extend([ temp for _ in range(t_end-t_start)])
        
        power_sum = np.sum(predicted_profile[:-1],axis = 0)
        others = predicted_profile[n_equipment_type]
        power_sum[ power_sum == 0 ] = 1
        print len(total_power_usage),len(power_sum)
        predicted_profile_2 = [ np.multiply( np.array(total_power_usage),np.divide(t,power_sum) ) for t in predicted_profile[:-1] ]

        return predicted_profile_2



class BayesianNILM_TimeConstraint(BayesianNILM):

	def __init__(self,appliance_power_dict = {}):
		super(BayesianNILM_TimeConstraint,self).__init__(appliance_power_dict)
		return 
	def Constraint(constraint):
		"""
		Constraint could be:
		{
		"2":[[12,1],[1,-1]]
		}
		which means appliance "2" is highly possible to be open during 12:00-13:00 and highly possible to be shut during 1:00 to 2:00...

		The Constraint will only change P(X,t), there could be two approaches:
		1. Save all the P(X,t) based on different t conditions, this will overwrite the training result


		2. Still use the old way to generate P(X) when training, lookup and apply the different P(X,t) on the fly while disaggregate

		Apparently there is a tradeoff between time and space complexity with these two different approaches.
		"""
		return
