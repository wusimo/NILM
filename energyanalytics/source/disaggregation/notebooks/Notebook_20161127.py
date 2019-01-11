###################################
# example to use bayesian_cp_3

###################################
# import dependent packages

import datetime
import os
import json
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load disaggregation package
mod_path = '/Users/bh56/Dropbox/Equota/energyanalytics/disaggregation'
if not (mod_path in sys.path):
    sys.path.insert(0, mod_path)
    
mod_path = '/Users/bohua/Documents/Equota/energyanalytics/disaggregation'
# mod_path = '/Users/bohua/Dropbox/Equota/energyanalytics/disaggregation'
if not (mod_path in sys.path):
    sys.path.insert(0, mod_path)

print sys.path
    
from bayesian_cp_detect import bayesian_cp_3 as bcp

def main(args):

    ###################################
    # Parse command line input

    disagg_date = datetime.datetime.strptime(  # default as 20160401, otherwise use the first arg
        "20160401" if len(args) < 2 else args[1]
        , "%Y%m%d").date()

    ###################################
    # Load sample data (raw_data)
    # skip this part if data is provided by database

    print "loading time series data ..."
    
    t, raw_data, t_utc = bcp.read_dat_0819(disagg_date, 0, 24, '../new_data/IHG')
    
    _, data = bcp.rel_change_filter_0819_3(range(len(raw_data)), raw_data, thre=.1)
    t_data = t

    ###################################
    # import previously learnt shapes (cluster_mean_2)

    print "loading learnt shapes ..."
    with open('../metadata/cluster result.json', 'r') as fid:
        var = json.load(fid)
        
    cluster_mean_2 = [var[i] for i in [0,1,4,5]]  # 2 and 3 do not look right

    ###################################
    # other parameters from user

    # about equipment
    equipment = [
        {'name': 'HVAC_1', 'power': 100, 'id': 0, 'number': 3, 'class': 'hvac'}, 
        {'name': 'HVAC_2', 'power': 100, 'id': 1, 'number': 0, 'class': 'hvac'}, 
        {'name': 'HVAC_3', 'power': 100, 'id': 2, 'number': 0, 'class': 'hvac'}, 
        {'name': 'pump_1', 'power': 75, 'id': 3, 'number': 3, 'class': 'pump'}, 
        {'name': 'pump_2', 'power': 75, 'id': 4, 'number': 0, 'class': 'pump'}, 
        {'name': 'pump_3', 'power': 75, 'id': 5, 'number': 0, 'class': 'pump'}
    ]
    n_equipment = len(equipment)
    print 'n_equipment = ', n_equipment

    ###################################
    # disaggregate
    print "disaggregate ..."
    opt = bcp.set_disaggregation_option(change_shape=cluster_mean_2, 
                                   init_pos_std = np.sqrt([float(200/3), float(200/3), float(400/3), float(400/3)])
                                    )
    cp_list = bcp.disaggregate( data, opt )
    print cp_list

    data_seg, n_seg = bcp.segment_data(data, cp_list)

    # Process shapes
    # for any shapes with average < 0, remove it
    shape_matched = [ t for t in cluster_mean_2 if np.mean(t) > 0]
    shape_dict = {x:y for x,y in enumerate(shape_matched)}
    print 'learnt shapes: ', len(shape_dict)

    # mapping from known shapes to equipment, note that this mapping is a n-to-n mapping
    shape_2_equip_map = {
        0: [0, 1, 2], # this is saying that shape 0 can be interpretated as equipment 0, 1, 2 change
        1:[0, 1, 2]
    }

    equip_2_shape_map = bcp.construct_equipment_to_shape_map(equipment, shape_2_equip_map)
    equip_2_shape_map

    equip_2_shape_map, shape_2_equip_map, shape_dict = bcp.complete_shapes( equip_2_shape_map, 
                                                           shape_2_equip_map, 
                                                           shape_dict, 
                                                           equipment, 
                                                           SHAPE_LEN = 50 )

    print 'equip_2_shape_map: ', equip_2_shape_map
    print 'shape_2_equip_map:', shape_2_equip_map
    print 'len(shape_dict): ', len(shape_dict)

    # viterbi algorithm
    print "viterbi ..."
    print equipment
    state_prob_list, best_state_transtion_recorder, past_state_best_path_recorder = bcp.viterbi_2(data_seg, equip_2_shape_map, shape_dict, equipment, init_state = (0,0,0,1,0,0))

    trace_list, shape_list = bcp.back_tracking(state_prob_list, best_state_transtion_recorder, past_state_best_path_recorder, shape_dict)
    print trace_list
    print shape_list

    # predict profile
    print "create equipment profiles ..."
    predicted_profile = bcp.generate_predicted_profile_2(cp_list, shape_dict, shape_list, raw_data, len(equipment), equip_2_shape_map, trace_list, equipment)
    len(predicted_profile)

    plt.figure(figsize=[18,3])
    for tmp in predicted_profile:
        plt.plot(t_data, tmp, linewidth=1)
    # plt.plot(t_data, raw_data, 'k-', markersize=2)

    plt.xlim([0,24])
    plt.xlabel('t/h')
    plt.ylabel('power')
    plt.show()


    [sum(x) for x in predicted_profile]

if __name__ == '__main__':
    main(sys.argv)

