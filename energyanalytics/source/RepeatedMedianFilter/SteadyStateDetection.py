#from __future__ import absolute_import
# hacky way to import here
import sys, os
sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', 'disaggregation'))
from bayesian_cp_detect import bayesian_cp_3 as bcp
from bayesian_cp_detect import cp_detect
from bayesian_cp_detect.example import Dissagregation_functions as example
import scipy as sp
import numpy as np
import RMFilter
reload(RMFilter)

# TODO change all the return values to pandas dataframe with proper index

class SteadyStateDetector():
    def __init__(self):
        self.RMFilter = RMFilter.RepeatedMedianFilter()
        return

    def _changepoint_subroutine(self, data):
        """
        Method for finding the changepoint of a given 1d numerical value data
        """
        y = data
        t = np.array([i + 1 for i in range(len(y))])

        # Compute the changepoints
        t_2, y_2 = bcp.rel_change_filter_0819_3(t, y)
        mu_list_list, sigma_list_list, prob_r_list_list, r_list_list = cp_detect.bayesian_change_point_4(
            y_2, r_blur=30)
        changepoint, changepoint_p = cp_detect.get_change_point(
            prob_r_list_list)
        changepoint.append(len(t_2) - 1)

        return changepoint

    def _changepoint_to_stationary_state(self, changepoint, data):
        """
        Helper function to generate stationary value given the changepoints and original data by simply taking mean
        :param changepoint: 1 dimension list of index where the changepoint has been deteced.
        :type changepoint: list.
        :param data: 1 dimension list of number for which the stationary stateds need to be detected.
        :type data: list.
        """
        
        toreturn = []
        for i in range(len(changepoint) - 1):
            temp = np.mean(data[changepoint[i]:changepoint[i + 1]])
            toreturn = toreturn + [temp
                                   ] * (changepoint[i + 1] - changepoint[i])
        return toreturn

    def steady_state_without_filter(self, data):
        """
        Method for detecting stationary by simply finding changepoints of the given original data
        :param data: 1 dimension list of number for which the stationary stateds need to be detected.
        :type data: list.
        """
        # the input here are all python list, need to handle pandas dataframe inputs
        temp = data.fillna(0).values
        data = [i[0] for i in temp]
        changepoint = self._changepoint_subroutine(data)
        base = self._changepoint_to_stationary_state(changepoint, data)
        return base

    def steady_state_with_double_filter(self, data, tw1=5, tw2=15):
        """
        Method for detecting stationary by finding changepoints of the double filtered original data
        :param data: 1 dimension list of number for which the stationary stateds need to be detected.
        :type data: list.
        :param tw1: the small time window length parameter passed to the first median filter
        :type tw1: int
        :param tw2: the large time window length parameter passed to the second repeated median filter
        :type tw2: int
        """
        temp = data.fillna(0).values
        data = [i[0] for i in temp]
        filtereddata = self.RMFilter.double_filter(data, k1=tw1, k2=tw2)
        changepoint = self._changepoint_subroutine(filtereddata)
        base = self._changepoint_to_stationary_state(changepoint, data)
        return base

    def steady_state_using_median_slope(self, df):
        """
        Method for detecting stationary by finding changepoints of the median filtered original data
        :param df: 1 dimension list of number for which the stationary stateds need to be detected.
        :type df: list.
        """
        # after finding the non stationary points, set all of them to nan
        temp = df.fillna(0).values
        data = [i[0] for i in temp]
        abnormal_points = self.RMFilter.slope_median_filter(data)
        base = [j if i == 1 else np.nan for i, j in zip(abnormal_points, data)]
        return base


    

    