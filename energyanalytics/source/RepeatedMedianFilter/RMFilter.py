import numpy as np
import scipy as sp
from scipy import signal


class RepeatedMedianFilter():
    def __init__(self):
        self.methods = {
            "repeated median hybrid filter": self.vec_repeated_median_hybrid_filters,
            "slope median filter " : self.slope_median_filter.
            "double filter" : self.double_filter
        }
        return

    #@jit
    #Vectorized version of RMF 3 times faster
    def vec_repeated_median_hybrid_filters(self, df, k=5):
        """
        Method for filter using repeated median hybrid filter, optimized version

        :param df: 1 dimension list of number in type pandas dataframe which need to be filtered.
        :type df: list.
        :param k: specified time window length used for repeated median filtering, should be adjusted /
        with respect to the frequency of the given data.
        :type k: int.
        """
        
        to_return = []
        N = len(data)
        for t in range(k, N - k):
            median_slope_list = []
            for i in range(-k, k):
                slope = [(data[t + i] - data[t + j]) / (i - j)
                         for j in range(-k, i)
                         ] + [(data[t + i] - data[t + j]) / (i - j)
                              for j in range(i + 1, k)]
                median_slope_list.append(slope)
            to_return.append(median_slope_list)
        temp = np.median(np.median(to_return, axis=2), axis=1)
        mu = np.median(
            [[data[t + j] - j * temp[t - k] for j in range(-k, k)]
             for t in range(k, N - k)],
            axis=1)
        return list(mu)

    #@profile
    def _repeated_median_hybrid_filter(self, data, k=5):
        """
        Method for filter using repeated median hybrid filter, before optimized

        :param data: 1 dimension list of number which need to be filtered.
        :type data: list.
        :param k: specified time window length used for repeated median filtering, should be adjusted /
        with respect to the frequency of the given data.
        :type k: int.
        """
        
        to_return = []
        N = len(data)
        for t in range(k, N - k):
            median_slope_list = []
            for i in range(-k, k):
                ''' compute the median of slope for each t+i '''
                slope = np.median([(data[t + i] - data[t + j]) / (i - j)
                                   for j in range(-k, k) if j != i])
                median_slope_list.append(slope)
            slope = np.median(median_slope_list)
            mu = np.median([data[t + j] - j * slope
                            for j in range(-k, k)])  #TODO: -j or j?
            mu_F = np.median(data[t - k:t])
            mu_B = np.median(data[t:t + k])
            to_return.append(np.median([mu_F, mu, mu_B]))
        return to_return

    #@profile
    def slope_median_filter(self, data, k=5, slope_threshold=20): 
        '''
         Method to return a list of indicator for abnormal slope points of the given data
        :param data: 1 dimension list of number which need to be filtered.
        :type data: list.
        :param k: specified time window length used for median filtering, should be adjusted /
        with respect to the frequency of the given data.
        :type k: int.
        :param slope_threshold: the absolute value threshold to report abnormal slope
        :type slope_threshold: float/int.

        '''
        
        to_return = []
        N = len(data)
        # the first k elements
        for t in range(k):
            median_slope_list = []
            for i in range(k):
                slope = np.median([(data[t + i] - data[t + j]) / (i - j)
                                   for j in range(k) if j != i])
                median_slope_list.append(slope)
            slope = np.median(median_slope_list)
            to_return.append(slope)
        # the elements in the middle
        for t in range(k, N - k):
            median_slope_list = []
            for i in range(-k, k):
                # compute the median of slope for each t+i
                slope = np.median([(data[t + i] - data[t + j]) / (i - j)
                                   for j in range(-k, k) if j != i])
                median_slope_list.append(slope)
            slope = np.median(median_slope_list)
            to_return.append(slope)
        # the last k elements
        for t in range(N - k, N):
            median_slope_list = []
            for i in range(-k, 0):
                slope = np.median([(data[t + i] - data[t + j]) / (i - j)
                                   for j in range(-k, 0) if j != i])
                median_slope_list.append(slope)
            slope = np.median(median_slope_list)
            to_return.append(slope)
        # compute the mean and standard deviation of median list
        mean = np.mean(to_return)
        std = np.std(to_return)
        # 1 stands for steady state points
        toreturn = [
            1 if abs(i - mean) < std and abs(i) < slope_threshold else 0
            for i in to_return
        ]
        return toreturn


    def double_filter(self, data, k1=5, k2=15):
        '''
         Method for filter using both median filter and repeated median hybrid filter
        :param data: 1 dimension list of number which need to be filtered.
        :type data: list.
        :param k1: specified time window length used for median filtering, should be adjusted /
        with respect to the frequency of the given data.
        :type k1: int.
        :param k2: specified time window length used for repeated median filtering, should be adjusted /
        with respect to the frequency of the given data.
        :type k2: int.

        '''
        
        filtered_data = sp.signal.medfilt(data, k1)
        filtered_data = list(
            filtered_data[:k2]) + self.vec_repeated_median_hybrid_filters(
                filtered_data, k2) + list(filtered_data[-k2:])
        #filtered_data = list(filtered_data[:k2])+self._repeated_median_hybrid_filter(filtered_data(RMF)
        return filtered_data