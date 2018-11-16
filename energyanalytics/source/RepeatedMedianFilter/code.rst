Documentation for the Steady State Detection Algorithm
******************************************************

Steady State Detection Algorithm
================================

The first step is pass the whole time series to a two step filtering procedure with two different time windows. This includes a median filter with small time window(preset at 5) to filter out the abnormal or potential outlier points and after that use a larger time window(preset at 15) to apply the repeated median filter to clean the data but at same time preserve the trends.

The second step of this algorithm is using BCP(Bayesian changepoint) to detect the changepoints of the given time series cleaned after step 1. And the segments between each of the consecutive changepoints will be considered as equilibrium states. We take the average value in each segments as the steady state value.

Explanation of the Repeated Median Filter Algorithm 
===================================================

.. automodule:: RepeatedMedianFilter.RMFilter


Explanation of the Changepoint Algorithm
========================================

.. automodule:: disaggregation.bayesian_cp_detect.cp_detect
    :members: bayesian_change_point_4

Explanation of the ChangepointAlgorithm will be found in the Disaggregation documentation 

