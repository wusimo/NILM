# coding=utf-8
from influxdb import DataFrameClient
import collections
import pandas as pd

class DisaggData(object):

	def __init__(self,appliance_list = []):
		"""
		appliance_list:
			{
				"microwave":[1.0,2.0,...,9.0] numpy array
				"over"     :[3.0,0.0,...,8.0] 	
			}
		"""
		self.appliance_list = appliance_list
		self.data = collections.defaultdict()

	def get_appliance_data(self):
		raise "Not implemented"

	def compute_total(self,appliance_list):
		raise "Not implemented"
		
	def get_data(self):
		return self.data

class ReddData(DisaggData):

	def __init__(self):
		super(ReddData,self).__init__([])
		self.explanation = {
		}

	def get_appliance_data(self):

		return 

class EquotaData(DisaggData):

	def __init__(self):
		
		super(EquotaData,self).__init__(["2","3","4","6","11"])
		
		self.explanation = {
			"2": "饮水机",
			"3": "冰箱"  ,
			"4": "打印机",
			"6": "投影仪",
			"11":"微波炉",

		}

	def get_appliance_data(self):
		
		client = DataFrameClient(
    		host = "120.132.6.207",
    		username = "svcuser",
    		password = "svcuser",
    		database = "weather"
		)
		
		for appliance in self.appliance_list:
			self.data[appliance] = client.query("select * from TS"+appliance+" order by time desc limit 100000")[u'TS'+appliance]
	
	def compute_total(self,appliance_list = None):	
		# TODO: fixme
		if appliance_list == None:
			appliance_list = self.appliance_list
		
		power_dataframe = pd.concat([self.data[app]["P"].fillna(method='ffill') for app in appliance_list], axis=1, keys=appliance_list).fillna(method = "ffill").resample("30S").pad().fillna(0)
		power_dataframe["total"] = power_dataframe.sum(axis = 1)
		
		return power_dataframe["total"]
