# coding=utf-8
from influxdb import DataFrameClient
import collections
import pandas as pd
import numpy as np

class DisaggData(object):

    def __init__(self,appliance_list = []):
        """
        self.data
            {
                "microwave":[1.0,2.0,...,9.0] numpy array
                "over"     :[3.0,0.0,...,8.0]     
            }
        self.appliance_list = ["2","3","4","6","11"] Name of each appliances
        """
        self.appliance_list = appliance_list
        self.data = collections.defaultdict()
        self.singleAppHistoryData = False

    def get_appliance_data(self):
        raise "Not implemented"

    def compute_total(self,appliance_list):
        raise "Not implemented"
        
    def get_data(self):
        return self.data

class ShenRuiData(DisaggData):

    def __init__(self):
        super(ShenRuiData,self).__init__(["JG196","FG157","G235","JG231","G7","G159","G342"])
        self.explanation = {
        "JG196":"空调1570W",
        "FG157":"热水器6000W",
        "G235":"高低温交变湿热湿试验箱5000W",
        "JG231":"美的空调2370W",
        "G7":"海尔空调4800W",
        "G159":"空调4600W",
        "G342":"大金空调4480W"
        }

        self.energy_usage = {
        "JG196":[0,1.57],
        "FG157":[0,6.0],
        "G235":[0,5.0],
        "JG231":[0,2.37],
        "G7":[0,4.8],
        "G159":[0,4.6],
        "G342":[0,4.48]
        }

    def get_appliance_data(self,filename = "~/Desktop/equota/申瑞6~8总进线.csv"):
        self.data = pd.read_csv(filename)
        return self.data

    def compute_total(self,appliance_list = None):
        return self.data['___P']
    

    # TODO: add plot result(generate pdf report) functionality and also generalize to other data

class ReddData(DisaggData):

    def __init__(self):
        super(ReddData,self).__init__(["3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20"])
        self.period = 1440
        self.singleAppHistoryData = True
        self.explanation = {
            "3":"oven1",
            "4":"oven2",
            "5":"refrigerator",
            "6":"dishwasher",
            "7":"kitchen_outlets1",
            "8":"kitchen_outlets1",
            "9":"lighting1",
            "10":"washer_dryer1",
            "11":"microwave",
            "12":"bathroom_gfi",
            "13":"electric_heat",
            "14":"stove",
            "15":"kitchen_outlets3",
            "16":"kitchen_outlets4",
            "17":"lighting2",
            "18":"lighting3",
            "19":"washer_dryer2",
            "20":"washer_dryer3"
        }
        
        for appliance in self.appliance_list:
            self.data[appliance] = []
        
        self.energy_usage = {
            "3" :[0,1685],
            "4" :[0,1730,2530],
            "5" :[6,195,425],
            "6" :[0,181,1100],
            "7" :[25], #(alway on)
            "8" :[20, 80],# (alway on)
            "9" :[0,81,146,340],
            "10" :[0,600,4500],
            "11" :[0,1550],
            "12" :[0,1600],
            "13" :[0,7,11,160],
            "14" :[0,1500],
            "15" :[0,1100],
            "16" :[0,1550],
            "17" :[0,65,72,100],
            "18" :[1,60,70],
            "19" :[0,2,20],
            "20" :[0,3050]
        }

    def readfile(self,f,Col): #read .csv files
        data=[]
        time=[]
        lines = f.readlines()
        #label=[]
        #head+=lines[0]
        for line in lines[1:]:
            line=line.strip('\n')
            line=line.split(',')
            tmp_time=float(line[0])
            tmp_data=0
            for i in range(len(Col)):
                tmp_data+=float(line[Col[i]])
                # TODO: fixme probably self.data[appliance] need to be timeseries?
                self.data[str(i+3)].append(float(line[Col[i]]))

            data.append(tmp_data),
            time.append(tmp_time),
        f.close()
        
        return (time,data) 

    def get_appliance_data(self,filename = None,filext = None,AppNo = None,N=1):
        """
        AppNo can be customized if only want to run the disaggregation on certain appliances
        for example:
            AppNo = [3,4,5]
        N: the disaggregation will be applied on Nth day's data 
        """
        if filename==None and filext==None:
            filename='/Users/Simo/Desktop/equota/disaggrREDD/house1_output15s'
            filext='.dat'
        f_input = file(filename+filext,'r')
        if AppNo==None:
            AppNo= self.appliance_list#Choose App#
        
        (t_all,y_all)=np.array(self.readfile(f_input,[int(i)-2 for i in AppNo]))
        t=np.array([i+1 for i in range(self.period)])
        y=y_all[N*self.period:(N+1)*self.period]
        
    def compute_total(self,appliance_list = None):
        if appliance_list == None:
            appliance_list = self.appliance_list
        
        power_dataframe = pd.concat([pd.TimeSeries(self.data[app]).fillna(method='ffill') for app in appliance_list], axis=1, keys=appliance_list).fillna(method = "ffill").resample("30S").pad().fillna(0)
        power_dataframe["total"] = power_dataframe.sum(axis = 1)
        
        return power_dataframe["total"]

class EquotaData(DisaggData):

    def __init__(self):
        
        super(EquotaData,self).__init__(["2","3","4","6","11"])
        
        self.singleAppHistoryData = True
        
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
            self.data[appliance] = client.query("select * from TS"+appliance+" order by time desc limit 2000")[u'TS'+appliance]
    
    def compute_total(self,appliance_list = None):    
        if appliance_list == None:
            appliance_list = self.appliance_list
        
        power_dataframe = pd.concat([self.data[app]["P"].fillna(method='ffill') for app in appliance_list], axis=1, keys=appliance_list).fillna(method = "ffill").resample("30S").pad().fillna(0)
        power_dataframe["total"] = power_dataframe.sum(axis = 1)
        
        return power_dataframe["total"]
