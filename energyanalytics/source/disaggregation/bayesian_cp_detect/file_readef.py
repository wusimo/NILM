import string
import copy
import numpy as np
from os import path

def readfile(date,Hstart,Hend,folder='IHG'): #read .csv files
    data=[]
    time=[]
    tmp_time=0
    for H in range(Hstart,Hend+1):
        filename= path.join(folder, 'TS143-2016-'+date+'-'+str(H)+'.csv')
        f = file(filename,'r')
        lines = f.readlines()

        for i in range(0,len(lines)):
            line=lines[i]
            line=line.strip('\n')
            line=line.split(',')
            tmp_time+=1
            tmp_data=float(line[10])*1000.0/3.0
            data.append(tmp_data),
            time.append(tmp_time),
        f.close()

    return (time,data)



