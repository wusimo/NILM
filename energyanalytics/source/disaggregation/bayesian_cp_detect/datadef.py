import time

#Mon-Sun Labels [1,7]
#time.strftime("%Y/%m/%d %H:%M:%S %a", time.gmtime(timestamp))
def getDate(timestamp):
    tm_wday=time.gmtime(timestamp)[6]+1
    tm_yday=time.gmtime(timestamp)[7]
    return (tm_wday, tm_yday)

def wdayformat(tm_wday):
    if tm_wday==1:
        strwday='Mon'
    elif tm_wday==2:
        strwday='Tue'
    elif tm_wday==3:
        strwday='Wed'
    elif tm_wday==4:
        strwday='Thu'
    elif tm_wday==5:
        strwday='Fri'
    elif tm_wday==6:
        strwday='Sat'
    elif tm_wday==7:
        strwday='Sun'            
    return strwday


# value_day: daily accumulated data
# tm_wday: Mon-Sun [1,7]
# tm_yday: #Day of year [1,366]
def dataread(timestamp,data):
    #Daily data
    value_day=[]
    #Label Mon-Sun
    tm_wday=[]
    tm_yday=[]

    tmp_value=data[0]

    for i in range(1,len(data)):
        if getDate(timestamp[i])[1]==getDate(timestamp[i-1])[1]:
            tmp_value+=data[i]
            update=0
        else:
            value_day.append(tmp_value)
            tm_wday.append(getDate(timestamp[i-1])[0])
            tm_yday.append(getDate(timestamp[i-1])[1])
            tmp_value=data[i]
            update=1

    if update==0:
        value_day.append(tmp_value)
        tm_wday.append(getDate(timestamp[i])[0])
        tm_yday.append(getDate(timestamp[i])[1])
    return (value_day, tm_wday, tm_yday)

'''Determine which wday'''
def data_wday(value_day,tm_yday,tm_wday,i_wday):
    idx=[]
    for index, item in enumerate(tm_wday):
        if item==i_wday:
            idx.append(index)
    value_sub=[value_day[i] for i in idx]
    yday_sub=[tm_yday[i] for i in idx]
    return (yday_sub,value_sub)
