from __future__ import division
import numpy as np
from outlierdef import *
from datadef import *
from changepoint_detection import cp_detect
import matplotlib.pyplot as plt
import copy

#version 1.0
def cartesian(arrays):
    arrays = [np.asarray(a) for a in arrays]
    shape = (len(x) for x in arrays)

    ix = np.indices(shape, dtype=int)
    ix = ix.reshape(len(arrays), -1).T

    for n, arr in enumerate(arrays):
        ix[:, n] = arrays[n][ix[:, n]]

    return ix

def viterbi(data_event_mean, delta_event, status, power, freq_conversion, conditions, num_change):
    N=len(data_event_mean)
    num_app=len(status)
    
    ########
    #define range of power changeable
    id_conversion=[]
    if sum(freq_conversion)>0:
        id_conversion=[i for i, item in enumerate(freq_conversion) if item==1][0]
        conversion_pwr_min=power[id_conversion][1]
        conversion_pwr_max=power[id_conversion][-1]
    #conversion_pwr_range=conversion_pwr_max-conversion_pwr_min
    
    #detect Frequency Conversion Equipment  
    conversion_event_idx=[]
    if sum(freq_conversion)>0:
        for i in range(N):
            if delta_event[i]>=conversion_pwr_min*0.85:
                if delta_event[i]<=conversion_pwr_max*1.05:
                    conversion_event_idx.append(i)    
    
    delta_event_mean=np.zeros(N)    
    for i in range(1,N):
        delta_event_mean[i]=data_event_mean[i]-data_event_mean[i-1]
        #delta_event_mean[i]=delta_event[i]

    #possible status for each event
    s=cartesian(status)
    num=len(s)
    del_i=[]
    for i in range(num):
        for j in range(num_app):
            if s[i][j]<conditions[j]:
                del_i.append(i)
                break
    del_i=np.asarray(del_i)
    cond_i=np.setdiff1d(np.linspace(0, num-1, num).astype(int), del_i)
    
    
    #possible status change sub_s
    sub_s=[[]]
    for i in range(num):
        for i_change in range(1,num_change+1):        
            for j in range(num):
                if (s[j]==s[i]).sum()==num_app-i_change:
                    sub_s[i].append(j)
        if i<num-1:
            sub_s.append([])
    
    sub_s=np.asarray(sub_s, dtype=np.int32)
    
    #Event delta
    #Initialize error matrix
    err_matrix=np.zeros((num,N))
    
    #Power of Conversion Equipment=delta or Min
    conversion_event_count=0
    for j in range(N):
        if j in conversion_event_idx:
            power_sub=delta_event_mean[conversion_event_idx[conversion_event_count]]
            power_matrix=power_dynamic(power, power_sub, id_conversion)
            conversion_event_count+=1
        else:
            power_matrix=copy.deepcopy(power)
        
        for i in range(num):
            err_matrix[i][j]=-data_event_mean[j]
            for num_app_i in range(num_app):
                idx=s[i][num_app_i]
                err_matrix[i][j]+=power[num_app_i][idx]    
    
                
    
    #Apply conditions
    for j in range(0,N):
        for i in range(num):    
            if (del_i==i).sum()==1:
                err_matrix[i,j]=np.inf
    
    
    #Apply power_dynamic
    chain=np.ones((num,N),dtype=np.int32)*-1
    for j in range(1,N):
        for i in cond_i:
            idx=np.argmin(abs(err_matrix[sub_s[i],j-1]))
            min_i=sub_s[i][idx]
            chain[i][j-1]=min_i
            err_matrix[i][j]=abs(err_matrix[min_i][j-1])+abs(err_matrix[i][j])
    
            
    min_i=np.argmin(abs(err_matrix[:,N-1]))
    chain[min_i][N-1]=min_i
        
    viterbi_chain=np.zeros(N,dtype=np.int32)
    viterbi_chain[N-1]=min_i
    
    for j in range(N-1):
        j=N-2-j
        viterbi_chain[j]=chain[viterbi_chain[j+1]][j]
        
    viterbi_status=s[viterbi_chain]
    
    viterbi_errors=np.zeros(N)
    conversion_event_count=0
    for j in range(N):
        if j in conversion_event_idx:
            power_sub=delta_event_mean[conversion_event_idx[conversion_event_count]]
            power_matrix=power_dynamic(power, power_sub, id_conversion)
            conversion_event_count+=1
        else:
            power_matrix=copy.deepcopy(power)
            
        viterbi_errors[j]=-data_event_mean[j]
        for num_app_i in range(num_app):
            idx=viterbi_status[j][num_app_i]
            viterbi_errors[j]+=power_matrix[num_app_i][idx]

    return (viterbi_status, viterbi_errors, delta_event_mean, id_conversion, conversion_event_idx)

def power_dynamic(power, power_sub, id_conversion):
    power_dynamic=copy.deepcopy(power)
    
    #Power of Conversion Equipment=delta
    idx=np.argmin(abs(power[id_conversion]-power_sub*np.ones(len(power[id_conversion]))))
    power_dynamic[id_conversion][idx]=power_sub
    
    return power_dynamic


def find_high(y):
    ydiff=np.diff(np.diff(y))
    yhigh=y[0]
    for i in range(0,len(ydiff)):
        if ydiff[i]>0:
            yhigh=y[i]
            if abs(ydiff[i])<=0.25*abs(ydiff[0]):
                break
        else:
            yhigh=y[i]
            break
    return i,yhigh

def find_low(y):
    y_reverse=y[::-1]
    ydiff=np.diff(y_reverse)
    ylow=y_reverse[0]
    for i in range(0,len(ydiff)):
        if ydiff[i]<0:
            ylow=y_reverse[i]
            if abs(ydiff[i])<=0.25*abs(ydiff[0]):
                break
        else:
            ylow=y_reverse[i]
            break
    return i,ylow

def stat_power(viterbi_status, id_conversion, conversion_event_idx, delta_event_mean, power, start_idx, end_idx, t, freq):
    num_app=len(power)
    N=len(viterbi_status)
    time_p=np.zeros((N,num_app),dtype=np.int32)
    stat_p=np.zeros(num_app,dtype=np.int32)
    '''
    for i in range(N):
        for j in range(num_app):
            status=viterbi_status[i][j]
            time_p[i][j]=power[j][status]
            stat_p[j]+=power[j][status]*(t[end_idx[i]]-t[start_idx[i]])/freq #kwh
    '''
    conversion_event_count=0
    for i in range(N):
        if i in conversion_event_idx:
            power_sub=delta_event_mean[conversion_event_idx[conversion_event_count]]
            power_matrix=power_dynamic(power, power_sub, id_conversion)
            conversion_event_count+=1
        else:
            power_matrix=copy.deepcopy(power)
        for j in range(num_app):
            status=viterbi_status[i][j]
            time_p[i][j]=power_matrix[j][status]
            stat_p[j]+=power_matrix[j][status]*(t[end_idx[i]]-t[start_idx[i]])/freq #kwh
            
    return (time_p, stat_p)
    
    
def disaggregation(filename,t,y,status,power,freq_conversion,conditions,smooth,sampling_interval=60,num_change=1):
    sampling_rate=3600/sampling_interval   
    #delete obvious outliers y=0
    N=len(y)
    y=preprocess(y)
    
    #rlowess filter
    fraction=20/float(N)
    try:
        y_filtered=rlowess(y, fraction, iter=3)
    except:
        y_filtered = s_filter(y, wspan=45, polyorder=2)
    
    y_filtered_save=copy.deepcopy(y_filtered)

    if smooth==0:
        y_filtered=y##

    plt.ion()
    plt.show()

    plt.plot(t,y,'g.--')
    plt.plot(t,y_filtered,'b')
    plt.show()
    plt.pause(0.001)

    #change point detection
    print "Change point detection"
    timestart = time.clock()
    
    print "Determine the start of each event"
    (idxpcp_start, pcpsum)=cp_detect(y_filtered[::-1])
    idxpcp_start=N-1-idxpcp_start[::-1]
    idxpcp_start=np.insert(idxpcp_start,0,0)

    print "Determine the end of each event"
    (idxpcp_end, pcpsum)=cp_detect(y_filtered)
    idxpcp_end=np.insert(idxpcp_end,len(idxpcp_end),N-1)
    
    timend = time.clock()
    print "Processing Time: %f s" % (timend - timestart)
    
    plt.subplot(2, 1, 1)
    #plt.plot(t,y,'g.--')
    plt.plot(t,y_filtered,'b.-')
    plt.subplot(2, 1, 2)
    plt.plot(t,pcpsum)
    plt.show()
    
    #Determine event for viterbi
    N_event=len(idxpcp_start)
    #mean value for each event
    value_event_mean=np.zeros(N_event)
    #delta between each event
    delta_event=np.zeros(N_event)
    yhigh=np.zeros(N_event)
    ylow=np.zeros(N_event)
    for i in range(N_event):
        datafilter_event=y_filtered_save[idxpcp_start[i]:idxpcp_end[i]]
        offseth,yhigh[i]=find_high(datafilter_event)
        offsetl,ylow[i]=find_low(datafilter_event)
        
        if sampling_interval<50:           
            if i>=1:
                idxpcp_start[i]+=offseth
            if i<N_event-1:
                idxpcp_end[i]-=offsetl
                
        if idxpcp_start[i]>=idxpcp_end[i]:
            value_event_mean[i]=y_filtered[idxpcp_start[i]]
        else:
            value_event_mean[i]=np.mean(y_filtered[idxpcp_start[i]:idxpcp_end[i]])
        if i>=1:
            if value_event_mean[i]-value_event_mean[i-1]>=0:
                #delta_event[i]=y_filtered[idxpcp_start[i]]-y_filtered[idxpcp_end[i-1]]            
                #delta_event[i]=value_event_mean[i]-value_event_mean[i-1]
                delta_event[i]=yhigh[i]-ylow[i-1]
            else:
                delta_event[i]=y_filtered[idxpcp_start[i]]-y_filtered[idxpcp_end[i-1]]
    #Viterbi disaggregation
    '''
        stat_p: Power Consumption(kwh) for each appliance for whole period
        time_p: Power status(kw) for each appliance for each event
    '''
    (viterbi_status, viterbi_errors, delta_event_mean, id_conversion, conversion_event_idx)=viterbi(value_event_mean, delta_event, status, power, freq_conversion,conditions, num_change)
    (time_p, stat_p)=stat_power(viterbi_status, id_conversion, conversion_event_idx, delta_event_mean, power, idxpcp_start, idxpcp_end, t, sampling_rate)
    
    # value_event_err: error compensation
    time_p_err=np.zeros((N_event,len(power)),dtype=np.int32)
    stat_p_err=np.zeros(len(power),dtype=np.int32)


    for i in range(N_event):
        if value_event_mean[i]+viterbi_errors[i]==0:
            err_ratio=1
        else:
            err_ratio=value_event_mean[i]/(value_event_mean[i]+viterbi_errors[i])
        time_p_err[i]=time_p[i]*err_ratio
        
        for j in range(len(power)):
            status=viterbi_status[i][j]
            stat_p_err[j]+=err_ratio*power[j][status]*(t[idxpcp_end[i]]-t[idxpcp_start[i]])/sampling_rate #kwh    
    # total kwh with error compensation
    total_p_err=sum(stat_p_err)
    
    
    ##PLOT##
    t_plot=np.vstack((t[idxpcp_start],t[idxpcp_end])).ravel([-1])
    y_plot=np.zeros((len(power),2*N_event),dtype=np.int32)
    for j in range(len(power)):
        y_plot[j]=np.vstack((time_p_err[:,j],time_p_err[:,j])).ravel([-1])
    
    for idx in conversion_event_idx:
        y_plot[id_conversion][idx*2]=time_p_err[idx][id_conversion]+y_filtered[idxpcp_start[idx]]-value_event_mean[idx]
        y_plot[id_conversion][idx*2+1]=time_p_err[idx][id_conversion]+y_filtered[idxpcp_end[idx]]-value_event_mean[idx]        

    plt.plot(t,y_filtered,color='black',label='Total')
    for j in range(len(power)):
        if j==0:
            colortype='red'
            labelname='Pump'
        elif j==1:
            colortype='blue'
            labelname='HVAC'
        #elif j==2:
           # colortype='green'
            #labelname='Fan'
        else:
            colortype='yellow'
            labelname='Others'
        plt.plot(t_plot,y_plot[j],color=colortype,label=labelname)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('disaggregation: '+filename)     
    plt.xlabel('Time ('+str(sampling_interval)+'s interval)')
    plt.ylabel('Power Consumption (kwh)')
    plt.tight_layout()
    plt.savefig(filename+'.tif')
    plt.show()
    
    print 'Power Consumption:\n'
    print ('Total: '+str(total_p_err)+'kwh')
    for j in range(len(power)):
        print ('Appliance '+str(j)+': '+str(int((stat_p_err[j])/float(total_p_err)*1000)/10)+'%, '+str(stat_p_err[j])+'kwh')

    
    return stat_p_err,delta_event, y_filtered, idxpcp_start, idxpcp_end, value_event_mean,viterbi_status
    
if __name__ == "__main__":
    main()
