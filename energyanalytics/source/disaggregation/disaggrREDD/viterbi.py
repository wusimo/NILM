from __future__ import division
import numpy as np
from outlierdef import *
from datadef import *
from changepoint_detection import cp_detect
import matplotlib.pyplot as plt

#version 1.0
def cartesian(arrays):
    arrays = [np.asarray(a) for a in arrays]
    shape = (len(x) for x in arrays)

    ix = np.indices(shape, dtype=int)
    ix = ix.reshape(len(arrays), -1).T

    for n, arr in enumerate(arrays):
        ix[:, n] = arrays[n][ix[:, n]]

    return ix

def viterbi(data_event, status, power, conditions, num_change):
    N=len(data_event)
    num_app=len(status)
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
            
    #Create error matrix
    err_matrix=np.zeros((num,N))
    for i in range(num):
        for j in range(N):
            err_matrix[i][j]=-data_event[j]
            for num_app_i in range(num_app):
                idx=s[i][num_app_i]
                err_matrix[i][j]+=power[num_app_i][idx]

    #Apply conditions
    for j in range(0,N):
        for i in range(num):    
            if (del_i==i).sum()==1:
                err_matrix[i,j]=np.inf
                
    chain=np.ones((num,N),dtype=np.int32)*-1
    for j in range(1,N):
        for i in cond_i:
            idx=np.argmin(abs(err_matrix[sub_s[i],j-1]))
            min_i=sub_s[i][idx]
            chain[i][j-1]=min_i
            err_matrix[i][j]=abs(err_matrix[min_i][j-1])+abs(err_matrix[i][j])
            #err_matrix[i][j]+=err_matrix[min_i][j-1]
            #err_matrix[i][j]+=abs(err_matrix[min_i][j-1])
            '''if abs(err_matrix[i][j])<=abs(err_matrix[i][j]+err_matrix[min_i][j-1]):
                err_matrix[i][j]+=0
            else:
                err_matrix[i][j]+=err_matrix[min_i][j-1]'''
            
    min_i=np.argmin(abs(err_matrix[:,N-1]))
    chain[min_i][N-1]=min_i
        
    viterbi_chain=np.zeros(N,dtype=np.int32)
    viterbi_chain[N-1]=min_i
       
    for j in range(N-1):
        j=N-2-j
        viterbi_chain[j]=chain[viterbi_chain[j+1]][j]
        
    viterbi_status=s[viterbi_chain]
    viterbi_errors=np.zeros(N)
    for j in range(N):
        viterbi_errors[j]=-data_event[j]
        for num_app_i in range(num_app):
            idx=viterbi_status[j][num_app_i]
            viterbi_errors[j]+=power[num_app_i][idx]

    return (viterbi_status, viterbi_errors)


def stat_power(viterbi_status, power, start_idx, end_idx, t, freq):
    num=len(power)
    N=len(viterbi_status)
    time_p=np.zeros((N,num),dtype=np.int32)
    stat_p=np.zeros(num,dtype=np.int32)

    for i in range(N):
        for j in range(num):
            status=viterbi_status[i][j]
            time_p[i][j]=power[j][status]
            stat_p[j]+=power[j][status]*(t[end_idx[i]]-t[start_idx[i]])/freq #kwh
            
    return (time_p, stat_p)
    
    
def disaggregation(filename,t,y,status,power,conditions,sampling_rate,num_change=1):
    #delete obvious outliers y=0
    N=len(y)
    #y=preprocess(y)
    
    #rlowess filter
    fraction=20/float(N)
    y_filtered=y
    '''try:
        y_filtered=rlowess(y, fraction, iter=3)
    except:
        y_filtered = s_filter(y, wspan=45, polyorder=2)'''
    
    
    plt.plot(t,y,'g.--')
    plt.plot(t,y_filtered,'b')
    plt.show()
    
    
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
    plt.plot(t,y_filtered,'b')
    plt.subplot(2, 1, 2)
    plt.plot(t,pcpsum)
    plt.show()
    
    #Determine event for viterbi
    N_event=len(idxpcp_start)
    value_event=np.zeros(N_event)
    for i in range(N_event):
        value_event[i]=np.mean(y_filtered[idxpcp_start[i]:idxpcp_end[i]])
    
    #Viterbi disaggregation
    '''
        stat_p: Power Consumption(kwh) for each appliance for whole period
        time_p: Power status(kw) for each appliance for each event
    '''
    (viterbi_status, viterbi_errors)=viterbi(value_event, status, power, conditions, num_change)
    (time_p, stat_p)=stat_power(viterbi_status, power, idxpcp_start, idxpcp_end, t, sampling_rate)
        
    # value_event_err: error compensation
    time_p_err=np.zeros((N_event,len(power)),dtype=np.int32)
    stat_p_err=np.zeros(len(power),dtype=np.int32)
    
    for i in range(N_event):
        if (value_event[i]+viterbi_errors[i])==0:
            err_ratio=0
        else:              
            err_ratio=value_event[i]/(value_event[i]+viterbi_errors[i])
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
    

    plt.plot(t,y_filtered,color='black',label='Total')
    for j in range(len(power)):
        if j==0:
            colortype='red'
            labelname='App 1'
        elif j==1:
            colortype='blue'
            labelname='App 2'
        elif j==2:
            colortype='green'
            labelname='App 3'
        else:
            colortype='yellow'
            labelname='App 4+'
        plt.plot(t_plot,y_plot[j],color=colortype,label=labelname)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('disaggregation: '+filename)     
    plt.xlabel('Time (*15s)')
    plt.ylabel('Power Consumption (kwh)')   
    
    plt.savefig('disaggr_'+filename+'.tif',bbox_inches='tight')
    plt.show()
    
    print 'Power Consumption:\n'
    print ('Total: '+str(total_p_err)+'kwh')
    for j in range(len(power)):
        print ('Appliance '+str(j)+': '+str(int((stat_p_err[j])/float(total_p_err)*1000)/10)+'%, '+str(stat_p_err[j])+'kwh')

    
    return stat_p_err,t_plot,y_plot
    
if __name__ == "__main__":
    main()
