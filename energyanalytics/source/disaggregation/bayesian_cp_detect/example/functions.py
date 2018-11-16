

def integrated_clustering(t_all,y_all,num_of_days=500,period = 1440,trim=10,min_n_clusters = 4, max_n_clusters=10,hierarchical=0):
    
    
    """
    method for finding the change shape based on unsupervised learning and changepoint detection on history data 
    
    :param t_all: 1 dimension list of index of the history data used for unsupervised learning
    :type t_all: list
    :param y_all: 1 dimension list containing values of the history data used for unsupervised learning
    :type y_all: list
    :param num_of_days: length of history data used in unit of days
    :type num_of_days: int
    :param period: How many data points per day, in other words, the inverse of frequency of the given data 
    :type period: int
    :param min_n_clusters: a prior knowledge on minimum number of clusters wanted
    :type min_n_clusters: int
    :param min_n_clusters: a prior knowledge on maximum number of clusters wanted
    :type min_n_clusters: int

    """



    all_seg_april = initial_disaggregate(t_all,y_all,num_of_days,period = period)
    
    ''' '''
    all_seg_april_normalized = [np.array(x[0])-np.mean(x[1]) for x in all_seg_april if len(x[1])==3]
    
    ''' filter the empty segments'''
    all_seg_april_normalized = [x for x in all_seg_april_normalized if len(x)>0]
    
    ''' clustering in different ranges will probably have a better result'''
    if hierarchical == 0:
        pass
    elif hierarchical ==1:
        all_seg_april_normalized = [x for x in all_seg_april_normalized if x.mean()>1000]
    else:
        all_seg_april_normalized = [x for x in all_seg_april_normalized if x.mean()<1000]
    
    ''' filter out the positive segments'''
    all_positive_seg_april_normalized = [x for x in all_seg_april_normalized if x.min()>0]
    
    
    all_seg_april_normalized_trim50 = extract_first_n(all_positive_seg_april_normalized, trim)
    cluster_average = []
    
    # find optimal clustering number using silhouette score
    
    optimal_dict = {}
    
    for n_clusters in range(min_n_clusters,max_n_clusters):
        
        y_pred = KMeans(n_clusters=n_clusters).fit_predict(all_seg_april_normalized_trim50)

        cluster_average = []
        for i_cluster in range(n_clusters):
            cluster_average.append(
                np.mean([np.mean(x) for i, x in enumerate(all_seg_april_normalized_trim50) if y_pred[i]==i_cluster])
            ) 

        # sihouette score
        cluster_labels = y_pred
        sample_silhouette_values = silhouette_samples(all_seg_april_normalized_trim50, cluster_labels)
        
        silhouette_avg = silhouette_score(pd.DataFrame(all_seg_april_normalized_trim50), cluster_labels)

        optimal_dict[n_clusters] = silhouette_avg +(sample_silhouette_values.min()+sample_silhouette_values.max())/2
    
    # n_clusters will give us the optimal number of clusters
    n_clusters = max(optimal_dict.iteritems(), key=operator.itemgetter(1))[0]

    #print n_clusters
    
    y_pred = KMeans(n_clusters=n_clusters).fit_predict(all_seg_april_normalized_trim50)

    cluster_average = []
    
    for i_cluster in range(n_clusters):
        cluster_average.append(
            np.mean([np.mean(x) for i, x in enumerate(all_seg_april_normalized_trim50) if y_pred[i]==i_cluster])
        ) 
    cluster_average_rank = np.argsort(cluster_average)[::-1]
    rank_map = {cluster_average_rank[i_cluster]:i_cluster for i_cluster in range(n_clusters)} # old index:new index

    y_pred_old = y_pred
    y_pred = [rank_map[x] for x in y_pred]
    all_seg_per_cluster = [[] for i in range(n_clusters) ]
    for i_seg in range(len(all_seg_april_normalized_trim50)):
        all_seg_per_cluster[y_pred[i_seg]].append(all_seg_april_normalized_trim50[i_seg])
        
    cluster_mean = [[] for i in range(n_clusters) ]
    cluster_std = [[] for i in range(n_clusters) ]
    for i_cluster in range(n_clusters):
        cluster_mean[ i_cluster ] = np.mean(np.array(all_seg_per_cluster[i_cluster]), axis=0)
        cluster_std[ i_cluster ] = np.std(np.array(all_seg_per_cluster[i_cluster]), axis=0)
    
    
    
    
    #cluster_mean_2 = cluster_mean[5:6]
    
    return cluster_mean,cluster_std,n_clusters,all_seg_per_cluster