import numpy as np

# Temporarily put these two analytics tool function in the util file
def F1Score(predicted,real):
    if len(predicted)!=len(real):
        return None
    # compute the True positives
    TP = np.sum(np.array([1 if (predicted[i]>0 and real[i]>0) else 0 for i in range(len(predicted))]))
    # compute the False positives
    FP = np.sum(np.array([1 if (predicted[i]>0 and real[i]==0) else 0 for i in range(len(predicted))]))
    # compute the precision
    print TP,FP
    P = float(TP)/float(TP+FP)
    
    # compute the False Negatives
    FN = np.sum(np.array([1 if (predicted[i]==0 and real[i]>0) else 0 for i in range(len(predicted))]))
    # compute the True Negatives
    TN = np.sum(np.array([1 if (predicted[i]==0 and real[i]==0) else 0 for i in range(len(predicted))]))
    print FN,TN
    # compute the recall
    if TP==0 and FN==0:
        return P
    R = float(TP)/float(TP+FN) 

    # return the harmonic average of the precision and recall
    print P,R
    if P==0 and R==0:
        return 0
    else:
        return 2*P*R/(P+R)

def RMSE(predicted,real):
    """
    compute the Root mean square error per device
    """
    if len(predicted)!=len(real):
        return None
    else:
        return np.sqrt(np.mean(np.square(np.array(predicted)-np.array(real))))

def find_nearest(known_array, test_array):
    """Find closest value in `known_array` for each element in `test_array`.
    Parameters
    ----------
    known_array : numpy array
        consisting of scalar values only; shape: (m, 1)
    test_array : numpy array
        consisting of scalar values only; shape: (n, 1)
    Returns
    -------
    indices : numpy array; shape: (n, 1)
        For each value in `test_array` finds the index of the closest value
        in `known_array`.
    residuals : numpy array; shape: (n, 1)
        For each value in `test_array` finds the difference from the closest
        value in `known_array`.
    """
    # from http://stackoverflow.com/a/20785149/732596


    index_sorted = np.argsort(known_array)
    known_array_sorted = known_array[index_sorted]

    idx1 = np.searchsorted(known_array_sorted, test_array)
    idx2 = np.clip(idx1 - 1, 0, len(known_array_sorted)-1)
    idx3 = np.clip(idx1,     0, len(known_array_sorted)-1)

    diff1 = known_array_sorted[idx3] - test_array
    diff2 = test_array - known_array_sorted[idx2]

    indices = index_sorted[np.where(diff1 <= diff2, idx3, idx2)]
    residuals = test_array - known_array[indices]
    return indices, residuals