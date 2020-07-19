import numpy as np

def expected_value(series):
    """
    Expected value of a binary random variable
    
    Calculated as:
    E(X) = P(X) * n(X) + P(^X) * n(^X)
    
    with ^X the event with binary value 0.
    
    Returns:
    - float    
    """
    
    # probability of the binary event X ( = 1)
    prob = series.sum() / len(series)
    
    # expected value of the random variable
    return prob * series.sum() + (1 - prob) * (len(series) - series.sum())


def entropy(series):
    """    
    Entropy of a binary random variable
    
    Calculated as:
    H(X) = -(P(X) * log(P(X)) + P(^X) * log(P(^X)))
    
    with ^X the event with binary value 0.
    
    Returns:
    - float    
    """
    prob = series.sum() / len(series)
    
    return  - (prob * np.log2(prob) + (1 - prob) * np.log2(1 - prob))



def greedy_search(item, candidates, metric_computer):
    """
    Greedy search of the item among the list of candidate items.
    
    Iterate exclude candidates until the remaining list of candidates has only 1 item left.
    
    At each iteration, evaluate the provided metric across all the features of the candidate items.
    Choose the feature to evaluate that minimizes the metric.
    
    An example of a metric is the expected value of each feature.
    
    Returns:
    - solution: pd.Series, the candidate item that the algorithm has found
    - count: int, the number of iterations the algorithm required to find the solution
    """
    
    print("Using {} as metric to minimize!".format(metric_computer.__name__))
    
    df = candidates.copy()
    
    # number of questions asked
    count = 0
    
    while len(df) > 1:
        # search for feature with lowest expected value
        expected_values = df.apply(metric_computer, axis=0)
        
        idx = expected_values.argmin()
        pivot = expected_values.index.values[idx]

        # check feature against item
        pivot_value = item[pivot]

        print("pivot: {}".format(pivot))

        df = df[df[pivot] == pivot_value]
        
        count += 1
        
        print("Remaining candidates: {}".format(len(df)))
        
        # safety valve
        if count > 10:
            break
    
    return df.squeeze(), count


def binary_search(item, sorted_candidates):
    """
    Binary Search for the item among a sorted list of candidate items.
    
    Returns:
    - solution: string, name of character
    - count: int, number of iterations it took to reach solution
    """
    
    candidates = sorted_candidates.copy()
    
    # number of questions asked
    count = 0
    
    while len(candidates) > 1:

        middle_index = int(len(candidates) / 2)
        print('middle index: {}'.format(middle_index))

        pivot = middle_index
        left_group = candidates[:pivot]
        
        print('Is item in group [{}, {}]?'.format(left_group[0], left_group[-1]))

        if item in left_group:
            candidates = left_group
            print('yes!')
        else:
            candidates = candidates[pivot:]
            print('no!')
            
        count += 1
        
        # safety valve
        if count > 10:
            break
            
        print('remaining candidates: {}'.format(candidates))
        
    return candidates[0], count