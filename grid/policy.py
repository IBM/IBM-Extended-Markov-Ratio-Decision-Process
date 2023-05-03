from typing import Tuple

import pickle
import numpy as np
import pandas as pd
from numpy import linalg as la

from grid.utils import sample_standard_simplex

def get_state(i: int, j: int, m: int, n: int) -> int:
    '''
    Get state index from cell indices. 

            Parameters:
                    i (int): cell grid row index.
                    j (int): cell grid column index.
                    m (int): number of rows in grid.
                    n (int): number of columns in grid.
                    
            Returns:
                    cell index.

    >>> get_state(1,2,3,5)
    7
    '''    
        
    assert all([i>=0,j>=0,i<m,j<n]), \
    'Illegal indices.'    

    return i*n+j

def next_state(states: list, mu: np.array=np.array([]), size: int=1) -> np.array:
    '''
    Sample state from state distribution. 

            Parameters:
                    states (list): state space.
                    mu (np.array): state distribution (default: np.array([]) -- sample uniformly).
                    size (int): number of states to sample (default: 1).
                    
            Returns:
                    numpy array of sampled state indices.

    >>> next_state(states,state_dist)
    array([2])
    ''' 

    if mu.size > 0:

        return np.random.choice(states,size=size,replace=True,p=mu)
    else:

        return np.random.choice(states,size=size,replace=True)

def get_cell_i(x: int, m: int, n: int) -> int:
    '''
    Get cell row coordinate from state index.

            Parameters:
                    x (int): state index.
                    m (int): number of rows in grid.
                    n (int): number of columns in grid.
                    
            Returns:
                    cell row index.

    >>> get_state(7,3,5)
    1
    '''        

    return int((x % (m*n)) / n)

def get_cell_j(x: int, m: int, n: int) -> int:
    '''
    Get cell column coordinate from state index.

            Parameters:
                    x (int): state index.
                    m (int): number of rows in grid.
                    n (int): number of columns in grid.
                    
            Returns:
                    cell column index.

    >>> get_state(7,3,5)
    2
    '''        

    return (x % (m*n)) % n

def state_tuple(s: int, m: int, n: int):
    '''
    Parse state index into (cell row, cell column) tuple.

            Parameters:
                    s (int): state index.
                    m (int): number of rows in grid.
                    n (int): number of columns in grid.
                    
            Returns:
                    state cell tuple.

    >>> state_tuple(5,3,3)
    (1, 2)
    '''        
    
    return (get_cell_i(s,m,n),get_cell_j(s,m,n))

def state_string(s: int, m: int, n: int):
    '''
    Parse state index into cell ij string.

            Parameters:
                    s (int): state index.
                    m (int): number of rows in grid.
                    n (int): number of columns in grid.
                    
            Returns:
                    state ij string.

    >>> state_tuple(5,3,3)
    '12'
    '''        
    
    return str(get_cell_i(s,m,n))+ str(get_cell_j(s,m,n))

def get_policy_action(policy: dict, state: int, actions: list, size: int=1) -> np.array:
    '''
    Get next action for given state from policy. 

            Parameters:
                    policy (dict): policy dictionary that matches to each state index a distribution over actions.
                    state (int): state index.
                    actions (list): action space.
                    size (int): number of actions to sample (default: 1).
                    
            Returns:
                    numpy array of sampled actions.

    >>> get_policy_action(p,3,actions)
    array(['R'], dtype='<U1')
    ''' 
    
    return np.random.choice(actions, size=size, replace=True, p=policy[state])

def get_policy(rho: np.array, states: list, actions: list) -> dict:
    '''
    Extract policy dictionary from occupation measure array rho. 

            Parameters:
                    rho (np.array): the occupation measure array of a policy.
                    states (list): state space.
                    actions (list): action space.
                    
            Returns:
                    policy dictionary that matches to each state index a distribution over actions.

    >>> get_policy(rho,states,actions)
    {0: array([0., 1., 0., 0., 0.]),
    ...
    8: array([1., 0., 0., 0., 0.])}
    ''' 
    
    state_num = len(states)
    action_num = len(actions)    
    
    assert rho.size == state_num*action_num, \
    'Could not produce sorted risk dict due to input size mismatch: state-action space size != state_num*action_num.'    
    
    p = rho.reshape(action_num,state_num) / rho.reshape(action_num,state_num).sum(axis=0)
    
    return {s: p[:,states[s]] for s in range(state_num)}

def sample_policy(states: list, actions: list, is_deterministic: bool=True, beta: float=1.0) -> dict:
    '''
    Sample policy. 

            Parameters:
                    states (list): state space.
                    actions (list): action space.
                    is_deterministic (bool): sample a deterinistic policy or not. Default: True.
                    beta (float): factor to downweight 'do nothing' policy. Default: 1.0.
                    
            Returns:
                    dictionary that matches a distribution over actions to a state.

    >>> sample_policy(beta=0.1)
    {0: array([0., 0., 0., 1., 0.]),
        1: array([1., 0., 0., 0., 0.]),
    ...,
        19: array([0., 1., 0., 0., 0.])}
    '''    
    
    a = len(actions)
    A = [*range(a)]
    
    assert a>1, 'Policy sampler expects at least two distinct actions.'
    
    weights = np.hstack([(1-beta/a)*np.ones(a-1)/(a-1),beta/a*np.ones(1)])

    if is_deterministic:
        
        p = {s: np.eye(1,a,np.random.choice(A,p=weights)).flatten() for s in states}
        
    else:

        p = {s: sample_standard_simplex(a) for s in states}
        
    return p

def policy_data(policy: dict, init_state: int, horizon: int, 
                states: list, actions: list, m: int, n: int,
                P: np.array, r: np.array, d: np.array) -> pd.DataFrame:
    '''
    Generate data from policy. 

            Parameters:
                    policy (dict): policy dictionary that matches to each state index a distribution over actions.
                    init_state (int): index of initial state.
                    horizon (int): size of data set.
                    actions (list): action space.
                    m (int): number of rows in grid.
                    n (int): number of columns in grid.
                    P (np.array): stochastic probability matrix [P(a1)|...|P(ak)].
                    r (np.array): expected reward vector. 
                    d (np.array): expected risk vector. 
                    
            Returns:
                    dataframe of data generated from given policy.

    >>> policy_data(p,init_state,T,actions,m,n,P,r,d)
     	state_i 	state_j 	action 	reward 	    risk
        0 	        0 	        4 	    -0.932472 	-2.011521
        0 	        0 	        3 	    -0.990383 	-0.998347
        0 	        1 	        3 	    -0.972179 	-1.006369
        0 	        1 	        2 	    -0.977767 	-0.999192
        0 	        0 	        2 	    -0.991093 	-2.002616
    '''    
    
    s = init_state
    ls = []

    for _ in range(horizon):

        action = get_policy_action(policy,s,actions)[0]
        ind = actions.index(action)
        A = P[:,m*n*ind:m*n*(ind+1)]
        snext = next_state(states,A[:,s])

        reward = r[m*n*ind:m*n*(ind+1)][s][0]
        risk = d[m*n*ind:m*n*(ind+1)][s][0]

        ls.append([get_cell_i(s,m,n),get_cell_j(s,m,n),ind,reward,risk])
        s = snext[0]

    df = pd.DataFrame(ls,columns=['state_i','state_j','action','reward','risk'])
    
    return df

def occupation_measure(policy: dict, mu: np.array, discount: float, P: np.array, tol: float=1e-8) -> np.array:
    '''
    Calculate the occupation measure of a given policy. Occupation measure is guaranteed to converge for 0 < discount factor < 1.

            Parameters:
                    policy (dict): policy dictionary that matches to each state index a distribution over actions.
                    mu (np.array): state distribution (default: np.array([]) -- sample uniformly).
                    discount (float): discount factor in (0,1).                    
                    P (np.array): stochastic probability matrix [P(a1)|...|P(ak)].
                    tol (float): tolerance for near-zero values (default 1e-8)
                    
            Returns:
                    occupation measure of the given policy.

    '''    
    
    assert (discount < 1) and (discount > 0), \
    'Illegal discount factor. Discount factor must be in (0,1), instead, received {}.'.format(discount)
    
    action_nums = {arr.size for arr in policy.values()}
    
    assert len(action_nums) == 1, \
    'Illegal policy. Action array must be of fixed length for all states.'

    assert len(policy.keys()) == mu.size, \
    'The state space size derived from the given policy ({}) must match that same size derived from the initial state distribution ({}).'.format(len(policy.keys()),mu.size)
    
    assert len(policy.keys())*len(policy[0]) == P.shape[-1], \
    'The state-action space size derived from the given policy ({}x{}={}) must match that same size derived from the transition probability matrix P (cols={}).'.format(len(policy.keys()),len(policy[0]),len(policy.keys())*len(policy[0]),P.shape[-1])
    
    states = list(policy.keys())
    state_num = len(states)
    action_num = list(action_nums)[0]
    
    rho = np.zeros(state_num*action_num)
    Prho = np.zeros(state_num*action_num)

    t = 0

    for s in range(state_num):
        for a in range(action_num):
            Prho[state_num*a+s] = policy[states[s]][a]*mu[s]

    rho = rho + (1-discount)*discount**t*Prho

    while True:
        t += 1
        Ptmp = Prho.copy()
        rhotmp = rho.copy()
        for s in range(state_num):
            for a in range(action_num):
                Prho[state_num*a+s] = policy[states[s]][a]*(P[s,:]@(Ptmp[:,None]))
        rho = rhotmp + (1-discount)*discount**t*Prho    
        if np.all(np.isclose(rho-rhotmp,0.0,atol=tol)):
            return rho

def occupation_measure_sanity(rho: np.array, A: np.array, mu: np.array, discount: float, report: dict, tol: float=1e-8):
    '''
    Sanity checks to make sure rho vector is an occupation measure.
    
            Parameters:
                    rho (np.array): candidate occupation measure vector.
                    A (np.array): extended Q matrix [I-discount*P(a1)|...|I-discount*P(ak)] stacked with risk vector.
                    mu (np.array): initial state distribution.
                    discount (float): discount factor in (0,1).
                    report (dict): dictionary report for type of error.
                    tol (float): tolerance for near-zero values (default 1e-8)
                    
    '''
    
    oops = 'Errors accrue, but we encountered a snafu.'
    atol = abs(int(np.log10(tol)))-2
    
    if not np.all(np.around(rho,atol)>=0):
        report['type'] = 'OccupationMeasureNegative'
        raise RuntimeError(oops + f'\nOccupation measure must be positive. We have \nrho = {rho}')    
    
    if not np.isclose(rho.sum(),1,atol=tol*10):
        report['type'] = 'OccupationMeasureLessOne'
        raise RuntimeError(oops + f'\nOccupation measure must sum up to 1. We have \nrho = {rho}')    
    
    if not np.all(np.isclose((A@(rho[:,None]))[:-1].flatten(),(1-discount)*mu,atol=tol*10)):
        report['type'] = 'OccupationMeasureInitialStateDistribution'
        raise RuntimeError(oops + f'\nOccupation measure must produce initial state distribution. The l2 distance between them is = {la.norm((A@(rho[:,None]))[:-1].flatten()-(1-discount)*mu)}.')    
    
def get_ratio(rho: np.array, r: np.array, d: np.array, omega: float=1.) -> Tuple[float, float, float]:
    '''
    Calculate the reward, risk, and ratio of the given occupation measure.
    
            Parameters:
                    rho (np.array): candidate occupation measure vector.
                    r (np.array): expected reward vector. 
                    d (np.array): expected risk vector. 
                    omega (float): risk exponent (default 1.0)

            Returns:
                    expected reward.
                    expected risk.
                    ratio.
                    
    '''
        
    reward = (rho@r)[0]
    risk = (rho@d)[0]
    ratio = abs(reward) / (abs(risk)**omega)
    
    return reward, risk, ratio

def support(rho: np.array) -> np.array:
    '''
    Calculate the support of a policy given in the form of its occupation measure.
    
            Parameters:
                    rho (np.array): the occupation measure vector of a policy.
                    
            Returns:
                    Boolean vector with True indicating (state,action) index in support and False otherwise.
                    
    '''                
    
    scale = np.floor(np.floor(np.log10(rho[rho>0])).mean())-1
    
    return rho > 10**scale
        
def extract_alg_policy(p_path, action_num: int) -> dict:
    '''
    Extract policy from pickle file. Size of state space is derived from pickled policy.
    
            Parameters:
                    p_path (path-like): absolute path to policy pickle file.
                    action_num (int): number of actions in action space.
                    
            Returns:
                    dictionary that matches to each state index a distribution over actions.

    >>> extract_policy(p_path)
    {0: array([0., 0., 0., 1., 0.]),
     1: array([1., 0., 0., 0., 0.]),
    ...,
     19: array([0., 1., 0., 0., 0.])}
    '''    
        
    with open(p_path, 'rb') as handle:
        pdict = pickle.load(handle)
    
    d = {(int(s[0]),int(s[1])): int(a) for s,a in pdict.items()}
    
    m = max([k[0] for k in d.keys()])+1
    n = max([k[1] for k in d.keys()])+1
    
    assert len(d.keys()) == m*n, \
    'Incomplete states space in given policy dictionary. Expected {}x{}={} states. Received {} states.'.format(m,n,m*n,len(d.keys()))
    
    p = {n*k[0]+k[1]: np.eye(1,action_num,v).flatten() for k,v in d.items()}

    return p

def policy_noising(p: dict, noise: float, states: list, actions: list, mu: np.array, discount: float, P: np.array, tol: float=1e-8) -> dict:
    '''
    Add noise to a given policy by a specified precentage amount. Adding noise=0 will produce the given policy 
    while adding noise=1 will produce a completely random policy.
    
            Parameters:
                    p (dict): policy dictionary that matches to each state index a distribution over actions.
                    noise (float): percentage of noise in [0,1]. 
                    states (list): state space.
                    actions (list): action space.
                    mu (np.array): initial state distribution.                    
                    discount (float): discount factor in (0,1).
                    P (np.array): stochastic probability matrix [P(a1)|...|P(ak)].
                    tol (float): tolerance for near-zero values (default 1e-8).
                                        
            Returns:
                    noised policy dictionary that matches to each state index a distribution over actions.
    '''    

    rho = occupation_measure(p,mu,discount,P,tol)

    assert (noise >= 0.) and (noise <= 1.), \
    'Noise parameter must be in [0,1]. Instead, noise={}'.format(noise)

    s = sample_standard_simplex(rho.size)

    rho_noise = noise*s + (1-noise)*rho

    p_noise = get_policy(rho_noise, states, actions)
    
    return p_noise    