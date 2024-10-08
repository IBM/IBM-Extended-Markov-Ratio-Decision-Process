from typing import Tuple, Optional
import logging
import time
import numpy as np
from numpy import linalg as la
from scipy.linalg import LinAlgWarning
from scipy.optimize import OptimizeWarning
from scipy.optimize import linprog

from grid.policy import support, occupation_measure

import warnings
warnings.filterwarnings(action='ignore', category=LinAlgWarning, module='scipy')
warnings.filterwarnings(action='ignore', category=OptimizeWarning, module='scipy')
warnings.filterwarnings(action='ignore', category=RuntimeWarning, module='scipy')

def _min_risk(Q: np.array, mu: np.array, discount: float, risk: np.array, report: dict, tol: float=1e-8) -> np.array:
    '''
    Calculate occupation measure of minimal risk policy.
    
            Parameters:
                    Q (np.array): the Q matrix [I-discount*P(a1)|...|I-discount*P(ak)].
                    mu (np.array): initial state distribution.
                    discount (float): discount factor in (0,1).
                    risk (np.array): expected risk vector.
                    report (dict): dictionary to report type of error.
                    tol (np.array): tolerance for near-zero values (default 1e-8)
                    
            Returns:
                    Occupation measure of minimal risk policy (1 dim'l np.array).
                    
    '''            

    assert discount > 0 and discount < 1, 'Discount factor must be in (0,1).'

    c = risk.flatten()

    res = linprog(c, A_eq = Q, b_eq = mu*(1-discount), bounds = (0,None), options = {'cholesky': False, 'sym_pos': False, 'tol': tol})

    if not res['success']:
        report['type'] = 'MinPolicyFail'
        raise RuntimeError('Minimal risk policy not found.')
    
    atol = abs(int(np.log10(tol)))

    return np.around(res['x'], atol)

def _naive_min_risk(d: np.array, state_num: int, action_num: int,
                    mu: np.array, discount: float, P: np.array, tol: float=1e-8) -> dict:
    '''
    Calculate the naive minimal risk policy as a sanity check.
    
            Parameters:
                    d (np.array): expected risk vector.
                    state_num (list): size of state space.
                    action_num (list): size of action space.
                    mu (np.array): initial state distribution.
                    discount (float): discount factor in (0,1).
                    P (np.array): stochastic probability matrix [P(a1)|...|P(ak)].
                    tol (np.array): tolerance for near-zero values (default 1e-8)
                    
            Returns:
                    Occupation measure of minimal risk policy (1 dim'l np.array).
                    
    '''            

    a = np.zeros((action_num,state_num))
    ai = np.expand_dims(np.argmin(d.reshape(action_num,state_num),axis=0), axis=0)
    np.put_along_axis(a, ai, 1, axis=0)

    p = {s: a[:,s] for s in range(state_num)}
    rho = occupation_measure(p,mu,discount,P,tol=tol)

    return rho

def _max_risk(A: np.array, mu: np.array, lb: float, discount: float, report: dict, tol: float=1e-12) -> float:
    '''
    Maximize risk associated with the given optimal basis A.
    
            Parameters:
                    A (np.array): an optimal basis A.
                    mu (np.array): initial state distribution.
                    lb (float): lower bound for risk.
                    discount (float): discount factor in (0,1). 
                    report (dict): dictionary to report type of error.
                    tol (float): tolerance for near-zero values (default 1e-8)
                    
            Returns:
                    The maximal risk associated with a deterministic policy spanned by A.
                    
    '''            

    atol = abs(int(np.log10(tol)))
    # convert to longdouble to avoid numpy overflow warnings in multiply
    A = np.array(A, dtype=np.longdouble)
    mu = np.array(mu, dtype=np.longdouble)
    lb = np.longdouble(lb)
    tol = np.longdouble(tol)

    c = np.zeros(A.shape[0])
    c[-1] = 1.0
    coeff = -1.0 # linprog minimizes

    res = linprog(
        coeff*c,
        A_ub = -A, b_ub = np.zeros(A.shape[0]), 
        A_eq = np.identity(A.shape[0])[:-1,], b_eq = (1-discount)*mu, 
        bounds = (lb,None),
        options = {'cholesky': False, 'sym_pos': False, 'tol': tol, 'lstsq': True, 'presolve': True}
    )

    if not res['success']:
        report['type'] = 'MaxNextRiskFail'
        raise RuntimeError('Next risk not found for the given support index set I.')    

    return np.around(res['x'][-1],atol)

def _sorted_risk_index(risk: np.array, state_num: int, action_num: int, 
              state_action_space: np.array=np.array([])) -> np.array:    
    '''
    Produce an array of sorted risk index per state column. 
    State actions not in the given state action space will come last.
    
            Parameters:
                    risk (np.array): expected risk vector.
                    state_num (list): size of state space.
                    action_num (list): size of action space.
                    state_action_space (np.array): binary vector representing the state-action space, 
                        with 1 indicating (state,action) pair in space and 0 otherwise. Defaults to full
                        (state,action) combinations.
                    
            Returns:
                    Array of dimension (action_num,state_num).

    >>> sort_risk(np.array([2.,1.,4.,0.,2.,1.]),3,2)
    array([[1, 0, 1],
           [0, 1, 0]])
    '''    
    
    
    assert risk.size == state_num*action_num, \
    'Could not sort risk due to input dimension mismatch: risk size != state_num*action_num.'
    
    sas = state_action_space if state_action_space.size > 0 else np.ones(risk.size)
    
    risk_nan = risk.flatten()*np.where(sas,sas.astype(bool).astype(int),np.nan)
    risk_mat = risk_nan.reshape(action_num,state_num)
    
    return np.argsort(risk_mat,axis=0)

def _sorted_risk_dict(sorted_risk_ind: np.array, states: list, actions: list, 
                     state_action_space: np.array=np.array([])) -> dict:
    '''
    Order actions by risk per state. State-action pairs not in the given space will be removed. 
    
            Parameters:
                    sorted_risk_ind (np.array): array of sorted risk index per state column.                        
                    states (list): state space.
                    actions (list): action space.
                    state_action_space (np.array): binary vector representing the state-action space, 
                        with 1 indicating (state,action) pair in space and 0 otherwise. Defaults to full
                        (state,action) combinations.
                    
                    
            Returns:
                    Dictionary of actions ordered by risk per state.                    
    '''        

    state_num = len(states)
    action_num = len(actions)
    
    sas = state_action_space if state_action_space.size > 0 else np.ones(state_num*action_num)

    assert sas.size == sorted_risk_ind.size, \
    'Could not produce sorted risk dict due to input size mismatch: sorted_risk_ind size != state action space size.'
    
    assert sas.size == state_num*action_num, \
    'Could not produce sorted risk dict due to input size mismatch: state-action space size != state_num*action_num.'
    
    # sas_shaped = sas.reshape(action_num,state_num)
    # action_num_per_state = {s: int(sas_shaped[:,s].sum()) for s in range(state_num)}
    # sorted_risk_dict = {s: list(sorted_risk_ind[:,s])[action_num-action_num_per_state[s]:] for s in range(state_num)}
    # sorted_risk_dict = {states[s]: [actions[a] for a in sorted_risk_dict[s]] for s in range(state_num)}
    
    sorted_risk_dict = {states[s]: [actions[a] for a in sorted_risk_ind[:,s] if sas[state_num*a+s]] for s in range(state_num)}
    
    return sorted_risk_dict    

def _support_adjust(support: np.array, sorted_risk_ind: np.array,
                   state_num: int, action_num: list) -> np.array:
    '''
    Adjust policy support. If a (s,a) pair is in the support, then all (s,a') such that 
    risk(s,a')<=risk(s,a) will be added to support.
    
            Parameters:
                    support (np.array): binary vector with 1 indicating (state,action) index in support and 
                        0 otherwise.
                    sorted_risk_ind (np.array): array of sorted risk index per state column.                        
                    state_num (list): size of state space.
                    action_num (list): size of action space.
                    
            Returns:
                    Boolean vector with True indicating index in adjusted support and False otherwise.                    
    '''
    
    assert state_num*action_num == support.size, \
    'Could not adjust support due to input dimension mismatch: support size != state_num*action_num.'
    
    supp_sorted = {s: support.reshape(action_num,state_num)[sorted_risk_ind[:,s],s] for s in range(state_num)}
    supp_arr = np.array(list(supp_sorted.values()))
    supp_cum = {s: (np.cumsum(supp_arr[:,::-1],axis=1)[:,::-1][s,:]).astype(bool) for s in range(state_num)}
    supp_unsort = {s: supp_cum[s][np.argsort(sorted_risk_ind[:,s])] for s in range(state_num)}
    supp = np.array(list(supp_unsort.values())).T.flatten()
        
    return supp

def _pair_support(s: int, a: str, states: list, actions: list) -> np.array:
    '''
    Produce the support of a single (state,action) pair as a binary vector with True at the (state,action) index and False elsewhere.
    
            Parameters:
                    s: a single state.
                    a: a single action.
                    states (list): state space.
                    actions (list): action space.
                    
            Returns:
                    Boolean vactor with a single True value at the given (state,index) pair (1 dim'l np.array).
                    
    '''   
    
    state_num = len(states)
    action_num = len(actions)
    state_action_num = state_num*action_num
    
    ind = lambda s,a: state_num*actions.index(a)+states.index(s)
    
    return np.where(np.eye(1,state_action_num,ind(s,a)),True,False).flatten()

def _neighbor(supp: np.array, A: np.array, mu: np.array, lb: float, discount: float, r: np.array, 
            states: list, actions: list, 
            sorted_risk_ind: dict, report: dict, state_action_space: np.array=np.array([]), tol: float=1e-8):
    '''
    Calculate the risk-maximizing deterministic policy whose support is in supp.
    
            Parameters:
                    supp (np.array): binary vector with True indicating (state,action) index in support and False otherwise.            
                    A (np.array): extended Q matrix [I-discount*P(a1)|...|I-discount*P(ak)] stacked with risk vector.
                    mu (np.array): initial state distribution.
                    lb (float): risk lower bound.
                    discount (float): discount factor in (0,1).
                    r (np.array): expected reward vector. 
                    states (list): state space.
                    actions (list): action space.
                    sorted_risk_ind (np.array): array of sorted risk index per state column.
                    report (dict): dictionary report for type of error.
                    state_action_space (np.array): binary vector representing the state-action space, 
                        with 1 indicating (state,action) pair in space and 0 otherwise. Defaults to full
                        (state,action) combinations.                    
                    tol (np.array): tolerance for near-zero values (default 1e-8).
                    logger_name (str): logger name to log next policy information (default None).
                    
            Returns:
                    Occupation measure of optimal neighboring policy.
                    Its expected risk.
                    Its expected reward.
                    
    '''    

    sas = state_action_space if state_action_space.size > 0 else np.ones(A.shape[-1])

    assert sas.size == A.shape[-1], \
    'Could not produce risk-maximizing deterministic policy due to input size mismatch: state-action space size != #cols in A.'
    
    state_actions = _sorted_risk_dict(sorted_risk_ind, states, actions, sas)

    candidates = {'states': [], 'actions': [], 'rhos': [], 'risks': [], 'rewards': []}
    
    for s in states:

        for a in state_actions[s]:

            try:

                I = supp + _pair_support(s,a,states,actions)
                AIinv = la.inv(A[:,I])
                B = (r.T[:,I]@AIinv)@A - r.T
                B[np.isclose(B,0.0,atol=tol)] = 0.0

                if np.all(B >= 0.0):

                    J = np.array([*range(A.shape[-1])])[(~(B > 0)).flatten()]
                    AJinv = la.inv(A[:,J])
                    next_risk = _max_risk(AJinv,mu,lb,discount,report,tol/10)
                    next_rhoJ = AJinv@np.hstack([(1-discount)*mu,np.array([next_risk])])[:,None]
                    next_rhoJ = np.where(np.isclose(next_rhoJ,0.0,atol=tol),0.0,next_rhoJ).flatten()
                    next_rho = np.zeros(A.shape[-1])
                    next_rho[J] = next_rhoJ
                    next_reward = (next_rho@r)[0]

                    candidates['states'].append(s)
                    candidates['actions'].append(a)
                    candidates['rhos'].append(next_rho)
                    candidates['risks'].append(next_risk)
                    candidates['rewards'].append(next_reward)
                                        
            except:

                report['type'] = 'MatrixInversionFail'

                continue
    
    return candidates

def optimal_policy(states: list, actions: list, 
                   Q: np.array, r: np.array, d: np.array, mu: np.array, discount: float, 
                   report: dict={}, is_eliminate: bool=False, sensitivity: float=1e-2, tol: float=1e-8, 
                   is_verbose: bool=False, logger_name: Optional[str]=None) -> Tuple[list, list, list, bool]:

    '''
    Calculate the Pareto frontier for the true optimal policy. The Pareto frontier is arranged by increasing optimality. The true optimum is last. 
    
            Parameters:
                    states (list): state space.
                    actions (list): action space.
                    Q (np.array): the Q matrix [I-discount*P(a1)|...|I-discount*P(ak)].
                    r (np.array): expected reward vector. 
                    d (np.array): expected risk vector. 
                    mu (np.array): initial state distribution.
                    discount (float): discount factor in (0,1).
                    report (dict): dictionary report for type of error.
                    is_eliminate (bool): eliminate current policy support from state-space (default False).
                    sensitivity (np.array): sensitivity to improvement in optimal values (default 1e-2).
                    tol (np.array): tolerance for near-zero values (default 1e-8).
                    logger_name (str): logger name to log policy information along Pareto frontier (default None).
                    
            Returns:
                    List of occupation measures on the Pareto frontier. The last element is the true optimum.
                    List of expected rewards along the Pareto frontier.
                    List of expected risks along the Pareto frontier.
                    Success / fail flag.
                    
    '''    

    rhos, risks, rewards = [], [], []
    A = np.vstack([Q,d.T])
    
    state_num = len(states)
    action_num = len(actions)
    
    sas = np.ones(state_num*action_num) # state-action space
    sorted_risk_ind = _sorted_risk_index(d,state_num,action_num,sas)

    rhos = [_min_risk(Q,mu,discount,d,report,tol/100)] # increase precision for finding min
    risks = [(d.T@rhos[-1])[0]]
    rewards = [(r.T@rhos[-1])[0]]
    
    success = False

    while sas.sum()>0: # reaches the end when is_eliminate=True

        rho, risk, reward = rhos[-1], risks[-1], rewards[-1]

        supp = support(rho)

        if not np.all(supp.reshape(action_num,state_num).sum(axis=0) == 1.0):
            report['type'] = 'NextPolicyNotDeterministic'
            raise RuntimeError('Next policy iterate is not deterministic.')  

        sas = sas*(~supp)

        candidates = _neighbor(supp,A,mu,risks[0],discount,r,states,actions,sorted_risk_ind,report,sas,tol)

        if candidates['risks']:
        
            if (not is_eliminate) and (len(risks)>1) and (len(candidates['risks'])==1): # reaches the end when is_eliminate=False

                if logger_name:
                    log = logging.getLogger(logger_name)
                    log.debug('optimal_policy: Optimization done')

                if is_verbose:
                    print('optimal_policy: Optimization done')
                
                return rhos, rewards, risks, success
                
            else:
            
                risk_max_idx = np.array(candidates['risks']).argmax()
                next_rho = candidates['rhos'][risk_max_idx]
                next_risk = candidates['risks'][risk_max_idx]
                next_reward = candidates['rewards'][risk_max_idx]

                # switch @ states and actions
                s = candidates['states'][risk_max_idx]
                a = candidates['actions'][risk_max_idx]

                if logger_name:
                    log = logging.getLogger(logger_name)
                    log.debug('optimal_policy: switched to action {} at s={}'.format(a,s))
                    log.debug('optimal_policy: next reward={:.3f}, next risk={:.3f}, next ratio={:.3f}'.format(next_reward,next_risk,next_reward/next_risk))

                if is_verbose:
                    print('optimal_policy: Switched to action {} at s={}'.format(a,s))
                    print('optimal_policy: Next reward: {:.3f}, Next risk: {:.3f}, Next ratio: {:.3f}'.format(next_reward,next_risk,next_reward/next_risk))
                    print('---------------------------------------------------------')

                success = (reward) / (risk) - (next_reward) / (next_risk) <= sensitivity # ratio is quasi-convex

                rhos.append(next_rho)
                risks.append(next_risk)
                rewards.append(next_reward)

        else:
            
            report['type'] = 'NeighborFail'

            return rhos, rewards, risks, success

        if not is_eliminate: # re-esbalish original state-action space.
            
            sas = np.ones(state_num*action_num)
        
    return rhos, rewards, risks, success