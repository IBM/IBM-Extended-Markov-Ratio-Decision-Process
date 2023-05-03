import numpy as np
from scipy.stats import norm

Rhigh = 1.0
Rlow = 5.0
Rbound = 10.0

def _check_stochastic(A: np.array, tol: float=1e-8):

    try:

        assert np.all(A<=1) and np.all(A>=0), \
        'Matrix is not stochastic. Entries must be >=0 and <=1.'    

        assert np.all(np.isclose(np.ones((1,A.shape[0])) @ A, np.ones((1,A.shape[0])), atol=tol)), \
        'Matrix is not stochastic. Columns must add up to 1.'

    except AssertionError as ae:
        raise ae    

    except Exception as e:
        raise e       

def _transition_probability_matrix_up(m: int, n: int, delta: float, tol: float=1e-8) -> np.array:

    PU = delta/(m*n-1) * np.ones(shape=(m*n,m*n)) 

    for i in range(m):
        for j in range(n):
            if i==0:
                    PU[i*n+j,i*n+j] = 1 - delta            
            else:
                    PU[(i-1)*n+j,i*n+j] = 1 - delta

    _check_stochastic(PU,tol=tol)

    return PU

def _transition_probability_matrix_down(m: int, n: int, delta: float, tol: float=1e-8) -> np.array:    

    PD = delta/(m*n-1)*np.ones(shape=(m*n,m*n)) 

    for i in range(m):
        for j in range(n):
            if i==m-1:
                PD[i*n+j,i*n+j] = 1 - delta            
            else:
                PD[(i+1)*n+j,i*n+j] = 1 - delta

    _check_stochastic(PD,tol=tol)

    return PD

def _transition_probability_matrix_left(m: int, n: int, delta: float, tol: float=1e-8) -> np.array:                       

    PL = delta/(m*n-1)*np.ones(shape=(m*n,m*n))

    for i in range(m):
        for j in range(n):
            if j==0:
                PL[i*n+j,i*n+j] = 1 - delta            
            else:
                PL[i*n+(j-1),i*n+j] = 1 - delta

    _check_stochastic(PL,tol=tol)

    return PL

def _transition_probability_matrix_right(m: int, n: int, delta: float, tol: float=1e-8) -> np.array:    

    PR = delta/(m*n-1)*np.ones(shape=(m*n,m*n)) 

    for i in range(m):
        for j in range(n):
            if j==n-1:
                PR[i*n+j,i*n+j] = 1 - delta            
            else:
                PR[i*n+(j+1),i*n+j] = 1 - delta

    _check_stochastic(PR,tol=tol)

    return PR

def _transition_probability_matrix_none(m: int, n: int, delta: float=0.5, tol: float=1e-8) -> np.array:        

    PN = delta/(m*n-1)*np.ones(shape=(m*n,m*n))

    for i in range(m):
        for j in range(n):
            PN[i*n+j,i*n+j] = 1 - delta

    _check_stochastic(PN,tol=tol)

    return PN

def transition_probability_matrix(m, n, delta: float, deltaN: float=0.5, tol: float=1e-8) -> np.array:

    PU = _transition_probability_matrix_up(m,n,delta)
    PD = _transition_probability_matrix_down(m,n,delta)
    PL = _transition_probability_matrix_left(m,n,delta)
    PR = _transition_probability_matrix_right(m,n,delta)
    PN = _transition_probability_matrix_none(m,n,deltaN)
                         
    P = np.hstack([PU,PD,PL,PR,PN])

    return P

def discounted_probability_matrix(m: int, n: int, discount: float, delta: float, deltaN: float=0.5, tol: float=1e-8):

    PU = _transition_probability_matrix_up(m,n,delta)
    PD = _transition_probability_matrix_down(m,n,delta)
    PL = _transition_probability_matrix_left(m,n,delta)
    PR = _transition_probability_matrix_right(m,n,delta)
    PN = _transition_probability_matrix_none(m,n,deltaN)
                         
    I = np.identity(m*n)
    Q = np.hstack([I-discount*PU,I-discount*PD,I-discount*PL,I-discount*PR,I-discount*PN])

    return Q

def expected_reward_vector(m: int, n: int, discount: float, final: int, obstacles: list, P: np.array, cost: float=1.0, epsilon: float=1e-2) -> np.array:

    M = 2/(1-discount) # obstacle penalty 

    c = cost*np.ones(m*n)[:,None]

    c[final] = 0.0
    for o in obstacles:
        c[o] = M

    r = (c.T @ P).T
    r = r + norm(loc=0, scale=epsilon).rvs(r.shape)
    r = np.where(r<0,0,r)

    return r

def expected_risk_vector(m: int, n: int, rhigh: float=Rhigh, rlow: float=Rlow, rbound: float=Rbound, epsilon: float=1e-2) -> np.array:

    UR = rbound*np.ones(m*n)[:,None]
    DR = rbound*np.ones(m*n)[:,None]
    LR = rbound*np.ones(m*n)[:,None]
    RR = rbound*np.ones(m*n)[:,None]
    NR = rlow*np.ones(m*n)[:,None]

    for i in range(m):

        LR[n*i] = rlow # walk into left boundary
        RR[n*i+(n-1)] = rlow # walk into right boundary
        
    for j in range(n):
        
        UR[j] = rlow # walk into upper boundary
        DR[n*(m-1)+j] = rlow # walk into lower boundary
            
        if j>0:
            LR[j] = rhigh # walk along upper boundary
            LR[n*(m-1)+j] = rhigh # walk along lower boundary
        
        if j<n-1:
            RR[j] = rhigh # walk along upper boundary
            RR[n*(m-1)+j] = rhigh # walk along lower boundary

    d = np.vstack([UR,DR,LR,RR,NR])
    d = d + norm(loc=0, scale=epsilon).rvs(d.shape)
    d = np.where(d<0,rbound,d)

    return d
