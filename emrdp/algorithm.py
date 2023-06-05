import logging
from collections import defaultdict
from typing import Optional, Tuple
import ray
import numpy as np
import pandas as pd

from emrdp.processing import DataProcessing
from collections import namedtuple

Policy = namedtuple('Policy', ['policy', 'reward', 'risk', 'grad'])

@ray.remote
def get_ratio(rho: np.array, r: np.array, d: np.array, omega: float=1.) -> Tuple[float, float, float]:
    '''
       Calculate expected reward to expected risk ratio.

               Parameters:
                       rho (np.array): occupation measure.
                       r (np.array): immediate reward vector.
                       d (np.array): immediate risk vector.
                       omega (float): power of the expected risk.

               Returns:
                       Reward, risk, and the ratio.

       '''
    reward = (rho@r)[0]
    risk = (rho@d)[0]
    ratio = abs(reward) / (abs(risk)**omega)
    
    return reward, risk, ratio

@ray.remote 
def occupation_measure_local(policy: dict, mu: np.array, discount: float, P: np.array, tol: float=1e-8) -> np.array:
#### local copy due to ray loading issues of imported grid modules to worker nodes

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

@ray.remote
def calc_empirical_probability(policy, dataframe: pd.DataFrame, logger_name: Optional[str]=None):
    '''
           Calculate empirical reward, risk, and probability to switch to next pair of state and action.

                   Parameters:
                           policy: policy.
                           dataframe: pd.DataFrame: data frame of the available data.
                           logger_name: logger.

                   Returns:
                           Dictionaries of empirical probabilities to next pair of state and action, reward and risk.

           '''
    count = defaultdict(int)
    reward = defaultdict(float)
    risk = defaultdict(float)
    frac_next_pair = defaultdict(dict)

    for curr_row, next_row in zip(dataframe[:-1].iterrows(), dataframe[1:].iterrows()):

        s1 = int(curr_row[1][0])
        s2 = int(curr_row[1][1])
        s = str(s1) + str(s2)
        a = str(int(curr_row[1][2]))
        r = curr_row[1][3]
        ri = curr_row[1][4]

        ss1 = int(next_row[1][0])
        ss2 = int(next_row[1][1])
        ss = str(ss1) + str(ss2)
        count[(s, a)] += 1
        reward[(s, a)] += r
        risk[(s, a)] += ri
        frac_next_pair[(s, a)][(ss, str(policy[ss]))] = 1

    for pair in reward:
        reward[pair] /= count[pair]
        risk[pair] /= count[pair]
    for pair in frac_next_pair:

        if logger_name:
            log = logging.getLogger(logger_name)
            log.debug('Calculate Empirical Probability: pair={}, sum={}'.format(pair, sum(frac_next_pair[pair].values())))

        total = sum(frac_next_pair[pair].values())
        for next_pair in frac_next_pair[pair]:
            frac_next_pair[pair][next_pair] /= total

    return frac_next_pair, reward, risk

@ray.remote(num_cpus=1)
def ope(policy, states, mapping,
        discount: float,
        state_dist, init_state: str,
        P: np.array, d: np.array, r: np.array,
        dataframe: pd.DataFrame,
        is_theory: bool, tol: float=1e-8, n_q_iteration: int=100, logger_name: Optional[str]=None):
    '''
           Calculate expected reward and risk for a given policy.

                   Parameters:
                           policy: cuurent policy
                           states: states
                           mapping: mapping of states to actions (all actions for each state)
                           discount (float): discount factor
                           state_dist: state distribution for theoretical occupation measure calculation
                           init_state (str): initial state
                           P (np.array): transition probability matrix
                           d (np.array): vector of immediate risks
                           r (np.array): vector of immediate rewards
                           dataframe (pd.DataFrame): data
                           is_theory (bool): if 1, theoretical algorithm will otherwise data learning one EMRDP
                           tol (float): tolerance
                           n_q_iteration (int): number of iterations for OPE run
                           logger_name: logger

                   Returns:
                           Expected reward and risk.

           '''
    assert all([discount>=0,discount<=1]), 'OPE: discount factor in [0,1].'

    frac_next_pair, reward, risk = ray.get(calc_empirical_probability.remote(policy,dataframe,logger_name))

    if logger_name:
        log = logging.getLogger(logger_name)
        log.debug('OPE: next_pair={} \nreward={} \nrisk={}'.format(frac_next_pair, reward, risk))

    Q_reward = {(s, a): 0 for s in states for a in mapping[s]}
    Q_risk = {(s, a): 0 for s in states for a in mapping[s]}

    if logger_name:
        log = logging.getLogger(logger_name)
        log.debug('OPE: Q reward=\n{} \nQ risk=\n{}'.format(Q_reward, Q_risk))

    q_values_reward = [list(Q_reward.values())]
    q_values_risk = [list(Q_risk.values())]

    for i in range(n_q_iteration):

        Q_new_reward = defaultdict(float)
        Q_new_risk = defaultdict(float)

        for pair in sorted(Q_reward.keys()):
            if all([(i+1)%10==0,logger_name]):
                log = logging.getLogger(logger_name)
                log.debug('OPE: iteration={}\nnext pairs (sorted)={}'.format(i+1,sorted(frac_next_pair[pair].keys())))
            expected_next_Q_reward = sum([frac_next_pair[pair][next_pair] * Q_reward[next_pair] for next_pair in sorted(frac_next_pair[pair].keys())])
            expected_next_Q_risk = sum([frac_next_pair[pair][next_pair] * Q_risk[next_pair] for next_pair in sorted(frac_next_pair[pair].keys())])
            Q_new_reward[pair] = reward[pair] + discount * expected_next_Q_reward
            Q_new_risk[pair] = risk[pair] + discount * expected_next_Q_risk

        Q_reward = Q_new_reward
        Q_risk = Q_new_risk

        if all([(i+1)%10==0,logger_name]):
            log = logging.getLogger(logger_name)
            log.debug('OPE: iteration={}, Q reward keys={}, Q reward values={}'.format(i+1,list(Q_reward.keys()),list(Q_reward.values())))

        ks = list(Q_reward.keys())
        q_values_reward.append(list(Q_reward.values()))
        q_values_risk.append(list(Q_risk.values()))

    q_values_reward = np.vstack(q_values_reward)
    q_values_risk = np.vstack(q_values_risk)

    if logger_name:
        log = logging.getLogger(logger_name)
        log.debug('OPE: Q reward values on final iteration=\n{}'.format(q_values_reward[n_q_iteration-1]))
        log.debug('OPE: Q risk values on final iteration=\n{}'.format(q_values_risk[n_q_iteration-1]))

    if is_theory:
        pdict = {(int(s[0]), int(s[1])): int(a) for s, a in policy.items()}
        n = max([k[1] for k in pdict.keys()]) + 1
        action_num = 5
        p = {n * k[0] + k[1]: np.eye(1, action_num, v).flatten() for k, v in pdict.items()}
        rho = ray.get(occupation_measure_local.remote(p, state_dist, discount, P, tol))
        reward, risk, _ = ray.get(get_ratio.remote(rho, r, d))
    else:
        init_state4q = ks.index((init_state, str(policy[init_state])))
        reward = q_values_reward[n_q_iteration-1][init_state4q]
        risk = q_values_risk[n_q_iteration-1][init_state4q]

    if logger_name:
        log = logging.getLogger(logger_name)
        log.debug('OPE: reward={}, risk={}, theory={}'.format(reward,risk,is_theory))

    return reward, risk

class EMRDPAlgorithm:
    '''
        EMRDP Algorithm Implementation.

        Attributes:
                           state_dist: state distribution for theoretical occupation measure calculation
                           state_columns: state variables (columns in tabular representation)
                           action_columns: action variables (columns in tabular representation)
                           reward_column: reward variables (column in tabular representation)
                           discount (float): discount factor
                           init_state (str): initial state
                           P (np.array): transition probability matrix
                           d (np.array): vector of immediate risks
                           r (np.array): vector of immediate rewards
                           dataframe (pd.DataFrame): data
                           is_theory (bool): if 1, theoretical algorithm will otherwise data learning one EMRDP
                           is_max: if False, the problem is the ratio minimization otherwise maximization
                           tol (float): tolerance
                           n_q_iteration (int): number of iterations for OPE run
                           logger_name: logger

       Methods:
                            initialize(self)
                                Computes minimum risk policy.
                            return_neighbouring_policies(self, current_policy) -> list
                                Computes neighboring policies
                            candidate_condition(self, reward, risk, relative_reward, relative_risk)
                                Check candidate conditions for the next policy on the optimal policies graph
                            grad(self, reward, risk, relative_reward, relative_risk)
                                Computes the gradient between current and neighboring policy
                            return_max_neighbour(self, current_policy, current_reward, current_risk)
                                Returns the best next policy, risk and reward
                            run(self)
                                Run the EMRDP algorithm and return the candidate list
                            return_optimal_ratio(self, candidates_list)
                                Computes and return optimal ratio and optimal policy
    '''

    def __init__(
            self,
            state_dist,
            P,
            tol,
            r,
            d,
            discount,
            dataframe,
            state_columns,
            action_columns,
            reward_column,
            risk_columns,
            is_theory=True,
            is_max=False,
            n_q_iteration=100,
            logger_name=None
    ):
        '''

                   Parameters:
                           state_dist: state distribution for theoretical occupation measure calculation
                           state_columns: state variables (columns in tabular representation)
                           action_columns: action variables (columns in tabular representation)
                           reward_column: reward variables (column in tabular representation)
                           discount (float): discount factor
                           init_state (str): initial state
                           P (np.array): transition probability matrix
                           d (np.array): vector of immediate risks
                           r (np.array): vector of immediate rewards
                           dataframe (pd.DataFrame): data
                           is_theory (bool): if 1, theoretical algorithm will otherwise data learning one EMRDP
                           is_max: if False, the problem is the ratio minimization otherwise maximization
                           tol (float): tolerance
                           n_q_iteration (int): number of iterations for OPE run
                           logger_name: logger


           '''
        self.dataframe = dataframe
        self.state_columns = state_columns
        self.action_columns = action_columns
        self.risk_columns = risk_columns
        self.reward_column = reward_column
        self.discount = discount
        self.n_q_iteration = n_q_iteration
        self.total_mapping = dict()
        self.total_intervals = dict()
        self.sub_state_mdps = dict()
        self.neighbour_states = defaultdict(list)
        self.mapping = defaultdict(set)
        self.current_policy = defaultdict(set)
        self.dp = DataProcessing(self.dataframe, self.state_columns, self.action_columns, self.reward_column, self.risk_columns, self.discount)
        self.min_policy = defaultdict(int)
        self.is_max = is_max
        self.max_number = np.inf
        self.state_dist = state_dist
        self.P = P
        self.tol = tol
        self.r = r
        self.d = d
        self.theory = is_theory
        self.logger_name = logger_name

    def initialize(self):
        '''
                   Computes minimum risk policy.
       '''
        self.dp.build_states_list()
        self.dp.build_actions_list_per_state()
        self.dp.order_actions_risk()
        self.init_state = self.dp.init_state

        if self.logger_name:
            log = logging.getLogger(self.logger_name)
            log.debug('Initial State: state={}.'.format(self.init_state))

        for s in self.dp.states:

            min_a = self.max_number
            if self.logger_name:
                log = logging.getLogger(self.logger_name)
                log.debug('Initialize: state s={}, dp.mapping[s]={}, is_max={}.'.format(s, self.dp.mapping[s],self.is_max))
            act_min = min(self.dp.mapping[s])

            for a in self.dp.mapping[s]:

                risk_mapping_tmp = abs(self.dp.risk_mapping[(s, a)]) if self.is_max else self.dp.risk_mapping[(s, a)]
                if risk_mapping_tmp < min_a:
                    min_a = risk_mapping_tmp
                    act_min = a
                    if self.logger_name:
                        log = logging.getLogger(self.logger_name)
                        log.debug('Initialize (inside loop): state s={}, act_min={}, min_a={}, is_max={}.'.format(s, act_min, min_a,self.is_max))

            if self.logger_name:
                log = logging.getLogger(self.logger_name)
                log.debug('Initialize (outside loop): state s={}, act_min={}, min_a={}, is_max={}.'.format(s, act_min, min_a,self.is_max))

            self.min_policy[s] = act_min

    def return_neighbouring_policies(self, current_policy) -> list:
        '''
                   Computes neighboring policies.

                        Parameters:
                           current_policy: current optimal policy

                           Returns:
                                   Neighboring policies list.

       '''
        neighb_policies_list = [{**current_policy, **{s: i}} for s in self.dp.states for i in self.dp.mapping[s] if int(i) != int(current_policy[s])]

        return neighb_policies_list

    def candidate_condition(self, reward, risk, relative_reward, relative_risk):
        '''
                   Defines a condition for shortlisting of neighboring policies.

                         Parameters:
                                   reward: next policy reward
                                   risk: next reward risk
                                   relative_reward: current policy reward
                                   relative_risk: current policy risk

                           Returns:
                                   A candidate condition.

       '''
        return abs(risk) > abs(relative_risk) if self.is_max else ((risk > relative_risk) and (reward > relative_reward))

    def grad(self, reward, risk, relative_reward, relative_risk):
        '''
                   Computes a gradient between two neighboring policies.
                        Parameters:
                                   reward: next policy reward
                                   risk: next reward risk
                                   relative_reward: current policy reward
                                   relative_risk: current policy risk

                           Returns:
                                   A gradient.

       '''
        grad = -self.max_number if self.is_max else self.max_number

        try:
            grad = (reward-relative_reward)/(risk-relative_risk)
        except ZeroDivisionError as e:
            print('Failed to compute neighboring policy gradient')
            print(e)
        except ValueError as e:
            print('Failed to compute neighboring policy gradient')
            print(e)
        except TypeError as e:
            print('Failed to compute neighboring policy gradient')
            print(e)
        finally:
            return grad

    def return_max_neighbour(self, current_policy, current_reward, current_risk):
        '''
                   Computes the best next policy.

                           Parameters:
                                   current_policy: current policy
                                   current_reward: current policy reward
                                   current_risk: current policy risk

                           Returns:
                                   Next optimal policy, reward and risk.

       '''
        opt_policy, opt_reward,  opt_risk = current_policy, current_reward, current_risk

        neighbors = self.return_neighbouring_policies(current_policy)

        futures = [ope.remote(policy,
                                self.dp.states, self.dp.mapping, self.discount, self.state_dist, self.init_state,
                                self.P, self.d, self.r, self.dataframe, self.theory, self.tol, self.n_q_iteration, self.logger_name
            ) for policy in neighbors
            ]
        opes = ray.get(futures)

        grads = [self.grad(*ope,current_reward,current_risk) for ope in opes]
        candidates = [Policy(policy, *ope, grad) for policy, ope, grad in zip(neighbors, opes, grads) if self.candidate_condition(*ope,current_reward,current_risk)]

        if len(candidates)>0:

             grad_candidates = [candidate.grad if self.is_max else -candidate.grad for candidate in candidates]
             opt_arg = np.argmax(np.array(grad_candidates))
             opt_candidate = candidates[opt_arg]

             if abs(opt_candidate.grad) < self.max_number:
                 opt_policy, opt_reward,  opt_risk = opt_candidate.policy, opt_candidate.reward, opt_candidate.risk

        return opt_policy, opt_reward,  opt_risk

    def run(self):
        '''
                   Computes the optimal path policies.

                           Returns:
                                   A list of optimal path policies.

       '''
        self.initialize()
        current_policy = self.min_policy
        futures = [ope.remote(current_policy,
                                self.dp.states, self.dp.mapping, self.discount, self.state_dist, self.init_state,
                                self.P, self.d, self.r, self.dataframe, self.theory, self.tol, self.n_q_iteration, self.logger_name)]
        opes = ray.get(futures)
        current_reward, current_risk = opes[0]
        candidates_list = []
        candidates_list.append([current_policy, current_reward, current_risk])
        if self.logger_name:
            log = logging.getLogger(self.logger_name)
            log.debug('Run: current_policy={}, current_reward={:.4f}, current_risk={:.4f}'.format(current_policy,current_reward,current_risk))
        flag = True
        while flag:
            next_policy, next_reward, next_risk = self.return_max_neighbour(current_policy, current_reward, current_risk)
            if self.logger_name:
                log = logging.getLogger(self.logger_name)
                log.debug('Run: next_policy={}'.format(next_policy))
            if self.is_max:
                if abs(current_risk) >= abs(next_risk):
                    flag = False
                else:
                    current_policy = next_policy
                    current_reward = next_reward
                    current_risk = next_risk
                    candidates_list.append([current_policy, current_reward, current_risk])
            else:
                if current_risk >= next_risk:
                    flag = False
                else:
                    current_policy = next_policy
                    current_reward = next_reward
                    current_risk = next_risk
                    candidates_list.append([current_policy, current_reward, current_risk])

        return candidates_list

    def return_optimal_ratio(self, candidates_list):
        '''
                  Computes the optimal ratio and its corresponding policy.

                          Parameters:
                                   candidates_list: the list of candidates from the optimal path

                          Returns:
                                  An optimal ratio and its corresponding policy.

        '''
        if self.is_max:

            ratio = -1*self.max_number
            opt_policy = 0
            for c in candidates_list:
                tmp = c[1]/c[2]
                if tmp > ratio:
                    ratio = tmp
                    opt_policy = c[0]
        else:

            self.rewards_alg = []
            self.risks_alg = []
            self.ratios_alg = []
            self.rewards_alg_true = []
            self.risks_alg_true = []
            self.ratios_alg_true = []
            ratio = self.max_number
            opt_policy = 0
            for c in candidates_list:
                tmp = c[1]/c[2]
                if self.logger_name:
                    log = logging.getLogger(self.logger_name)
                    log.debug('Return Ratio: Reward={}, Risk={}, Ratio={}'.format(c[1],c[2],tmp))
                self.rewards_alg.append(c[1])
                self.risks_alg.append(c[2])
                self.ratios_alg.append(tmp)
                if tmp < ratio:
                    ratio = tmp
                    opt_policy = c[0]

                pdict = c[0]
                d = {(int(s[0]), int(s[1])): int(a) for s, a in pdict.items()}

                n = max([k[1] for k in d.keys()]) + 1
                action_num = 5
                p = {n * k[0] + k[1]: np.eye(1, action_num, v).flatten() for k, v in d.items()}
                rho = ray.get(occupation_measure_local.remote(p, self.state_dist, self.discount, self.P, self.tol))
                rewardSim, riskSim, ratioSim = ray.get(get_ratio.remote(rho, self.r, self.d))
                self.rewards_alg_true.append(rewardSim)
                self.risks_alg_true.append(riskSim)
                self.ratios_alg_true.append(ratioSim)
                if self.logger_name:
                    log = logging.getLogger(self.logger_name)
                    log.debug('Return Ratio: RewardTrue={}, RiskTrue={}, RatioTrue={}'.format(rewardSim,riskSim, ratioSim))
        if self.logger_name:
            log = logging.getLogger(self.logger_name)

            pdict = opt_policy
            d = {(int(s[0]), int(s[1])): int(a) for s, a in pdict.items()}

            n = max([k[1] for k in d.keys()]) + 1
            action_num = 5
            p = {n * k[0] + k[1]: np.eye(1, action_num, v).flatten() for k, v in d.items()}
            rho = ray.get(occupation_measure_local.remote(p, self.state_dist, self.discount, self.P, self.tol))
            rewardTrue, riskTrue, ratioTrue = ray.get(get_ratio.remote(rho, self.r, self.d))

            log.debug('Return Optimal Ratio: OptimalRatio={}, OptimalRatioTrue ={}, OptimalPolicy={}'.format(ratio, ratioTrue, opt_policy))

        return ratio, opt_policy

