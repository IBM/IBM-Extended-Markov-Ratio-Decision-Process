# import os
import argparse
# import pickle
# import pandas as pd
import numpy as np
import os
import yaml
from collections import defaultdict
import pickle
from datetime import datetime
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

from grid.utils import is_path, is_file
from grid.utils import get_s3_object
from grid.policy import occupation_measure, state_string, state_tuple, get_ratio
from grid.utils import load_all
from processing_old import DataProcessing
# from scripts.grid_data import generate

class EMRDPAlgorithm:
    """
        High Level Description:
        """

    def __init__(
            self,
            state_dist,
            P,
            tol,
            r,
            d,
            state_columns,
            action_columns,
            cost_column,
            dataframe,
            risk_columns,
            init_st,
            m,
            n,
            beta=0.9,
            max_ratio=False,
    ):
        """

        :param state_columns:
        :param action_columns:
        :param cost_column:
        :param dataframe:
        """
        self.state_columns = state_columns
        # self.intervals = intervals
        self.action_columns = action_columns
        self.risk_columns = risk_columns
        self.cost_column = cost_column
        self.dataframe = dataframe
        self.beta = beta
        self.n_q_iteration = 100
        self.total_mapping = dict()
        self.total_intervals = dict()
        self.sub_state_mdps = dict()
        self.neighbour_states = defaultdict(list)
        self.mapping = defaultdict(set)
        self.current_policy = defaultdict(set)
        self.dp = DataProcessing(self.state_columns, self.action_columns, self.cost_column, self.dataframe, self.risk_columns)
        self.min_policy = defaultdict(int)
        self.max_ratio = max_ratio
        self.max_number = sys.maxsize
        self.init_state = init_st
        self.m = m
        self.n = n
        self.state_dist = state_dist
        self.P = P
        self.tol = tol
        self.r = r
        self.d = d
        self.theory = True


    def initialize(self):
        self.dp.build_states_list()
        self.dp.build_actions_list_per_state()
        self.dp.order_actions_risk()
        # self.min_policy = {s: min(self.dp.mapping[s]) for s in self.dp.states}
        # self.min_policy = {'00': '1', '01': '0', '02': '1', '03': '1', '04': '0', '10': '0', '11': '0', '12': '0', '13': '0', '14': '0',
        #  '20': '0', '21': '0', '22': '0', '23': '0', '24': '1', '30': '0', '31': '0', '32': '0', '33': '0', '34': '0'}
        if self.max_ratio:
            for s in self.dp.states:
                min_a = self.max_number
                # print(s, self.dp.mapping[s])
                act_min = min(self.dp.mapping[s])
                for a in self.dp.mapping[s]:
                    if abs(self.dp.risk_mapping[(s, a)]) < min_a:
                        min_a = abs(self.dp.risk_mapping[(s, a)])
                        act_min = a
                        # print('RRRRR', s, act_min, min_a)
                # print('FFFF', s, act_min, min_a)
                self.min_policy[s] = act_min
        else:
            for s in self.dp.states:
                min_a = self.max_number
                # print(s, self.dp.mapping[s])
                act_min = min(self.dp.mapping[s])
                for a in self.dp.mapping[s]:
                    if self.dp.risk_mapping[(s, a)] < min_a:
                        min_a = self.dp.risk_mapping[(s, a)]
                        act_min = a
                        # print('RRRRR', s, act_min, min_a)
                # print('FFFF', s, act_min, min_a)
                self.min_policy[s] = act_min
        # self.min_policy = {'00': '3', '01': '1', '02': '0', '03': '0', '04': '0', '10': '0', '11': '0', '12': '0', '13': '0', '14': '0', '20': '0', '21': '0', '22': '0', '23': '0', '24': '0', '30': '0', '31': '1', '32': '0', '33': '0', '34': '0'}
        # for s in self.dp.states:
        #     print(s, self.min_policy[s])
        # print(self.min_policy)
        return self

    def return_neighbouring_policies(self, current_policy) -> list:
        neighb_policies_list = [{**current_policy, **{s: i}} for s in
                                self.dp.states for i in self.dp.mapping[s] if int(i) != int(current_policy[s])]
        return neighb_policies_list

    def calc_empirical_probability(self, policy):
        count = defaultdict(int)
        reward = defaultdict(float)
        risk = defaultdict(float)
        frac_next_pair = defaultdict(dict)

        for curr_row, next_row in zip(self.dataframe[:-1].iterrows(), self.dataframe[1:].iterrows()):
            s1 = int(curr_row[1][0])
            s2 = int(curr_row[1][1])
            s = str(s1) + str(s2)
            a = str(int(curr_row[1][2]))
            r = curr_row[1][3]
            ri = curr_row[1][4]

            ss1 = int(next_row[1][0])
            ss2 = int(next_row[1][1])
            ss = str(ss1) + str(ss2)
            # aa = str(int(next_row[1][2]))
            # rr = next_row[1][3]
            # rrisk = next_row[1][4]
            count[(s, a)] += 1
            reward[(s, a)] += r
            risk[(s, a)] += ri
            frac_next_pair[(s, a)][(ss, str(policy[ss]))] = 1

        # take the average
        for pair in reward:
            reward[pair] /= count[pair]
            risk[pair] /= count[pair]
        for pair in frac_next_pair:
            # print(pair, sum(frac_next_pair[pair].values()))
            total = sum(frac_next_pair[pair].values())
            for next_pair in frac_next_pair[pair]:
                frac_next_pair[pair][next_pair] /= total

        return [frac_next_pair, reward, risk]

    def ope(self, policy):
        [frac_next_pair, reward, risk] = self.calc_empirical_probability(policy)
        # print('FF', frac_next_pair, reward, risk)
        # initialize Q function
        # Q = defaultdict(float)
        Q_reward = {(s, a): 0 for s in self.dp.states for a in self.dp.mapping[s]}
        Q_risk = {(s, a): 0 for s in self.dp.states for a in self.dp.mapping[s]}
        # print('Q_reward', Q_reward)
        # print('Q_risk', Q_risk)

        # main of Q-evaluation for the discounted cost
        q_values_reward = [list(Q_reward.values())]
        q_values_risk = [list(Q_risk.values())]
        for _ in range(self.n_q_iteration):
            # repeat N times such that discount_factor**N << 1
            Q_new_reward = defaultdict(float)
            Q_new_risk = defaultdict(float)
            for pair in sorted(Q_reward.keys()):
                # expected_next_Q = sum_{(ss,aa)} p((ss,aa)|(s,a)) * Q(ss,aa)
                # for next_pair in sorted(frac_next_pair[pair].keys()):
                #     print('P', next_pair)
                expected_next_Q_reward = sum([frac_next_pair[pair][next_pair] * Q_reward[next_pair] for next_pair in sorted(frac_next_pair[pair].keys())])
                expected_next_Q_risk = sum(
                    [frac_next_pair[pair][next_pair] * Q_risk[next_pair] for next_pair in sorted(frac_next_pair[pair].keys())])
                Q_new_reward[pair] = reward[pair] + self.beta * expected_next_Q_reward
                Q_new_risk[pair] = risk[pair] + self.beta * expected_next_Q_risk
            Q_reward = Q_new_reward
            Q_risk = Q_new_risk
            # print(list(Q_reward.values()))
            # print(list(Q_reward.keys()))
            ks = list(Q_reward.keys())
            # print((state_string(self.init_state, 3, 4), str(policy[state_string(self.init_state, 3, 4)])))
            # print(ks.index((state_string(self.init_state, 3, 4), str(policy[state_string(self.init_state, 3, 4)]))))
            q_values_reward.append(list(Q_reward.values()))
            q_values_risk.append(list(Q_risk.values()))
        q_values_reward = np.vstack(q_values_reward)
        q_values_risk = np.vstack(q_values_risk)

        # reward = np.random.randint(50)
        # print('V', list(Q_reward.values()))
        # print('K', list(Q_reward.keys()))
        # print('RV', list(Q_risk.values()))
        # print('RK', list(Q_risk.keys()))
        # init_state = 15
        # print(q_values_reward[self.n_q_iteration-1])
        init_state4q = ks.index((state_string(self.init_state, self.m, self.n), str(policy[state_string(self.init_state, self.m, self.n)])))
        reward = q_values_reward[self.n_q_iteration-1][init_state4q]
        # print('Reward', reward)
        risk = q_values_risk[self.n_q_iteration-1][init_state4q]
        # print('Risk', risk)
        if self.theory:
            pdict = policy
            d = {(int(s[0]), int(s[1])): int(a) for s, a in pdict.items()}
            # m = max([k[0] for k in d.keys()]) + 1
            n = max([k[1] for k in d.keys()]) + 1
            action_num = 5
            p = {n * k[0] + k[1]: np.eye(1, action_num, v).flatten() for k, v in d.items()}
            rho = occupation_measure(p, self.state_dist, self.beta, self.P, self.tol)
            rewardSim, riskSim, ratioSim = get_ratio(rho, self.r, self.d)
            # print('Reward', rewardSim)
            # print('Risk', riskSim)
            # print('Ratio', ratioSim)
            reward = rewardSim
            risk = riskSim
        return [reward, risk]

    def return_max_neighbour(self, current_policy, current_reward, current_risk):
        neighb_policies_list = self.return_neighbouring_policies(current_policy)
        # print('NB', neighb_policies_list)
        max_policy = current_policy
        max_reward = current_reward
        max_risk = current_risk
        if self.max_ratio:
            grad = -1*self.max_number
        else:
            grad = -1*self.max_number
        for policy in neighb_policies_list:
            # print(policy)
            [reward, risk] = self.ope(policy)
            # print(risk, max_risk, policy)
            if self.max_ratio:
                if abs(risk) > abs(max_risk):
                    # print(risk, policy)
                    tmp = (reward - max_reward)/(risk - max_risk)
                    # print(tmp, grad)
                    if tmp > grad:
                        grad = tmp
                        max_reward = reward
                        max_policy = policy
                        max_risk = risk
            else:
                if risk > max_risk and reward > max_reward:
                # if reward > max_reward:
                    # print(risk, policy)
                    tmp = (reward - max_reward)/(risk - max_risk)
                    # print('TTT', tmp, grad)
                    if tmp > grad:
                        grad = tmp
                        max_reward = reward
                        max_policy = policy
                        max_risk = risk
        return [max_reward, max_policy, max_risk]

    def run_alg(self):
        self.initialize()
        # init_state = 15
        current_policy = self.min_policy
        [reward, risk] = self.ope(current_policy)
        current_reward = reward
        current_risk = risk
        candidates_list = []
        candidates_list.append([current_policy, current_reward, current_risk])
        # print('RR', current_reward, 'Risk', current_risk, 'Policy', current_policy)
        flag = True
        # print(current_policy)
        while flag:
            [next_reward, next_policy, next_risk] = self.return_max_neighbour(current_policy, current_reward, current_risk)
            # if current_reward >= next_reward or current_risk >= next_risk:
            print(next_policy)
            if self.max_ratio:
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

    def return_max_ratio(self, candidates_list):
        ratio = -1*self.max_number
        max_policy = 0
        for c in candidates_list:
            tmp = c[1]/c[2]
            # print('Reward ', c[1], ' Risk ', c[2], ' Ratio ', tmp)
            if tmp > ratio:
                ratio = tmp
                max_policy = c[0]
        return [ratio, max_policy]

    def return_min_ratio(self, candidates_list):
        self.rewards_alg = []
        self.risks_alg = []
        self.ratios_alg = []
        self.rewards_alg_true = []
        self.risks_alg_true = []
        self.ratios_alg_true = []
        ratio = self.max_number
        min_policy = 0
        for c in candidates_list:
            tmp = c[1]/c[2]
            print('Reward ', c[1], ' Risk ', c[2], ' Ratio ', tmp)
            self.rewards_alg.append(c[1])
            self.risks_alg.append(c[2])
            self.ratios_alg.append(tmp)
            if tmp < ratio:
                ratio = tmp
                min_policy = c[0]

            # if self.theory == False:
            pdict = c[0]
            d = {(int(s[0]), int(s[1])): int(a) for s, a in pdict.items()}
            # m = max([k[0] for k in d.keys()]) + 1
            n = max([k[1] for k in d.keys()]) + 1
            action_num = 5
            p = {n * k[0] + k[1]: np.eye(1, action_num, v).flatten() for k, v in d.items()}
            rho = occupation_measure(p, self.state_dist, self.beta, self.P, self.tol)
            rewardSim, riskSim, ratioSim = get_ratio(rho, self.r, self.d)
            self.rewards_alg_true.append(rewardSim)
            self.risks_alg_true.append(riskSim)
            self.ratios_alg_true.append(ratioSim)
            print('RewardTrue ', rewardSim, ' RiskTrue ', riskSim, ' RatioTrue ', ratioSim)
            # print('Reward', rewardSim)
            # print('Risk', riskSim)
            # print('Ratio', ratioSim)
            # reward = rewardSim
            # risk = riskSim
        return [ratio, min_policy]

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('-p_in', '--path_in', type=is_path, help="Directory to load data objects locally.")
        parser.add_argument('-p_out', '--path_out', type=is_path, help="Directory to store data objects locally.")
        parser.add_argument('-c', '--configs', type=is_file,
                            help="Absolute path to configs yaml to upload results to COS.")

        args = parser.parse_args()

        args_dict = args.__dict__
        kwargs = {k: v for k, v in args_dict.items()}

        if any([kwargs['path_out'], kwargs['configs']]):

            is_local = True if kwargs['path_out'] else False

            if is_local:

                dir_path_out = kwargs['path_out']

            else:

                configs_file = kwargs['configs']

                with open(configs_file, 'r') as file:
                    try:
                        configs = yaml.safe_load(file)
                    except yaml.YAMLError as e:
                        print('({}) ERROR ... Could not load configs yaml.'.format('root'))
                        raise e

                s3 = get_s3_object(configs)
                bucket = configs['s3']['buckets']['results']

            # df = pd.read_csv('../notebooks/data_grid45_newest.csv')
            # m = 3
            # n = 5
            # configs_file = '/Users/zalex/Documents/emrdp/emrdp/configs.yaml'
            data_obj_list = load_all(configs=configs_file, is_move=False)
            for data in data_obj_list:
                # print(data['noise'])
                # data_obj = generate(grid_rows=m, grid_cols=n, discount=0.9, horizon=10000, noise=0.2, tolerance=1e-6, obstacle_num=1)
                # p_file = 'grid_world_data_{}.pickle'.format(data_obj['policy_data_id'])
                # p_path = os.path.join('../notebooks/', p_file)
                #
                # with open(p_path, 'wb') as handle:
                #     pickle.dump(data_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
                #
                # # data_file = '../notebooks/grid_world_data_2022-06-16_1207.pickle'
                #
                # # with open(data_file, 'rb') as handle:
                # #     data = pickle.load(handle)
                # with open(p_path, 'rb') as handle:
                #     data = pickle.load(handle)
                df = data['df']
                state_dist = data['state_dist']
                discount = data['discount']
                P = data['P']
                r = data['r']
                d = data['d']
                tol = data['tol']
                m = data['m']
                n = data['n']
                # df.to_csv(index=False)
                # print(df)
                init_st = data['init_state']
                # print(init_st)
                # dataframe = df
                # print(df.head(10))
                # print(state_tuple(init_st, 3, 4))
                emr = EMRDPAlgorithm(state_dist, P, tol, r, d, ['state_i', 'state_j'], ['action'], ['reward'],
                                     df, ['risk'], init_st, m, n, max_ratio=False)
                # emr.initialize()
                candidates_list = emr.run_alg()
                # emr.dp.order_actions_risk()
                # print(emr.max_ratio)
                if emr.max_ratio:
                    [ratio, max_policy] = emr.return_max_ratio(candidates_list)
                else:
                    [ratio, max_policy] = emr.return_min_ratio(candidates_list)
                print([ratio, max_policy])
                final = data['final']
                obstacles = data['obstacles']
                delta = data['delta']
                noise = data['noise']
                epsilon = data['epsilon']
                p_opt_noise = data['p_data']
                p_opt = data['p_opt']
                rewards = data['pareto_rewards']
                risks = data['pareto_risks']
                rhos = data['pareto_rhos']
                policy_data_id = data['policy_data_id']
                now = datetime.now().strftime('%Y-%m-%d-%H%M')
                data_results = {
                    'm': m,
                    'n': n,
                    'final': final,
                    'obstacles': obstacles,
                    'init_state': init_st,
                    'state_dist': state_dist,
                    'discount': discount,
                    'delta': delta,
                    'tol': tol,
                    'noise': noise,
                    'epsilon': epsilon,
                    'P': P,
                    'r': r,
                    'd': d,
                    'p_data': p_opt_noise,
                    'p_opt': p_opt,
                    'pareto_rewards': rewards,
                    'pareto_risks': risks,
                    'pareto_rhos': rhos,
                    'rewards_alg': emr.rewards_alg,
                    'risks_alg': emr.risks_alg,
                    'ratios_alg': emr.ratios_alg,
                    'rewards_alg_true': emr.rewards_alg_true,
                    'risks_alg_true': emr.risks_alg_true,
                    'ratios_alg_true': emr.ratios_alg_true,
                    'policy_data_id': policy_data_id,
                    'policy_opt': max_policy,
                    'theory': emr.theory,
                    'timestamp': now
                }

                print('PR', data['pareto_rewards'])
                print('PRisk', data['pareto_risks'])
                ratio_div = [i / j for i, j in zip(data['pareto_rewards'], data['pareto_risks'])]
                print('Pratio', ratio_div)
                # out_file = 'results_data_{}.pickle'.format(now)
                # root = '/Users/zalex/Documents/emrdp/emrdp/emrdp/'
                # out_path = os.path.join(root, out_file)

                # with open(out_path, 'wb') as handle:
                #     pickle.dump(data_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

                # data = generate(**kwargs)
                # p_file = 'grid_world_data_{}.pickle'.format(data['policy_data_id'])
                p_file = 'grid_world_results_{}_theory{}_m{}n{}_noise{}.pickle'.format(data['policy_data_id'],
                                                                                       emr.theory, m, n,
                                                                                       int(100 * noise))

                if is_local:

                    p_path = os.path.join(dir_path_out, p_file)

                    with open(p_path, 'wb') as file:
                        pickle.dump(data_results, file, protocol=pickle.HIGHEST_PROTOCOL)

                else:

                    s3.Bucket(bucket).put_object(Body=pickle.dumps(data_results, protocol=pickle.HIGHEST_PROTOCOL),
                                                 Key=p_file)

        else:

            print('Args should include either an absolute path to a local dir or a configs yaml for COS.')
