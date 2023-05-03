from collections import defaultdict
import pandas as pd
from itertools import product
import math

class DataProcessing:
    """
    High Level Description:
    """

    def __init__(
            self,
            state_columns,
            action_columns,
            cost_column,
            dataframe,
            risk_columns,
            # intervals,
            beta=0.99,
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
        self.total_mapping = dict()
        self.total_intervals = dict()
        self.sub_state_mdps = dict()
        self.neighbour_states = defaultdict(list)
        self.mapping = defaultdict(set)
        self.risk_mapping = defaultdict(dict)

    def build_states_list(self, discr = True):
        self.states = set()
        if discr:
            for index, row in self.dataframe.iterrows():
                st = ''
                for c in self.state_columns:
                    s = int(row[c])
                    st = st + str(s)
                self.states.add(st)
            self.states = sorted(self.states)
        return self

    def get_state_from_data(self, row):
        st = ''
        for c in self.state_columns:
            s = int(row[c])
            st = st + str(s)
        state = st
        return state

    def get_action_from_data(self, row):
        act = ''
        for a in self.action_columns:
            a = int(row[a])
            act = act + str(a)
        action = act
        return action

    def build_actions_list_per_state(self):
        for index, row in self.dataframe.iterrows():
            state = self.get_state_from_data(row)
            action = self.get_action_from_data(row)
            # if state == '32':
            #     print(state)
            #     print(action)
            self.mapping[state].add(action)
        return self

    def order_actions_risk(self):
        # actions = ['U','D','L','R','N']
        self.risk_mapping = {(s, a): 0 for s in self.states for a in self.mapping[s]}
        count = defaultdict(int)
        count = {(s, a): 0 for s in self.states for a in self.mapping[s]}
        for index, row in self.dataframe.iterrows():
            state = self.get_state_from_data(row)
            action = self.get_action_from_data(row)
            self.risk_mapping[(state, action)] += row['risk']
            count[(state, action)] += 1
        for s in self.states:
            for a in self.mapping[s]:
                self.risk_mapping[(s, a)] = self.risk_mapping[(s, a)]/count[(s, a)]
                # print(self.risk_mapping[(s, a)])
        return self

    def get_action(self, state, mapping):
        action = mapping[state][0]
        return action

    def return_interval_index(self, variable, intervals, value):
        count = 0
        for interval in intervals[variable]:
            if value >= interval:
                count += 1
        interval_value = count - 1
        length = len(intervals[variable]) - 2  # ensure that we never beyond the upper interval index
        interval_value = min(interval_value, length)
        return interval_value

    def return_mid_interval(self, variable, intervals, value):
        min_bound = 0
        max_bound = 0
        for interval in intervals[variable]:
            if value >= interval:
                min_bound = interval
        for interval_max in list(reversed(intervals[variable])):
            if value <= interval_max:
                max_bound = interval_max
        value_act = (min_bound + max_bound) / 2
        value_act = math.floor(value_act)
        return value_act

if __name__ == '__main__':
    df = pd.read_csv('../notebooks/grid45_rand_nind.csv')
    dataframe = df
    DP = DataProcessing(['state_i', 'state_j'], ['action'], ['reward'], dataframe, ['risk'])
    DP.build_states_list()
    DP.build_actions_list_per_state()
    print(DP.states)
    for s in DP.states:
        for i in DP.mapping[s]:
            print(s, i)
