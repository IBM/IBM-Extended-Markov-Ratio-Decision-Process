from collections import defaultdict

class DataProcessing:
    '''
        Data Processing Module of EMRDP Algorithm.

        Attributes:
                           state_columns: state variables (columns in tabular representation)
                           action_columns: action variables (columns in tabular representation)
                           reward_column: reward variables (column in tabular representation)
                           discount (float): discount factor
                           init_state (str): initial state
                           dataframe (pd.DataFrame): data

       Methods:
                            build_states_list(self, discr = True)
                                Builds state list from the data.
                            get_state_from_data(self, row)
                                Computes state from a single data row
                            get_action_from_data(self, row)
                                Computes action from a single data row
                            build_actions_list_per_state(self)
                                Builds mapping of actions per states
                            order_actions_risk(self)
                                Computes from data an empirical risk per state and action
    '''

    def __init__(
            self,
            dataframe,
            state_columns,
            action_columns,
            reward_column,
            risk_columns,
            discount,
    ):
        '''
                   Parameters:
                           state_columns: state variables (columns in tabular representation)
                           action_columns: action variables (columns in tabular representation)
                           reward_column: reward variables (column in tabular representation)
                           discount (float): discount factor
                           init_state (str): initial state
                           dataframe (pd.DataFrame): data

       '''
        self.state_columns = state_columns
        self.action_columns = action_columns
        self.risk_columns = risk_columns
        self.reward_column = reward_column
        self.dataframe = dataframe
        self.discount = discount
        self.total_mapping = dict()
        self.total_intervals = dict()
        self.sub_state_mdps = dict()
        self.neighbour_states = defaultdict(list)
        self.mapping = defaultdict(set)
        self.risk_mapping = defaultdict(dict)

    def build_states_list(self, discr = True):
        '''
                   Builds state list from the data.

                   Parameters:
                           discr = True: whether data is discretized (binned) already
        '''
        self.states = set()
        if discr:
            for index, row in self.dataframe.iterrows():
                st = ''
                for c in self.state_columns:
                    s = int(row[c])
                    st = st + str(s)
                self.states.add(st)
                if index == 0:
                    self.init_state = st
            self.states = sorted(self.states)
        return self

    def get_state_from_data(self, row):
        '''
                   Computes state from a single data row

                   Parameters:
                           row: data set row

                   Returns:
                           A state corresponding to this data set row.
        '''
        st = ''
        for c in self.state_columns:
            s = int(row[c])
            st = st + str(s)
        state = st
        return state

    def get_action_from_data(self, row):
        '''
                   Computes action from a single data row

                   Parameters:
                           row: data set row

                   Returns:
                           A action corresponding to this data set row.
        '''
        act = ''
        for a in self.action_columns:
            a = int(row[a])
            act = act + str(a)
        action = act
        return action

    def build_actions_list_per_state(self):
        '''
                   Builds mapping of actions per states
        '''
        for index, row in self.dataframe.iterrows():
            state = self.get_state_from_data(row)
            action = self.get_action_from_data(row)
            self.mapping[state].add(action)
        return self

    def order_actions_risk(self):
        '''
                   Computes from data an empirical risk per state and action
        '''
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
        return self

    # def get_action(self, state, mapping):
    #     action = mapping[state][0]
    #     return action
