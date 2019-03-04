# Co-ordination Confidence Based Action Selector
# ==============================================

from scipy.spatial import distance

from action_selectors.hat_actionselector import HATActionSelector


class CCActionSelector:
    def __init__(self, action_space, state_length, state_weights, cc_input_file, agent_input_file):
        """Reads the input file and stores the states and their coordination confidence values
        in a dictionary

        :param action_space: action space of gym environment
        :param state_length: number of states for the agent
        :param state_weights: indicate the weights of the state vector for distance calculation
                                0 - exact match calculation
                                1 - euclidean distance calculation
                                -1 - ignore
        :param cc_input_file: input file name of the agent states and co-ordination
                                confidence values
               file-format: each line in the format - '(state):confidence value'
        :param agent_input_file: state and action mapping file for the agent
        """
        input_file = open(cc_input_file, 'r')
        self.states = {}  # to store states(as keys), coordination confidence
        # and euclidean distance values
        self.state_weights = state_weights
        self.max_cc = -1
        self.max_cc_state = ()
        self.hat = HATActionSelector(action_space, state_length, agent_input_file)
        # read the state and create a dictionary index for each state

        for line in input_file:
            read_state = []
            temp = list(line[1:line.find(')') + 1])
            number = ''
            for character in temp:
                if character in (',', ')'):
                    read_state.append(int(number))
                    number = ''
                elif character == ' ':
                    continue
                else:
                    number = number + character

            # create a dictionary item with state, coordination confidence and euclidean distance
            for i in range(len(read_state) - 1, -1, -1):
                if self.state_weights[i] == -1:
                    read_state.pop(i)
            if tuple(read_state) not in self.states.keys():
                self.states[tuple(read_state)] = [1 - float(line[line.find(')') + 2:len(line)]), 0]

    def euclidean_distance(self, state, confidence_threshold, distance_threshold):
        """function to calculate euclidean distance and get the state with
            maximum coordination confidence

        :param state: current state of the agent
        :param confidence_threshold: a threshold value for Co-ordination Confidence
        :param distance_threshold: a threshold value for euclidean distance of the states
        # :param trace_file: file to output cc complete trace
        # :param non_existent_states_agent: file to output non visited states in demonstration
        :return: return the state with maximum co-ordination confidence value for the given state;
        """
        # calculate euclidean distance
        match_index = []
        for i in range(len(state) - 1, -1, -1):
            if self.state_weights[i] == 0:
                match_index.append(i)

        for key in self.states:
            temp = True
            if key == state:
                print('weighted: exact match found')
                self.max_cc = float(self.states[key][0])  # if the input state matches,
                # return its confidence value
                self.max_cc_state = state
                break
            else:
                key_list, state_list = list(key), list(state)
                for index in match_index:
                    if key[index] != state[index]:
                        temp = False
                        break
                    # else:
                    #     state_list.pop(index)
                    #     key_list.pop(index)
                if temp:
                    self.states[key][1] = distance.euclidean(key_list, list(state_list))
                    print('matched states', (key_list, state_list, self.states[key][1]),
                          sep=' ')

        if self.max_cc == -1:
            near_points_list = list(filter((lambda x: x[1][1] != 0 and x[1][1] <
                                                      distance_threshold), self.states.items()))
            print('\nnear_points_list: ', near_points_list)
            # print(state, file=non_existent_states_agent)
            # def return_cc_value(elem):
            #     return elem[0]
            #
            # def return_distance_value(elem):
            #     return elem[1]

            print("point 1: ", near_points_list)
            near_points_list.sort(key=lambda x: (-x[1][0], x[1][1]))
            print("point 2: ", near_points_list)

            if len(near_points_list):
                self.max_cc = max(point[1][0] for point in near_points_list)
                self.max_cc_state = list(filter((lambda x: x[1][0] == self.max_cc),
                                                near_points_list))[0][0]

        if self.max_cc < confidence_threshold:  # (1-coordination_confidence) --> as the values
            # in the file are normalized values
            # print('obtained max values: ', self.max_cc, self.max_cc_state)
            # print('failed confidence test')
            self.max_cc = -1
            self.max_cc_state = ()

    def select_action(self, state, state_weights, confidence_threshold, distance_threshold):
        """function to get an action for

        :param state: current state of the agent
        :param state_weights: indicate the weights of the satte vector for distance calculation
                                0 - exact match calculation
                                1 - euclidean distance calculation
                                -1 - ignore
        :param confidence_threshold: a threshold value for Co-ordination Confidence
        :param distance_threshold: a threshold value for euclidean distance of the states
        # :param trace_file: file to output cc complete trace
        # :param non_existent_states_agent: file to output non visited states in demonstration
        :return: return the co-ordination confidence value for the given state; -1 if state
        """
        self.max_cc = -1
        self.max_cc_state = ()
        self.state_weights = state_weights
        state = list(state)
        exact_match = True

        for key in self.states:
            self.states[key][1] = 0

        for item in self.state_weights:
            if item == 1:
                exact_match = False
                break

        if exact_match:
            for key in self.states:
                # print("exact state match needed", exact_match)
                if key == tuple(state) and float(self.states[key][0]) >= confidence_threshold:
                    self.max_cc = float(self.states[key][0])
                    self.max_cc_state = tuple(state)
        else:
            # print('weighted state match needed')
            self.euclidean_distance(tuple(state), confidence_threshold, distance_threshold)

        if self.max_cc != -1:
            # print('self.max_cc_state: ', self.max_cc_state)
            # print('self.mac_cc: ', self.max_cc)
            # print('final result: ', self.hat.select_action(self.max_cc_state), self.max_cc_state)
            return self.hat.select_action(self.max_cc_state), self.max_cc_state
        else:
            return -1, ()

# CCActionSelector = CCActionSelector([0, 1, 2, 3, 4], 10, (1, 1, 1, 1, 1, 1, 1, 1, 1, 1), "blind_coord_a1.dat",
#                                     "agent1.arff")
# obs = (1,0,0,0,0,0,1,1,1,1)
# if obs[3] != 0 or obs[4] != 0:
#     cc_agent1_weights = (1, 0, 1, 0, 0, 1, 0, 1, 1, 1)
# else:
#     cc_agent1_weights = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
# print(cc_agent1_weights)
# print(CCActionSelector.select_action(obs, cc_agent1_weights,
#                                       0.7, 2.5))

# CCActionSelector2 = CCActionSelector([0, 1, 2, 3, 4], 2, (1, 1), "blind_coord_a2.dat", "agent2.arff")
# obs = (2,6)
# cc_agent1_weights = (1, 1)
# print(CCActionSelector2.select_action(obs, cc_agent1_weights, 0.7, 2.5))
#
