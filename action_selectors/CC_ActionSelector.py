# Co-ordination Confidence Based Action Selector
# ==============================================

import random

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
            temp = list(line[1:line.find(')')+1])
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
                self.states[tuple(read_state)] = [1-float(line[line.find(')')+2:len(line)]), 0]

    def euclidean_distance(self, state, confidence_threshold, distance_threshold):
        """function to calculate euclidean distance and get the state with
            maximum coordination confidence

        :param state: current state of the agent
        :param confidence_threshold: a threshold value for Co-ordination Confidence
        :param distance_threshold: a threshold value for euclidean distance of the states
        :return: return the state with maximum co-ordination confidence value for the given state;
        """
        # check the indexes if only few positions are to be checked for exact match
        match_index = []
        for i in range(len(state) - 1, -1, -1):
            if self.state_weights[i] == 0:
                match_index.append(i)

        for key in self.states:
            temp = True
            # if an exact match of the complete input state is found
            if key == state:
                self.max_cc = float(self.states[key][0])  # if the input state matches,
                self.max_cc_state = state
                if self.max_cc >= confidence_threshold:
                    return
                else:
                    self.max_cc = -1
                    self.max_cc_state = ()
                    return
            else:
                # check the indexes of the states with that of the input state
                key_list, state_list = list(key), list(state)
                for index in match_index:
                    if key[index] != state[index]:
                        temp = False
                        break
                # if the index match passes
                if temp:
                    # calculate euclidean distance
                    self.states[key][1] = distance.euclidean(key_list, list(state_list))

        # get the list of points which satisfied the above conditions
        # (exclude those with 0 distance from the states list)
        points_list = list(filter(lambda x: x[1][1] != 0, self.states.items()))
        # get the least distance calculated
        points_list.sort(key=lambda x: x[1][1])
        if len(points_list):
            min_distance = points_list[0][1][1]
            # distance threshold check
            if min_distance > distance_threshold:
                return

            # if distance threshold check is passed, get all the points with that min distance
            closest_point_list = list(
                filter(lambda x: x[1][1] == min_distance and x[1][1] >= confidence_threshold, points_list))

            # if multiple points are closest
            if len(closest_point_list) > 1:
                # get the list (or may be just one) with highest confidence
                closest_point_list.sort(key=lambda x: -x[1][0])
                max_confidence = closest_point_list[0][1][0]
                max_confidence_list = list(filter(lambda x: x[1][0] == max_confidence, closest_point_list))

                # if multiple matches with highest confidence found
                if len(max_confidence_list) > 1:
                    point = random.choice(max_confidence_list)
                    self.max_cc = point[1][0]
                    self.max_cc_state = point[0]

                # if only one state with highest confidence is found
                elif len(max_confidence_list):
                    self.max_cc = max_confidence_list[0][1][0]
                    self.max_cc_state = max_confidence_list[0][0]
                    return

            # if only one closest point with min distance found
            elif len(closest_point_list):
                self.max_cc = closest_point_list[0][1][0]
                self.max_cc_state = closest_point_list[0][0]
                return

    def select_action(self, state, state_weights, confidence_threshold, distance_threshold):
        """function to get an action for

        :param state: current state of the agent
        :param state_weights: indicate the weights of the satte vector for distance calculation
                                0 - exact match calculation
                                1 - euclidean distance calculation
                                -1 - ignore
        :param confidence_threshold: a threshold value for Co-ordination Confidence
        :param distance_threshold: a threshold value for euclidean distance of the states
        :return: return the co-ordination confidence value for the given state; -1 if state
        """
        self.max_cc = -1
        self.max_cc_state = ()
        self.state_weights = state_weights
        state = list(state)
        exact_match = True

        # check if the state needs an exact match
        for key in self.states:
            self.states[key][1] = 0

        for item in self.state_weights:
            if item == 1:
                exact_match = False
                break

        # if exact match is needed, check the states for a match and test its confidence
        if exact_match:
            for key in self.states:
                if key == tuple(state) and float(self.states[key][0]) >= confidence_threshold:
                    self.max_cc = float(self.states[key][0])
                    self.max_cc_state = tuple(state)

        # if an exact match is not required (distance matches are allowed)
        else:
            self.euclidean_distance(tuple(state), confidence_threshold, distance_threshold)

        if self.max_cc != -1:
            return self.hat.select_action(self.max_cc_state)[0], self.max_cc
        else:
            return -1, -1

# CCActionSelector = CCActionSelector([0, 1, 2, 3, 4, 5, 6, 7], 11, (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0),
# "blind_coord_a1.dat", "agent1.arff")
# obs = (1,0,0,0,0,0,1,1,1,1,0)
# if obs[3] != 0 or obs[4] != 0:
#     cc_agent1_weights = (1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0)
# else:
#     cc_agent1_weights = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0)
# print(cc_agent1_weights)
# print(CCActionSelector.select_action(obs, cc_agent1_weights, 0.7, 2.5))

# CCActionSelector2 = CCActionSelector([0, 1, 2, 3, 4], 13, (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
#                                      "bdn_coord_a1.dat", "symmetric_demo_bdn_noise_agent_v1.arff")
# obs = (4,1,1,-1,16,3,13,3,4,0,-50,2,50)
# cc_agent1_weights =(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
# print(CCActionSelector2.select_action(obs, cc_agent1_weights, 0.7, 2.5))
