import numpy as np
import random
import pickle

#laMBDA = 0.9 #if 0 then traces turned off

class LearningAgent:
    def __init__(self, num_features, num_actions, al, gamma):
        self.numFeatures = num_features
        self.numActions = num_actions
        self.alpha = al
        self.gamma = gamma
        self.Q={}
        # self.traces={}
        # self.visitedStates=[]

    def queue_update(self, al, last_state, last_action, r, k, curr_state):

        """Creates/Updates the Q table:

        Parameters
        ----------
        :param al: alpha (this allows changing alpha)
        :param last_state: last state vector (to be updated)
        :param last_action: last action (to be updated)
        :param r: reward
        :param k: duration of action (usually 1 step)
        :param curr_state: current state vector (empty if episode ended)
        """

        assert self.numFeatures == len(last_state)
        assert (0 <= last_action < self.numActions)
        self.alpha = al
        if last_state not in self.Q:
            self.Q[last_state] = [0.0] * self.numActions
        if curr_state not in self.Q:
            self.Q[curr_state] = [0.0] * self.numActions
        if len(curr_state) <= 0:
            delta = (r - self.Q[last_state][last_action])  # EndEpisode
        else:
            delta = (r + (self.gamma**k) * max(self.Q[curr_state]) - self.Q[last_state][last_action])

        self.Q[last_state][last_action] += self.alpha * delta
    
    def get_queue(self, last_state, last_action =- 1):
        if last_state not in self.Q:
            self.Q[last_state]=[0.0] * self.numActions
        if last_action < 0:
            return self.Q[last_state]
        elif 0 <= last_action < self.numActions:
            return self.Q[last_state][last_action]
        else:
            print("Invalid action")
            return 0.0

    def get_action(self, curr_state, eps):

        """Creates actions

        Parameters
        ----------
        :param curr_state: current state vector
        :param eps: exploration probability (epsilon; this allows changing epsilon)

        Returns
        -------
        :return: Returns an action corresponding to max of Q(curr_state,b), i.e., arg max_b Q(curr_state,b). If multiple actions correspond to the max value, returns one of them chosen randomly.
        """

        x = random.random()
        if (x < eps) or (curr_state not in self.Q):
            return random.choice( range(self.numActions) )
        else:
            values = np.array(self.Q[curr_state])
            return np.random.choice(np.flatnonzero(values == values.max()))

    '''
    def print_queue_vals(self, oV):
        print "RL found 'RL states':",len(self.Q)
        if oV not in self.Q:
            print "Unknown obs:",oV
        else:
            print self.Q[oV]
    '''

    def save_queue(self, filename):

        """Creates a pickle dump of the Q table

        Parameters
        ----------
        :param filename: file name of the pickle file
        """

        fs = open(filename,'w')
        pickle.dump(self.Q, fs)
        fs.close()

    def load_queue(self, filename):

        """Loads a pickle dump of the Q table into the environment

        Parameters
        ----------
        :param filename: file name of the pickle file
        """

        fs = open(filename, 'r')
        self.Q = pickle.load(fs)
        fs.close()
