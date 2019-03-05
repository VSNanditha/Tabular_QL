import math
import os
import pickle
import random

import numpy as np
from mpi4py import MPI

from action_selectors.CC_ActionSelector import CCActionSelector
from action_selectors.CHAT_ActionSelector import CHATActionSelector


# lambda = 0.9  # if 0 then traces turned off


class LearningAgent:
    def __init__(self, num_features, num_actions, alpha, gamma):

        """Initialization

        Parameters
        ----------
        :param num_features: number of observations (int)
        :param num_actions: number of actions (int)
        :param alpha: learning rate (float --> 0.0 - 1.0)
        :param gamma: discount factor (float --> 0.0 - 1.0)
        """

        self.num_features = num_features
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.Q = {}
        # self.traces={}
        # self.visitedStates=[]

    def q_update(self, alpha, last_state, last_action, r, k, curr_state):

        """Creates/Updates the Q table:

        Parameters
        ----------
        :param alpha: learning rate
        :param last_state: last state vector; to be updated (observation vector)
        :param last_action: last action; to be updated (int)
        :param r: reward (float)
        :param k: duration of action; usually 1 step (int)
        :param curr_state: current state vector; empty if episode ended (observation vector)
        """
        # assert self.num_features == len(last_state)
        # assert (0 <= last_action < self.num_actions)
        self.alpha = alpha
        if last_state not in self.Q:
            self.Q[last_state] = [0.0] * self.num_actions
        if curr_state not in self.Q:
            self.Q[curr_state] = [0.0] * self.num_actions
        if len(curr_state) <= 0:
            delta = (r - self.Q[last_state][last_action])  # EndEpisode
        else:
            delta = (r + (self.gamma ** k) * max(self.Q[curr_state]) - self.Q[last_state][last_action])
        self.Q[last_state][last_action] += self.alpha * delta

    def get_q(self, last_state, last_action=-1):
        if last_state not in self.Q:
            self.Q[last_state] = [0.0] * self.num_actions
        if last_action < 0:
            return self.Q[last_state]
        elif 0 <= last_action < self.num_actions:
            return self.Q[last_state][last_action]
        else:
            print("Invalid action")
            return 0.0

    def get_action(self, curr_state, eps):

        """Creates actions

        Parameters
        ----------
        :param curr_state: current state vector
        :param eps: exploration probability epsilon; this allows changing epsilon (float 0.0 - 1.0)

        Returns
        -------
        :return: Returns an action corresponding to max of Q(curr_state,b), i.e., arg max_b Q(curr_state,b).
                    If multiple actions correspond to the max value, returns one of them chosen randomly.
        """

        if random.random() <= eps or curr_state not in self.Q:
            return random.choice(range(self.num_actions))
        else:
            values = np.array(self.Q[curr_state])
            print(values, values.max)
            return np.random.choice(np.flatnonzero(values == values.max()))

    def save_q(self, filename):

        """Creates a pickle dump of the Q table

        Parameters
        ----------
        :param filename: file name of the pickle file (string)
        """

        fs = open(filename, 'wb')
        pickle.dump(self.Q, fs)
        fs.close()

    def load_q(self, filename):

        """Loads a pickle dump of the Q table into the environment

        Parameters
        ----------
        :param filename: file name of the pickle file
        """

        fs = open(filename, 'rb')
        self.Q = pickle.load(fs)
        fs.close()


def learn(env, num_features_agent1, num_features_agent2, num_actions_agent1, num_actions_agent2, alpha=1e-3,
          gamma=0.98, eps=0.05, max_episodes=100000, horizontal_moving_obstacles=0, vertical_moving_obstacles=0):
    """Learn function for tabular Q-Learning

    Parameters
    ----------
    :param env: gym environment
    :param num_features_agent1: number of observations for Agent 1(int)
    :param num_features_agent2: number of observations for Agent 2(int)
    :param num_actions_agent1: number of actions for agent 1(int)
    :param num_actions_agent2: number of actions for agent 2(int)
    :param alpha: learning rate (float --> 0.0 - 1.0)
    :param gamma: discount factor (float --> 0.0 - 1.0)
    :param eps: exploration rate
    :param max_episodes: maximum number of episodes
    :param horizontal_moving_obstacles: number of horizontally moving obstacles in the domain
    :param vertical_moving_obstacles: number of vertically moving obstacles in the domain
    Returns
    -------
    :return: agent1, agent2 - LearningAgent objects
    """

    agent1 = LearningAgent(num_features_agent1, num_actions_agent1, alpha, gamma)
    agent2 = LearningAgent(num_features_agent2, num_actions_agent2, alpha, gamma)

    episode_rewards = [0.0]
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    filename = 'episode_rewards_tabQL_' + str(rank) + '.raw'
    f = open(filename, 'a+')
    for i in range(max_episodes):
        steps = 0
        if env.unwrapped.env_name() in ("GuideDog-v1"):
            env.unwrapped.get_moving_objects(horizontal_moving_obstacles, vertical_moving_obstacles)
        obs, done = env.reset(), False
        while not done:
            # env.render()
            steps += 1
            action1 = int(agent1.get_action(tuple(obs[0]), eps))
            action2 = int(agent2.get_action(tuple(obs[1]), eps))
            env_action = (action1, action2)
            new_obs, rew, done, info = env.step(env_action)
            episode_rewards[-1] += rew
            if not done:
                agent1.q_update(alpha, tuple(obs[0]), action1, rew, 1, tuple(new_obs[0]))
                agent2.q_update(alpha, tuple(obs[1]), action2, rew, 1, tuple(new_obs[1]))
            else:
                agent1.q_update(alpha, tuple(obs[0]), action1, rew, 1, tuple([]))
                agent2.q_update(alpha, tuple(obs[1]), action2, rew, 1, tuple([]))
                print(i, episode_rewards[-1], steps)
                print(i, episode_rewards[-1], steps, file=f)
                break
            obs = new_obs
        episode_rewards.append(0.0)
    f.close()
    return agent1, agent2


def learn_from_pickle(env, pickle1, pickle2, num_features_agent1, num_features_agent2, num_actions_agent1,
                      num_actions_agent2, alpha=1e-3, gamma=0.98, max_episodes=100000, horizontal_moving_obstacles=0,
                      vertical_moving_obstacles=0):
    """Learn function for tabular Q-Learning with pickle load

    Parameters
    ----------
    :param env: gym environment
    :param pickle1: saved Q values for agent 1
    :param pickle2: saved Q values for agent 2
    :param num_features_agent1: number of observations for Agent 1(int)
    :param num_features_agent2: number of observations for Agent 2(int)
    :param num_actions_agent1: number of actions for agent 1(int)
    :param num_actions_agent2: number of actions for agent 2(int)
    :param alpha: learning rate (float --> 0.0 - 1.0)
    :param gamma: discount factor (float --> 0.0 - 1.0)
    :param max_episodes: maximum number of episodes
    :param horizontal_moving_obstacles: number of horizontally moving obstacles in the domain
    :param vertical_moving_obstacles: number of vertically moving obstacles in the domain
    Returns
    -------
    :return: agent1, agent2 - LearningAgent objects
    """
    agent1 = LearningAgent(num_features_agent1, num_actions_agent1, alpha, gamma)
    agent2 = LearningAgent(num_features_agent2, num_actions_agent2, alpha, gamma)
    agent1.load_q(pickle1)
    agent2.load_q(pickle2)

    episode_rewards = [0.0]
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    ep_rew = 'episode_rewards' + str(rank) + '.raw'
    ep_reward = open(ep_rew, 'a+')

    for i in range(max_episodes):
        steps = 0
        if env.unwrapped.env_name() in ("GuideDog-v0", "GuideDog-v1"):
            env.unwrapped.get_moving_objects(horizontal_moving_obstacles, vertical_moving_obstacles)
        obs, done = env.reset(), False
        while not done:
            # env.render()
            steps += 1
            action1 = agent1.get_action(tuple(obs[0]), 0.0)
            action2 = agent2.get_action(tuple(obs[1]), 0.0)
            env_action = (action1, action2)
            new_obs, rew, done, info = env.step(env_action)
            obs = new_obs
            episode_rewards[-1] += rew[0] + rew[1]

            if done:
                print('episode complete', i, max_episodes)
                print(i, episode_rewards[-1], steps, file=ep_reward)
                break
        episode_rewards.append(0.0)
    ep_reward.close()
    return agent1, agent2


def human_demonstration(env, num_features_agent1, num_features_agent2, num_actions_agent1, num_actions_agent2,
                        alpha=1e-3, gamma=0.98, max_episodes=100000, horizontal_moving_obstacles=0,
                        vertical_moving_obstacles=0):
    """Learn function for tabular Q-Learning with human input values

    Parameters
    ----------
    :param env: gym environment
    :param num_features_agent1: number of observations for Agent 1(int)
    :param num_features_agent2: number of observations for Agent 2(int)
    :param num_actions_agent1: number of actions for agent 1(int)
    :param num_actions_agent2: number of actions for agent 2(int)
    :param alpha: learning rate (float --> 0.0 - 1.0)
    :param gamma: discount factor (float --> 0.0 - 1.0)
    :param max_episodes: maximum number of episodes
    :param horizontal_moving_obstacles: number of horizontally moving obstacles in the domain
    :param vertical_moving_obstacles: number of vertically moving obstacles in the domain
    Returns
    -------
    :return: agent1, agent2 - LearningAgent objects
    """
    agent1 = LearningAgent(num_features_agent1, num_actions_agent1, alpha, gamma)
    agent2 = LearningAgent(num_features_agent2, num_actions_agent2, alpha, gamma)

    episode_rewards = [0.0]

    agent1_file, agent2_file, agents_file = open('agent1.arff', 'a+'), open('agent2.arff', 'a+'), open('agents.arff',
                                                                                                       'a+')
    ep_reward = open('episode_rewards.raw', 'a+')
    demonstration = open('demonstration.raw', 'a+')
    if os.stat('agent1.arff').st_size == 0:
        # agent1_file.write(
        #     '@relation IndividualActionPredictor\n@attribute \'ax\' numeric\n@attribute \'ay\' numeric\n@attribute'
        #     ' \'grasp\' numeric\n@attribute \'ox\' numeric\n@attribute \'oy\' numeric\n@attribute \'act\' {\'0\', '
        #     '\'1\', \'2\', \'3\', \'4\'}\n@data\n')
        agent1_file.write('@relation IndividualActionPredictor\n@attribute \'corner0\' numeric\n@attribute \'corner1\' '
                          'numeric\n@attribute \'corner2\' numeric\n@attribute \'corner3\' numeric\n@attribute \''
                          'corner4\' numeric\n@attribute \'corner5\' numeric\n@attribute \'corner6\' numeric\n'
                          '@attribute \'corner7\' numeric\n@attribute \'corner8\' numeric\n@attribute \'corner9\' '
                          'numeric\n@attribute \'communication\' numeric\n@attribute \'act\' {\'0\', \'1\', \'2\', '
                          '\'3\', \'4\', \'5\', \'6\', \'7\'}\n@data\n')
    if os.stat('agent2.arff', ).st_size == 0:
        # agent2_file.write(
        #     '@relation IndividualActionPredictor\n@attribute \'ax\' numeric\n@attribute \'ay\' numeric\n'
        #     '@attribute \'grasp\' numeric\n@attribute \'ox\' numeric\n@attribute \'oy\' numeric\n'
        #     '@attribute \'act\' {\'0\', \'1\', \'2\', \'3\', \'4\'}\n@data\n')
        agent2_file.write('@relation IndividualActionPredictor\n@attribute \'ax\' numeric\n@attribute \'ay\' '
                          'numeric\n@attribute \'communication\' numeric\n@attribute \'act\' {\'0\', \'1\', \'2\', '
                          '\'3\', \'4\'}\n@data\n')
    if os.stat('agents.arff', ).st_size == 0:
        # agents_file.write(
        #     '@relation JointActionPredictor\n@attribute \'a1x\' numeric\n@attribute \'a1y\' numeric\n'
        #     '@attribute \'a1grasp\' numeric\n@attribute \'a1ox\' numeric\n@attribute \'a1oy\' numeric\n'
        #     '@attribute \'a2x\' numeric\n@attribute \'a2y\' numeric\n@attribute \'a2grasp\' numeric\n'
        #     '@attribute \'a2ox\' numeric\n@attribute \'a2oy\' numeric\n@attribute \'act\' {\'00\',	'
        #     '\'10\',	\'20\',	\'30\',	\'40\',	\'01\',	\'11\',	\'21\',	\'31\',	\'41\',	\'02\',	\'12\',	'
        #     '\'22\',	\'32\',	\'42\',	\'03\',	\'13\',	\'23\',	\'33\',	\'43\',	\'04\',	\'14\',	\'24\',	'
        #     '\'34\',	\'44\'}\n@data\n')
        agents_file.write('@relation JointActionPredictor\n@attribute \'corner0\' numeric\n@attribute \''
                          'corner1\' numeric\n@attribute \'corner2\' numeric\n@attribute \'corner3\' '
                          'numeric\n@attribute \'corner4\' numeric\n@attribute \'corner5\' numeric\n'
                          '@attribute \'corner6\' numeric\n@attribute \'corner7\' numeric\n@attribute \''
                          'corner8\' numeric\n@attribute \'corner9\' numeric\n@attribute \'communication\' numeric\n'
                          '@attribute \'ax\' numeric\n@attribute \'ay\' numeric\n@attribute \'communication\' '
                          'numeric\n@attribute \'act\' {\'00\',	\'10\',	\'20\',	\'30\', \''
                          '40\',	\'01\',	\'11\',	\'21\',	\'31\',	\'41\',	\'02\',	\'12\',	\'22\',	\'32\',	\''
                          '42\', \'03\',	\'13\',	\'23\',	\'33\',	\'43\',	\'04\',	\'14\',	\'24\',	\'34\',	\'44\', '
                          '\'50\', \'51\', \'52\', \'53\', \'54\', \'60\', \'61\', \'62\', \'63\', \'64\', \'70\', '
                          '\'71\', \'72\', \'73\', \'74\', }\n@data\n')

    for i in range(max_episodes):
        steps = 0
        if env.unwrapped.env_name() in ("GuideDog-v1"):
            env.unwrapped.get_moving_objects(horizontal_moving_obstacles, vertical_moving_obstacles)
        obs, done = env.reset(), False
        print('Episode: ', i + 1, file=demonstration)
        while not done:
            env.render()
            steps += 1
            action1 = int(input("Agent 1 action"))
            action2 = int(input("Agent 2 action"))
            env_action = (action1, action2)
            print('s1', tuple(obs[0]), sep=':', file=demonstration)
            print('s2', tuple(obs[1]), sep=':', file=demonstration)
            print('o1', *obs[0], sep=':', file=demonstration)
            print('o2', *obs[1], sep=':', file=demonstration)
            print('ja intended', action1, action2, sep=':', file=demonstration)
            new_obs, rew, done, info = env.step(env_action)
            if not done:
                agent1.q_update(alpha, tuple(obs[0]), action1, rew, 1, tuple(new_obs[0]))
                agent2.q_update(alpha, tuple(obs[1]), action2, rew, 1, tuple(new_obs[1]))
            episode_rewards[-1] += rew
            print(*obs[0], sep=',', end=',', file=agent1_file)
            print('\'' + str(action1) + '\' ', file=agent1_file)
            print(*obs[0], sep=',', end=',')
            print('\'' + str(action1) + '\' ')
            print(*obs[1], sep=',', end=',', file=agent2_file)
            print('\'' + str(action2) + '\' ', file=agent2_file)
            print(*obs[1], sep=',', end=',')
            print('\'' + str(action2) + '\' ')
            print(*obs[0], sep=',', end=',', file=agents_file)
            print(*obs[1], sep=',', end=',', file=agents_file)
            print('\'' + str(action1), end='', file=agents_file)
            print(str(action2) + '\' ', file=agents_file)
            if done:
                agent1.q_update(alpha, tuple(obs[0]), action1, rew, 1, tuple([]))
                agent2.q_update(alpha, tuple(obs[1]), action2, rew, 1, tuple([]))
                print('episode complete', i, max_episodes)
                print(i, episode_rewards[-1], steps, file=ep_reward)
                print('Steps:', steps, sep='', file=demonstration)
                break
            obs = new_obs
        episode_rewards.append(0.0)
    ep_reward.close()
    agent1_file.close()
    agent2_file.close()
    agents_file.close()
    demonstration.close()
    return agent1, agent2


def coord_chat_learn(env, num_features_agent1, num_features_agent2, num_actions_agent1, num_actions_agent2,
                     cc_agent1, cc_agent2, cc_agent1_weights, cc_agent2_weights, chat_agent1, chat_agent2,
                     alpha=1e-3, gamma=0.98, eps=0.05, min_psi=0.05, max_episodes=100000, confidence_threshold=0.7,
                     distance_threshold=10.0, horizontal_moving_obstacles=0, vertical_moving_obstacles=0):
    """Learn function for tabular Q-Learning with CHAT and Co-ordination confidence action selectors

    Parameters
    ----------
    :param env: gym environment
    :param num_features_agent1: number of observations for Agent 1(int)
    :param num_features_agent2: number of observations for Agent 2(int)
    :param num_actions_agent1: number of actions for agent 1(int)
    :param num_actions_agent2: number of actions for agent 2(int)
    :param cc_agent1: input file for coordination confidence values of agent1
    :param cc_agent2: input file for coordination confidence values of agent2
    :param cc_agent1_weights: weights of the observation of agent 1
    :param cc_agent2_weights: weights of the observation of agent 2
    :param chat_agent1: input file for state and action values of agent1 for CHAT
    :param chat_agent2: input file for state and action values of agent2 for CHAT
    :param alpha: learning rate (float --> 0.0 - 1.0)
    :param gamma: discount factor (float --> 0.0 - 1.0)
    :param eps: exploration rate
    :param min_psi: minimum psi value; threshold parameter for action selectors; Decaying psi value used
    :param max_episodes: maximum number of episodes
    :param confidence_threshold: threshold for CHAT and CC Action selectors
    :param distance_threshold: euclidean distance threshold for coordination confidence
    :param horizontal_moving_obstacles: number of horizontally moving obstacles in the domain
    :param vertical_moving_obstacles: number of vertically moving obstacles in the domain
    Returns
    -------
    :return: agent1, agent2 - LearningAgent objects
    """
    agent1, agent2 = LearningAgent(num_features_agent1, num_actions_agent1, alpha, gamma), \
                     LearningAgent(num_features_agent2, num_actions_agent2, alpha, gamma)

    episode_rewards, psi = [0.0], 1.0
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    ep_reward = 'episode_rewards_coordchat_' + str(rank) + '.raw'
    action_choice = 'action_choice_' + str(rank) + '.raw'
    # action_trace = 'action_trace_' + str(rank) + '.raw'
    ep_reward = open(ep_reward, 'a+')
    action_choice = open(action_choice, 'a+')
    # action_trace = open(action_trace, 'a+')

    if env.unwrapped.env_name() == "GuideDog-v1":
        action_space_agent1 = env.unwrapped.action_values(env.unwrapped.actions_agent1)
        action_space_agent2 = env.unwrapped.action_values(env.unwrapped.actions_agent2)
        chat1 = CHATActionSelector(action_space_agent1, num_features_agent1, chat_agent1)
        chat2 = CHATActionSelector(action_space_agent2, num_features_agent2, chat_agent2)
        cc1 = CCActionSelector(action_space_agent1, num_features_agent1, cc_agent1_weights, cc_agent1, chat_agent1)
        cc2 = CCActionSelector(action_space_agent2, num_features_agent2, cc_agent2_weights, cc_agent2, chat_agent2)
    else:
        action_space = env.unwrapped.action_values(env.unwrapped.actions)
        chat1 = CHATActionSelector(action_space, num_features_agent1, chat_agent1)
        chat2 = CHATActionSelector(action_space, num_features_agent2, chat_agent2)
        cc1 = CCActionSelector(action_space, num_features_agent1, cc_agent1_weights, cc_agent1, chat_agent1)
        cc2 = CCActionSelector(action_space, num_features_agent2, cc_agent2_weights, cc_agent2, chat_agent2)

    for i in range(max_episodes):
        steps = 0
        cc_action_choice, chat_action_choice, chat_equals_cc, chat_not_equals_cc, q_choice = 0, 0, 0, 0, 0
        if env.unwrapped.env_name() in ("GuideDog-v1"):
            env.unwrapped.get_moving_objects(horizontal_moving_obstacles, vertical_moving_obstacles)
        obs, done = env.reset(), False
        # print('Episode:', i, file=action_trace)
        while not done:
            # env.render()
            steps += 1

            # Action selection
            if env.unwrapped.env_name() in ("GuideDog-v0", "GuideDog-v1"):
                if obs[0][3] != 0 or obs[0][4] != 0:
                    cc_agent1_weights = (1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1)
                else:
                    cc_agent1_weights = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
            # print('cc_agent1_weights: ', cc_agent1_weights, file=action_trace)
            cc_action1 = cc1.select_action(obs[0], cc_agent1_weights, confidence_threshold, distance_threshold)
            cc_action2 = cc2.select_action(obs[1], cc_agent2_weights, confidence_threshold, distance_threshold)
            chat_action1 = chat1.select_action(obs[0], confidence_threshold)
            chat_action2 = chat2.select_action(obs[1], confidence_threshold)

            if env.unwrapped.env_name() == "BlockDudes-v0":
                HAT_Mapping = {'0': 2, '1': 3, '2': 1, '3': 4, '4': 0}
                if cc_action1[0] != -1:
                    cc_action1 = list(cc_action1)
                    cc_action1[0] = HAT_Mapping[str(cc_action1[0])]
                    cc_action1 = tuple(cc_action1)
                if cc_action2[0] != -1:
                    cc_action2 = list(cc_action2)
                    cc_action2[0] = HAT_Mapping[str(cc_action2[0])]
                    cc_action2 = tuple(cc_action2)
                if chat_action1 != -1:
                    chat_action1 = HAT_Mapping[str(chat_action1)]
                if chat_action2 != -1:
                    chat_action2 = HAT_Mapping[str(chat_action2)]

            if random.random() <= psi:
                if cc_action1[0] != -1:
                    action1 = cc_action1[0]
                    cc_action_choice += 1
                    # print('cc action 1', end=' ', file=action_trace)
                    if cc_action1[0] == chat_action1:
                        chat_equals_cc += 1
                    else:
                        chat_not_equals_cc += 1
                else:
                    action1 = agent1.get_action(tuple(obs[0]), eps)
                    q_choice += 1
                    # print('q action 1', end=' ', file=action_trace)
            else:
                action1 = agent1.get_action(tuple(obs[0]), eps)
                q_choice += 1
                # print('q action 1', end=' ', file=action_trace)

            if random.random() <= psi:
                if cc_action2[0] != -1:
                    action2 = cc_action2[0]
                    cc_action_choice += 1
                    # print('cc action 2', end=' ', file=action_trace)
                    if cc_action2[0] == chat_action2:
                        chat_equals_cc += 1
                    else:
                        chat_not_equals_cc += 1
                else:
                    action2 = agent2.get_action(tuple(obs[1]), eps)
                    q_choice += 1
                    # print('q action 2', end=' ', file=action_trace)
            else:
                action2 = agent2.get_action(tuple(obs[1]), eps)
                q_choice += 1
                # print('q action 2', end=' ', file=action_trace)

            env_action = (action1, action2)
            # print(obs, action1, action2, cc_action1, cc_action2, chat_action1, chat_action2,
            #       psi, file=action_trace)
            new_obs, rew, done, info = env.step(env_action)
            if env.unwrapped.env_name() in ("ObjectTransport-v0"):
                if not done:
                    agent1.q_update(alpha, tuple(obs[0]), action1, rew[0], 1, tuple(new_obs[0]))
                    agent2.q_update(alpha, tuple(obs[1]), action2, rew[1], 1, tuple(new_obs[1]))
                episode_rewards[-1] += rew[0] + rew[1]
                if done:
                    agent1.q_update(alpha, tuple(obs[0]), action1, rew[0], 1, tuple([]))
                    agent2.q_update(alpha, tuple(obs[1]), action2, rew[1], 1, tuple([]))
                    print(i, episode_rewards[-1], steps, file=ep_reward)
                    print(i, episode_rewards[-1], steps)
                    print(i, cc_action_choice, chat_action_choice, q_choice, chat_equals_cc, chat_not_equals_cc,
                          file=action_choice)
                    break
            else:
                if not done:
                    agent1.q_update(alpha, tuple(obs[0]), action1, rew, 1, tuple(new_obs[0]))
                    agent2.q_update(alpha, tuple(obs[1]), action2, rew, 1, tuple(new_obs[1]))
                episode_rewards[-1] += rew
                if done:
                    agent1.q_update(alpha, tuple(obs[0]), action1, rew, 1, tuple([]))
                    agent2.q_update(alpha, tuple(obs[1]), action2, rew, 1, tuple([]))
                    print(i, episode_rewards[-1], steps, file=ep_reward)
                    print(i, episode_rewards[-1], steps)
                    print(i, cc_action_choice, chat_action_choice, q_choice, chat_equals_cc, chat_not_equals_cc,
                          file=action_choice)
                    break
            obs = new_obs
        episode_rewards.append(0.0)
        psi = psi * math.exp(math.log(min_psi) / max_episodes)
    ep_reward.close()
    action_choice.close()
    # action_trace.close()
    return agent1, agent2


def chat_learn(env, num_features_agent1, num_features_agent2, num_actions_agent1, num_actions_agent2,
               chat_agent1, chat_agent2, alpha=1e-3, gamma=0.98, eps=0.05, min_psi=0.05, max_episodes=100000,
               confidence_threshold=0.7, horizontal_moving_obstacles=0, vertical_moving_obstacles=0):
    """Learn function for tabular Q-Learning with CHAT and Co-ordination confidence action selectors

    Parameters
    ----------
    :param env: gym environment
    :param num_features_agent1: number of observations for Agent 1(int)
    :param num_features_agent2: number of observations for Agent 2(int)
    :param num_actions_agent1: number of actions for agent 1(int)
    :param num_actions_agent2: number of actions for agent 2(int)
    :param chat_agent1: input file for state and action values of agent1 for CHAT
    :param chat_agent2: input file for state and action values of agent2 for CHAT
    :param alpha: learning rate (float --> 0.0 - 1.0)
    :param gamma: discount factor (float --> 0.0 - 1.0)
    :param eps: exploration rate
    :param min_psi: minimum psi value; threshold parameter for action selectors; Decaying psi value used
    :param max_episodes: maximum number of episodes
    :param confidence_threshold: threshold for CHAT and CC Action selectors
    :param horizontal_moving_obstacles: number of horizontally moving obstacles in the domain
    :param vertical_moving_obstacles: number of vertically moving obstacles in the domain
    Returns
    -------
    :return: agent1, agent2 - LearningAgent objects
    """
    agent1, agent2 = LearningAgent(num_features_agent1, num_actions_agent1, alpha, gamma), \
                     LearningAgent(num_features_agent2, num_actions_agent2, alpha, gamma)

    episode_rewards, psi = [0.0], 1.0
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    ep_reward = 'episode_rewards_chat_' + str(rank) + '.raw'
    ep_reward = open(ep_reward, 'a+')
    # action_choice = open(action_choice, 'a+')
    # action_trace = open(action_trace, 'a+')

    # actions = env.action_space
    # action_space = list(actions.index(x) for x in actions)
    if env.unwrapped.env_name() == "GuideDog-v1":
        action_space_agent1 = env.unwrapped.action_values(env.unwrapped.actions_agent1)
        action_space_agent2 = env.unwrapped.action_values(env.unwrapped.actions_agent2)
        chat1 = CHATActionSelector(action_space_agent1, num_features_agent1, chat_agent1)
        chat2 = CHATActionSelector(action_space_agent2, num_features_agent2, chat_agent2)
    else:
        action_space = env.unwrapped.action_values(env.unwrapped.actions)
        chat1 = CHATActionSelector(action_space, num_features_agent1, chat_agent1)
        chat2 = CHATActionSelector(action_space, num_features_agent2, chat_agent2)

    for i in range(max_episodes):
        steps = 0
        chat_action_choice, q_choice = 0, 0
        if env.unwrapped.env_name() in ("GuideDog-v1"):
            env.unwrapped.get_moving_objects(horizontal_moving_obstacles, vertical_moving_obstacles)
        obs, done = env.reset(), False
        # print('Episode:', i, file=action_trace)
        while not done:
            # env.render()
            steps += 1

            # Action selection
            chat_action1 = chat1.select_action(obs[0], confidence_threshold)
            chat_action2 = chat2.select_action(obs[1], confidence_threshold)

            if random.random() <= psi:
                if chat_action1 != -1:
                    action1 = chat_action1
                    chat_action_choice += 1
                    # print('chat action 1', end=' ', file=action_trace)
                else:
                    action1 = agent1.get_action(tuple(obs[0]), eps)
                    q_choice += 1
                    # print('q action 1', end=' ', file=action_trace)
            else:
                action1 = agent1.get_action(tuple(obs[0]), eps)
                q_choice += 1
                # print('q action 1', end=' ', file=action_trace)

            if random.random() <= psi:
                if chat_action2 != -1:
                    action2 = chat_action2
                    chat_action_choice += 1
                    # print('chat action 2', end=' ', file=action_trace)
                else:
                    action2 = agent2.get_action(tuple(obs[1]), eps)
                    q_choice += 1
                    # print('q action 2', end=' ', file=action_trace)
            else:
                    action2 = agent2.get_action(tuple(obs[1]), eps)
                    q_choice += 1
                    # print('q action 2', end=' ', file=action_trace)

            env_action = (action1, action2)
            # print(env_action, chat_action1, chat_action2, chat_action_choice, q_choice, file=action_trace)
            new_obs, rew, done, info = env.step(env_action)
            if env.unwrapped.env_name() in ("ObjectTransport-v0"):
                if not done:
                    agent1.q_update(alpha, tuple(obs[0]), action1, rew[0], 1, tuple(new_obs[0]))
                    agent2.q_update(alpha, tuple(obs[1]), action2, rew[1], 1, tuple(new_obs[1]))
                episode_rewards[-1] += rew[0] + rew[1]
                if done:
                    agent1.q_update(alpha, tuple(obs[0]), action1, rew[0], 1, tuple([]))
                    agent2.q_update(alpha, tuple(obs[1]), action2, rew[1], 1, tuple([]))
                    print(i, episode_rewards[-1], steps, file=ep_reward)
                    print(i, episode_rewards[-1], steps)
                    break
            else:
                if not done:
                    agent1.q_update(alpha, tuple(obs[0]), action1, rew, 1, tuple(new_obs[0]))
                    agent2.q_update(alpha, tuple(obs[1]), action2, rew, 1, tuple(new_obs[1]))
                episode_rewards[-1] += rew
                if done:
                    agent1.q_update(alpha, tuple(obs[0]), action1, rew, 1, tuple([]))
                    agent2.q_update(alpha, tuple(obs[1]), action2, rew, 1, tuple([]))
                    print(i, episode_rewards[-1], steps, file=ep_reward)
                    print(i, episode_rewards[-1], steps)
                    break
            obs = new_obs
        episode_rewards.append(0.0)
        psi = psi * math.exp(math.log(min_psi) / max_episodes)
    ep_reward.close()
    return agent1, agent2
