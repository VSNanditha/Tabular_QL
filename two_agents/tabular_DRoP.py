import math
import random

from mpi4py import MPI

import two_agents.tabular_ql as ql
from action_selectors.cc_actionselector import CCActionSelector
from action_selectors.hat_actionselector import HATActionSelector


class DRoPLearningAgent:
    def __init__(self, num_features, num_actions, alpha, gamma):
        """Initialization

        Parameters
        ----------
        :param num_features: number of observations (int)
        :param num_actions: number of actions (int)
        :param alpha: learning rate (float --> 0.0 - 1.0)
        :param gamma: discount factor (float --> 0.0 - 1.0)"
        """
        self.lA = ql.LearningAgent(num_features, num_actions, alpha, gamma)
        self.CPCC = ql.LearningAgent(num_features, 1, alpha, gamma)
        self.CPCHAT = ql.LearningAgent(num_features, 1, alpha, gamma)
        self.CQ = ql.LearningAgent(num_features, 1, alpha, gamma)
        self.input_alpha = alpha
        self.alpha = alpha
        self.conf = 0.0
        self.CCUsed = False
        self.CHATUsed = False

    '''
    def getActionHD(self, obs, eps, classifierAct):
        cpVal = self.CP.get_q(obs, 0)
        cqVal = self.CQ.get_q(obs, 0)
        if cpVal > cqVal:
            self.PUsed = True
            return classifierAct
        else:
            return self.lA.get_action( obs, eps )
    '''

    def get_action_sd(self, obs, eps, chat_act, cc_act, chat_conf, cc_conf):
        """

        :param obs: observation
        :param eps: exploration rate
        :param chat_act: chat action for the observation
        :param cc_act: cc action for the observation
        :param chat_conf: chat confidence for the observation
        :param cc_conf: cc confidence for the observation
        :return:
        """
        cpCCVal = self.CPCC.get_q(obs, 0)
        cpCHATVal = self.CPCHAT.get_q(obs, 0)
        cqVal = self.CQ.get_q(obs, 0)
        r = max(abs(cpCCVal), abs(cpCHATVal), abs(cqVal))
        if r > 0:
            rpCC, rpCHAT, rq = cpCCVal / r, cpCHATVal / r, cqVal / r
        else:
            rpCC, rpCHAT, rq = cpCCVal, cpCHATVal, cqVal
        p1 = (math.tanh(rpCC) + 1) / (math.tanh(rpCC) + math.tanh(rpCHAT) + math.tanh(rq) + 3)
        p2 = (math.tanh(rpCC) + 1 + math.tanh(rpCHAT) + 1) / (math.tanh(rpCC) + math.tanh(rpCHAT) + math.tanh(rq) + 3)
        x = random.random()
        if x <= p1:
            self.CCUsed = True
            self.alpha = self.input_alpha * cc_conf
            self.conf = cc_conf
            return cc_act, "CC"
        elif x <= p2:
            self.CHATUsed = True
            self.alpha = self.input_alpha * chat_conf
            self.conf = chat_conf
            return chat_act, "CHAT"
        else:
            self.alpha = self.input_alpha
            self.conf = 1.0
            return self.lA.get_action(obs, eps), "Q"

    def q_update(self, current_obs, action, reward, new_obs):
        """

        :param current_obs: current observation of the state
        :param action: action taken
        :param reward: reward for the action taken
        :param new_obs: final observation after the action is performed
        """
        self.lA.q_update(self.alpha, current_obs, action, reward, 1, new_obs)

        if self.CCUsed:
            # self.CPCC.q_update(self.alpha, cObs, 0, reward, 1, nObs)  # DRU
            self.CPCC.q_update(self.alpha, current_obs, 0, reward * self.conf, 1, new_obs)  # DCU
        elif self.CHATUsed:
            # self.CPCHAT.q_update( self.alpha, cObs, 0, reward, 1, nObs )  # DRU
            self.CPCHAT.q_update(self.alpha, current_obs, 0, reward * self.conf, 1, new_obs)  # DCU
        else:
            self.CQ.q_update(self.alpha, current_obs, 0, reward, 1, new_obs)
        self.CCUsed, self.CHATUsed = False, False

    def get_conf_values(self):
        v1, v2, v3 = 0, 0, 0
        cnt1, cnt2, cnt3 = 0, 0, 0
        for obs in self.CPCC.Q:
            v1 += self.CPCC.get_q(obs, 0)
            cnt1 += 1
        for obs in self.CPCHAT.Q:
            v2 += self.CPCHAT.get_q(obs, 0)
            cnt2 += 1
        for obs in self.CQ.Q:
            v3 += self.CQ.get_q(obs, 0)
            cnt3 += 1
        av1, av2, av3 = v1 / cnt1 if cnt1 > 0 else 0, v2 / cnt2 if cnt2 > 0 else 0, v3 / cnt3 if cnt1 > 0 else 0
        return av1, av2, av3


def learn(env, num_features_agent1, num_features_agent2, num_actions_agent1, num_actions_agent2,
          cc_agent1, cc_agent2, cc_agent1_weights, cc_agent2_weights, chat_agent1, chat_agent2,
          alpha=1e-3, gamma=0.98, eps=0.05, min_psi=0.05, max_episodes=100000, confidence_threshold=0.0,
          distance_threshold=10.0, horizontal_moving_obstacles=0, vertical_moving_obstacles=0):
    """Learn function for tabular DRoP-Learning with CHAT and Co-ordination confidence action selectors

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
    agent1, agent2 = DRoPLearningAgent(num_features_agent1, num_actions_agent1, alpha, gamma), \
                     DRoPLearningAgent(num_features_agent2, num_actions_agent2, alpha, gamma)

    print(cc_agent1, chat_agent1, chat_agent2)
    episode_rewards, psi = [0.0], 1.0
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    ep_reward = 'episode_rewards_drop_' + str(rank) + '.raw'
    conf_values_a1 = 'confvalues_drop_a1_' + str(rank) + '.raw'
    conf_values_a2 = 'confvalues_drop_a2_' + str(rank) + '.raw'
    action_choice = 'action_choice_drop_' + str(rank) + '.raw'
    action_choice = open(action_choice, 'a+')
    ep_reward = open(ep_reward, 'a+')
    conf_values_a1 = open(conf_values_a1, 'a+')
    conf_values_a2 = open(conf_values_a2, 'a+')
    conf_output = int(max_episodes / 50)

    if env.unwrapped.env_name() == "GuideDog-v1":
        action_space_agent1 = env.unwrapped.action_values(env.unwrapped.actions_agent1)
        action_space_agent2 = env.unwrapped.action_values(env.unwrapped.actions_agent2)
        hat1 = HATActionSelector(action_space_agent1, num_features_agent1, chat_agent1)
        hat2 = HATActionSelector(action_space_agent2, num_features_agent2, chat_agent2)
        cc1 = CCActionSelector(action_space_agent1, num_features_agent1, cc_agent1_weights, cc_agent1, chat_agent1)
        cc2 = CCActionSelector(action_space_agent2, num_features_agent2, cc_agent2_weights, cc_agent2, chat_agent2)
    else:
        action_space = env.unwrapped.action_values(env.unwrapped.actions)
        hat1 = HATActionSelector(action_space, num_features_agent1, chat_agent1)
        hat2 = HATActionSelector(action_space, num_features_agent2, chat_agent2)
        cc1 = CCActionSelector(action_space, num_features_agent1, cc_agent1_weights, cc_agent1, chat_agent1)
        cc2 = CCActionSelector(action_space, num_features_agent2, cc_agent2_weights, cc_agent2, chat_agent2)

    for i in range(max_episodes):
        cc_action_choice, chat_action_choice, chat_equals_cc, chat_not_equals_cc, q_choice = 0, 0, 0, 0, 0
        steps = 0
        if env.unwrapped.env_name() in ("GuideDog-v1"):
            env.unwrapped.get_moving_objects(horizontal_moving_obstacles, vertical_moving_obstacles)
        obs, done = env.reset(), False
        while not done:
            # env.render()
            steps += 1

            # Action selection
            if env.unwrapped.env_name() in ("GuideDog-v0", "GuideDog-v1"):
                if obs[0][3] != 0 or obs[0][4] != 0:
                    cc_agent1_weights = (1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0)
                else:
                    cc_agent1_weights = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0)
            cc_action1 = cc1.select_action(obs[0], cc_agent1_weights, confidence_threshold, distance_threshold)
            cc_action2 = cc2.select_action(obs[1], cc_agent2_weights, confidence_threshold, distance_threshold)
            hat_action1 = hat1.select_action(obs[0])
            hat_action2 = hat2.select_action(obs[1])

            # cc_action1 = (4, 0) if cc_action1[0] == -1 else cc_action1
            # cc_action2 = (4, 0) if cc_action2[0] == -1 else cc_action2

            action1 = agent1.get_action_sd(tuple(obs[0]), eps, hat_action1[0], cc_action1[0], hat_action1[1],
                                           cc_action1[1])
            action2 = agent2.get_action_sd(tuple(obs[1]), eps, hat_action2[0], cc_action2[0], hat_action2[1],
                                           cc_action2[1])

            if action1[1] == 'CC':
                cc_action_choice += 1
                if cc_action1[0] == hat_action1:
                    chat_equals_cc += 1
                else:
                    chat_not_equals_cc += 1
            elif action1[1] == 'CHAT':
                chat_action_choice += 1
            elif action1[1] == 'Q':
                q_choice += 1

            if action2[1] == 'CC':
                cc_action_choice += 1
                if cc_action2[0] == hat_action2:
                    chat_equals_cc += 1
                else:
                    chat_not_equals_cc += 1
            elif action2[1] == 'CHAT':
                chat_action_choice += 1
            elif action2[1] == 'Q':
                q_choice += 1

            env_action = (action1[0], action2[0])
            new_obs, rew, done, info = env.step(env_action)
            # print('actions: ', cc_action1, cc_action2, hat_action1, hat_action2, action1, action2)
            if not done:
                agent1.q_update(tuple(obs[0]), action1[0], rew, tuple(new_obs[0]))
                agent2.q_update(tuple(obs[1]), action2[0], rew, tuple(new_obs[1]))
            episode_rewards[-1] += rew
            if done:
                if not ((i + 1) % conf_output):
                    print(agent1.get_conf_values(), file=conf_values_a1)
                    print(agent2.get_conf_values(), file=conf_values_a2)
                agent1.q_update(tuple(obs[0]), action1[0], rew, tuple([]))
                agent2.q_update(tuple(obs[1]), action2[0], rew, tuple([]))
                print(i, episode_rewards[-1], steps, file=ep_reward)
                print(i, cc_action_choice, chat_action_choice, q_choice, chat_equals_cc, chat_not_equals_cc,
                      file=action_choice)
                # print(i, cc_action_choice, chat_action_choice, q_choice, chat_equals_cc, chat_not_equals_cc)
                break
            obs = new_obs
        episode_rewards.append(0.0)
        psi = psi * math.exp(math.log(min_psi) / max_episodes)
    ep_reward.close()
    conf_values_a1.close()
    conf_values_a2.close()
    return agent1, agent2
