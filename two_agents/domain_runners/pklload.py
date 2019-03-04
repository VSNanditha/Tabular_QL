import gym
from mpi4py import MPI

from two_agents.tabular_ql import learn_from_pickle


def main():
    env = gym.make("GuideDog-v1")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    for i in range(1):
        print('pickle load', i)
        try:
            pickle_file0 = "cc_model9_0.pkl"
            pickle_file1 = "cc_model9_1.pkl"
            learn_from_pickle(env,
                              pickle_file0,
                              pickle_file1,
                              num_features_agent1=11,
                              num_features_agent2=3,
                              num_actions_agent1=8,
                              num_actions_agent2=5,
                              max_episodes=100,
                              horizontal_moving_obstacles=0,
                              vertical_moving_obstacles=1
                              )
        except FileNotFoundError:
            continue


if __name__ == '__main__':
    main()
