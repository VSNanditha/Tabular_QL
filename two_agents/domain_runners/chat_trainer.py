from os.path import dirname, abspath

import gym

from two_agents import tabular_ql


def main():
    env = gym.make("GuideDog-v1")
    file_path = dirname(dirname(abspath(__file__)))
    tabular_ql.chat_learn(
        env,
        num_features_agent1=10,
        num_features_agent2=2,
        num_actions_agent1=4,
        num_actions_agent2=4,
        chat_agent1=file_path + '/rundata/guidedog/demonstrations/noise_0.1/no_moving_obstacles/agent1.arff',
        chat_agent2=file_path + '/rundata/guidedog/demonstrations/noise_0.1/no_moving_obstacles/agent2.arff',
        alpha=1e-2,
        gamma=0.99,
        eps=0.05,
        min_psi=0.05,
        max_episodes=100000,
        confidence_threshold=0.7
    )
    print('Chat learning complete')


if __name__ == '__main__':
    main()
