from os.path import dirname, abspath

import gym

from two_agents import tabular_DRoP


def main():
    env = gym.make("GuideDog-v1")
    file_path = dirname(dirname(abspath(__file__)))
    tabular_DRoP.learn(
        env,
        num_features_agent1=11,
        num_features_agent2=3,
        num_actions_agent1=7,
        num_actions_agent2=4,
        cc_agent1=file_path + '/rundata/guidedog/demonstrations/final_arff/moving_obstacles/blind_coord_a1.dat',
        cc_agent2=file_path + '/rundata/guidedog/demonstrations/final_arff/moving_obstacles/blind_coord_a2.dat',
        cc_agent1_weights=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        cc_agent2_weights=(1, 1, 1),
        chat_agent1=file_path + '/rundata/guidedog/demonstrations/final_arff/moving_obstacles/agent1.arff',
        chat_agent2=file_path + '/rundata/guidedog/demonstrations/final_arff/moving_obstacles/agent2.arff',
        alpha=0.005,
        gamma=0.99,
        eps=0.05,
        min_psi=0.05,
        max_episodes=100000,
        confidence_threshold=0.7,
        distance_threshold=5.0,
        horizontal_moving_obstacles=0,
        vertical_moving_obstacles=1
    )
    print('DRoP learning complete')


if __name__ == '__main__':
    main()
