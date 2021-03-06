import gym

from two_agents import tabular_ql


def main():
    env = gym.make("GuideDog-v1")
    tabular_ql.human_demonstration(
        env,
        num_features_agent1=11,
        num_features_agent2=3,
        num_actions_agent1=7,
        num_actions_agent2=4,
        alpha=0.005,
        gamma=0.99,
        max_episodes=10,
        horizontal_moving_obstacles=0,
        vertical_moving_obstacles=1
    )


if __name__ == '__main__':
    main()
