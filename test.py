import gym
import numpy as np
from dqn import DeepQNetwork
import matplotlib.pyplot as plt

env = gym.make('Acrobot-v1')
env = env.unwrapped

# print(env.action_space)
# print(env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)

RL = DeepQNetwork(n_actions=env.action_space.n, n_features=env.observation_space.shape[0], learning_rate=0.001,
                  e_greedy=0.9,
                  replace_target_iter=300, memory_size=3000, e_greedy_increment=0.005, output_graph=True)

rewards_his = [];

def run_100_episodes():
    total_steps = 0
    best = -500
    for i_episode in range(10):

        observation = env.reset()
        ep_r = 0
        while True:
            env.render()

            action = RL.choose_action(observation)

            observation_, reward, done, info = env.step(action)

            # print(action, reward)

            costheta1, sintheta1, costheta2, sintheta2, thetaDot, thetaDot2 = observation_

            # height of c.g of lower bar 越大 reward 越大
            reward_to_store = (-costheta1 + sintheta1 * sintheta2 - costheta1 * costheta2)

            # print(reward_to_store)

            RL.store_transition(observation, action, reward_to_store, observation_)

            if total_steps > 200:
                RL.learn()

            ep_r += reward
            if done:
                # get = '| Get' if -costheta1 + sintheta1 * sintheta2 - costheta1 * costheta2 >= 1.0 else '| ----'
                # print('Epi: ', i_episode,
                #       get,
                #       '| Ep_r: ', round(ep_r, 4),
                #       '| Epsilon: ', round(RL.epsilon, 2))
                best = max(ep_r, best)
                rewards_his.append(ep_r)
                break

            observation = observation_
            total_steps += 1

    return best

run_100_episodes()

plt.plot(np.arange(len(rewards_his)), rewards_his)
plt.ylabel('reward')
plt.xlabel('episode')
plt.show()

# RL.plot_cost()

# bests = []
#
# for i in range(100):
#     print("running test ", i)
#     best = run_100_episodes()
#     bests.append(best)
#
# print(np.average(bests), np.std(bests))
