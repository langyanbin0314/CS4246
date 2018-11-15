import gym
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy, GreedyQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy
from OneOverT import TimeAnnealedPolicy

env = gym.make('Acrobot-v1')
env = env.unwrapped
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

memory = SequentialMemory(limit=3000, window_length=1)

# # greedy
# model = Sequential()
# model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
# model.add(Dense(20))
# model.add(Activation('relu'))
# model.add(Dense(20))
# model.add(Activation('relu'))
# model.add(Dense(20))
# model.add(Activation('relu'))
# model.add(Dense(nb_actions))
# model.add(Activation('linear'))
# print(model.summary())
#
# policy = GreedyQPolicy()
# greedyAgent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
#                  target_model_update=1e-2, policy=policy)
# greedyAgent.compile(Adam(lr=1e-3), metrics=['mae'])
# hist = greedyAgent.fit(env, nb_max_episode_steps = 10000, visualize=False, verbose=2, nb_steps = 1000000)
#
# reward_his_greedy = hist.history.get('episode_reward')
#
#
# # e-greedy
# # Select a policy. We use eps-greedy action selection, which means that a random action is selected
# # with probability eps. We anneal eps from 1.0 to 0.000002 over the course of 500000 steps. This is done so that
# # the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# # (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# # so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
# policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.000001, value_test=.05,
#                               nb_steps=1000000)
# model = Sequential()
# model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
# model.add(Dense(20))
# model.add(Activation('relu'))
# model.add(Dense(20))
# model.add(Activation('relu'))
# model.add(Dense(20))
# model.add(Activation('relu'))
# model.add(Dense(nb_actions))
# model.add(Activation('linear'))
#
#
# egreedyAgent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
#                  target_model_update=1e-2, policy=policy)
# egreedyAgent.compile(Adam(lr=1e-3), metrics=['mae'])
# hist = egreedyAgent.fit(env, nb_max_episode_steps = 10000, visualize=False, verbose=2, nb_steps = 1000000)
#
# reward_his_egreedy = hist.history.get('episode_reward')

#boltzmann
# policy = BoltzmannQPolicy()
#
# model = Sequential()
# model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
# model.add(Dense(20))
# model.add(Activation('relu'))
# model.add(Dense(20))
# model.add(Activation('relu'))
# model.add(Dense(20))
# model.add(Activation('relu'))
# model.add(Dense(nb_actions))
# model.add(Activation('linear'))
#
#
# BoltzmannAgent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
#                  target_model_update=1e-2, policy=policy)
# BoltzmannAgent.compile(Adam(lr=1e-3), metrics=['mae'])
# hist = BoltzmannAgent.fit(env, nb_max_episode_steps = 10000, visualize=False, verbose=2, nb_steps = 1000000)
#
# reward_his_boltzmann = hist.history.get('episode_reward')


policy = TimeAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', nb_steps=1000000)
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(20))
model.add(Activation('relu'))
model.add(Dense(20))
model.add(Activation('relu'))
model.add(Dense(20))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))


eTimegreedyAgent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
                 target_model_update=1e-2, policy=policy)
eTimegreedyAgent.compile(Adam(lr=1e-3), metrics=['mae'])
hist = eTimegreedyAgent.fit(env, nb_max_episode_steps = 10000, visualize=False, verbose=2, nb_steps = 1000000)

reward_his_eTimegreedy = hist.history.get('episode_reward')


# np.savetxt("e-greedy.txt", reward_his_egreedy, fmt='%1.3e')
# np.savetxt("greedy.txt", reward_his_greedy, fmt='%1.3e')
# np.savetxt("boltzmann.txt", reward_his_boltzmann, fmt='%1.3e')
np.savetxt("e-time-greedy.txt", reward_his_eTimegreedy, fmt='%1.3e')

# plt.plot(np.arange(len(reward_his_egreedy)), reward_his_egreedy, 'r', label = 'e-greedy')
# plt.plot(np.arange(len(reward_his_greedy)), reward_his_greedy, 'b', label = 'greedy')
# plt.xlabel('episode')
# plt.ylabel('reward')
# plt.legend()
# plt.show()
