import gym
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy, GreedyQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy

env = gym.make('Acrobot-v1')
env = env.unwrapped
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n
memory = SequentialMemory(limit=3000, window_length=1)

#no duel
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
print(model.summary())

policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.000002, value_test=.05,
                              nb_steps=500000)
noDuelAgent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
                 target_model_update=1e-2, policy=policy)
noDuelAgent.compile(Adam(lr=1e-3), metrics=['mae'])
hist = noDuelAgent.fit(env, nb_max_episode_steps = 10000, visualize=False, verbose=2, nb_steps = 500000)

reward_his_noDuel = hist.history.get('episode_reward')


#duel with max
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.000002, value_test=.05,
                              nb_steps=500000)
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


duelMaxAgent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
                 target_model_update=1e-2, policy=policy, enable_dueling_network=True, dueling_type='max')
duelMaxAgent.compile(Adam(lr=1e-3), metrics=['mae'])
hist = duelMaxAgent.fit(env, nb_max_episode_steps = 10000, visualize=False, verbose=2, nb_steps = 500000)

reward_his_duelMax = hist.history.get('episode_reward')

#duel with avg
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.000002, value_test=.05,
                              nb_steps=500000)
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


duelAvgAgent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
                 target_model_update=1e-2, policy=policy, enable_dueling_network=True, dueling_type='avg')
duelAvgAgent.compile(Adam(lr=1e-3), metrics=['mae'])
hist = duelAvgAgent.fit(env, nb_max_episode_steps = 10000, visualize=False, verbose=2, nb_steps = 500000)

reward_his_duelAvg = hist.history.get('episode_reward')



np.savetxt("noDuel.txt", reward_his_noDuel, fmt='%1.3e')
np.savetxt("duelMax.txt", reward_his_duelMax, fmt='%1.3e')
np.savetxt("duelAvg.txt", reward_his_duelAvg, fmt='%1.3e')

plt.plot(np.arange(len(reward_his_noDuel)), reward_his_noDuel, 'b', label = 'no duel')
plt.plot(np.arange(len(reward_his_duelMax)), reward_his_duelMax, 'r', label = 'duel max')
plt.plot(np.arange(len(reward_his_duelAvg)), reward_his_duelAvg, 'g', label = 'duel avg')
plt.xlabel('episode')
plt.ylabel('reward')
plt.legend()
plt.show()
