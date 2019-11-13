import gym
from ddpg_agent import *
from matplotlib import pyplot as plt
import pickle

ENV_NAME = 'Pendulum-v0' #'InvertedPendulum-v1'
MAX_EPISODES = 100
MAX_ITERATIONS = 200


def save_rewards(rewards):
	with open("reward_data.pickle", "wb") as handle:
		pickle.dump(rewards, handle)

# with tf.Session() as sess:
env = gym.make(ENV_NAME)
# env = gym.make('Pendulum-v0')
# print(env.observation_space.shape)
# print(env.action_space.shape)
n_states = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]

agent = DDPGAgent(n_states, n_actions)

rewards = []

for episode in range(MAX_EPISODES):
	state = env.reset()
	total_reward = 0
	for itr in range(MAX_ITERATIONS):
		action = agent.getNoisyAction(state)
		state_, reward, done, _ = env.step(action[0])
		agent.observe(state,action,reward,state_,done)
		state = state_
		total_reward += reward
		if done:
			break
	rewards.append(total_reward)
	print(episode, total_reward)

	if episode >= 30 and episode%10 == 0:
		state = env.reset()
		t_reward = 0
		for itr in range(MAX_ITERATIONS):
			action = agent.getAction(state)
			state_, reward, done, _ = env.step(action[0])
			state = state_
			t_reward += 1
			if done:
				print("---------- Agent Succeeded ----------")
				break


# Save Model
agent.save()

save_rewards(rewards)

# Plot rewards
plt.plot(rewards)
# rolling_mean = rewards.rolling(window=10).mean()
# plt.plot(rolling_mean, color='orange')
plt.show()

# Read saved rewards
with open("reward_data.pickle","rb") as handle:
	rewards = pickle.load(handle)
