import tensorflow as tf
from collections import deque
import numpy as np
import random

from actor_network import ActorNetwork
from critic_network import CriticNetwork
from replay_buffer import ReplayBuffer
from ou_noise import OUNoise

BATCH_SIZE = 64
BUFFER_SIZE = 50000
GAMMA = 0.99
LEARNING_RATE = 1e-3
TAU = 1e-3

class DDPGAgent:
	def __init__(self, env):
		self.sess = tf.InteractiveSession()
		#self.params = loadparams() # ???
		self.env = env
		self.n_states = env.observation_space.shape[0]
		self.n_actions = env.action_space.shape[0]
		self.low = self.env.action_space.low
		self.high = self.env.action_space.high
		self.actor_network = ActorNetwork(self.sess, self.n_states, self.n_actions)
		self.trainable_var_count = self.actor_network.get_trainable_var_count()
		self.critic_network = CriticNetwork(self.sess, self.n_states, self.n_actions, \
			self.actor_network, self.trainable_var_count)
		self.replay_buffer = ReplayBuffer(BUFFER_SIZE) #params['buffer_size']???
		self.exploration_noise = OUNoise(self.n_actions)
		# self.noise = Noise()
		self.gamma = GAMMA
		self.sess.run(tf.global_variables_initializer())

	def getNoisyAction(self, current_state):
		current_state = np.reshape(current_state, (1, self.n_states))
		# print ("current_state =", np.shape(current_state))
		action = self.actor_network.predict(current_state)
		return np.clip(action + self.exploration_noise.noise(), self.low, self.high)

	def getAction(self, current_state):
		return self.actor_network.predict( \
			np.reshape(current_state, (1, self.n_states)))

	def observe(self, state, action, reward, state_, done):
		self.replay_buffer.add(state, action[0], reward, state_, done)
		# batch = tf.concat([batch, (state,action,reward,state_)]) # axis???
		if (self.replay_buffer.count > 500):
			batch = self.replay_buffer.sampleBatch(BATCH_SIZE)
			self.updateActorAndCritic(batch)
		if done:
			self.exploration_noise.reset()

	def updateActorAndCritic(self, batch):
		# states, actions, rewards, states_, dones = zip(*batch)
		states = np.asarray([data[0] for data in batch])
		actions = np.asarray([data[1] for data in batch])
		rewards = np.asarray([data[2] for data in batch])
		states_ = np.asarray([data[3] for data in batch])
		dones = np.asarray([data[4] for data in batch])

		current_batch_size = BATCH_SIZE

		states = np.reshape(states, (current_batch_size, self.n_states))
		# print("actions shape----------", np.shape(actions))
		# actions = np.reshape(actions, (current_batch_size, self.n_actions))
		states_ = np.reshape(states_, (current_batch_size, self.n_states))

		actions_ = self.actor_network.predict_target(states_)

		y_batch = []
		q_batch = []
		yi =[]
		for i in range(current_batch_size):
			if dones[i]:
				yi = rewards[i]
			else:
				yi = rewards[i] + \
					self.gamma * self.critic_network.predict_target( \
						np.reshape(states_[i], (1, self.n_states)), \
						np.reshape(actions[i],(1, self.n_actions)))
			y_batch.append(yi)

		y_batch = np.reshape(y_batch,(current_batch_size,1))

		# print("critic update begins")
		self.critic_network.update(y_batch, states, actions)
		# print("critic update ends")

		# print("action batch begins")
		action_batch_for_gradient = self.actor_network.predict(states)
		# print("action batch ends")
		# action_batch_for_gradient = np.reshape( \
		# 	action_batch_for_gradient,(current_batch_size, 1))
		# print("q batch gradient begins")
		q_gradient_batch = self.critic_network.get_action_gradient(states, action_batch_for_gradient)
		# print("q batch gradient done")
		# q_gradient_batch = np.reshape( \
		# 	q_gradient_batch,(current_batch_size,1))
		# print("actor update begins")
		self.actor_network.update(states, q_gradient_batch)
		# print("actor update ends")

	def save(self):
		self.critic_network.save()




