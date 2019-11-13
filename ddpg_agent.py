from actor_network import *
from critic_network import *
from replay_buffer import *
from ou_noise import OUNoise

BATCH_SIZE = 3
BUFFER_SIZE = 10
GAMMA = 0.99
LEARNING_RATE = 1e-3
TAU = 1e-3

class DDPGAgent:
	def __init__(self, n_states, n_actions):
		self.sess = tf.InteractiveSession()
		#self.params = loadparams() # ???
		self.n_states = n_states
		self.n_actions = n_actions
		self.actor_network = ActorNetwork(self.sess, n_states, n_actions)
		self.trainable_var_count = self.actor_network.get_trainable_var_count()
		self.critic_network = CriticNetwork(self.sess, n_states, n_actions, \
			self.actor_network, self.trainable_var_count)
		self.replay_buffer = ReplayBuffer(BUFFER_SIZE) #params['buffer_size']???
		self.exploration_noise = OUNoise(self.n_actions)
		# self.noise = Noise()
		self.gamma = GAMMA
		self.sess.run(tf.global_variables_initializer())

	def getNoisyAction(self, current_state):
		action = self.actor_network.predict( \
			np.reshape(current_state, (1, self.n_states)))
		return action + self.exploration_noise.noise()

	def getAction(self, current_state):
		return self.actor_network.predict( \
			np.reshape(current_state, (1, self.n_states)))

	def observe(self, state, action, reward, state_, done):
		self.replay_buffer.add(state, action, reward, state_, done)
		batch = self.replay_buffer.sampleBatch(BATCH_SIZE) #params['batch_size']???
		# batch = tf.concat([batch, (state,action,reward,state_)]) # axis???
		self.updateActorAndCritic(batch)

	def updateActorAndCritic(self, batch):
		# states, actions, rewards, states_, dones = zip(*batch)
		states = np.asarray([data[0] for data in batch])
		actions = np.asarray([data[1] for data in batch])
		rewards = np.asarray([data[2] for data in batch])
		states_ = np.asarray([data[3] for data in batch])
		dones = np.asarray([data[4] for data in batch])

		current_batch_size = actions.size

		states = np.reshape(states, (current_batch_size, self.n_states))
		actions = np.reshape(actions, (current_batch_size, self.n_actions))
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

		self.critic_network.update(y_batch, states, actions)

		action_batch_for_gradient = self.actor_network.predict(states)
		action_batch_for_gradient = np.reshape( \
			action_batch_for_gradient,(current_batch_size, 1))
		q_gradient_batch = self.critic_network.get_action_gradient(states, action_batch_for_gradient)
		q_gradient_batch = np.reshape( \
			q_gradient_batch,(current_batch_size, 1))
		self.actor_network.update(states, q_gradient_batch)

	def save(self):
		self.critic_network.save()




