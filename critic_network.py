import tensorflow as tf

BUFFER_SIZE = 10
BATCH_SIZE = 3
GAMMA = 0.99
LEARNING_RATE = 1e-3
TAU = 1e-3

class CriticNetwork:
	def __init__(self, sess, n_states, n_actions, actor_network, trainable_var_count):
		self.sess = sess
		self.actor_network = actor_network
		self.n_actions = n_actions
		self.n_states = n_states
		self.state = tf.placeholder(tf.float32, [None, self.n_states], name='state')
		self.state_ = tf.placeholder(tf.float32, [None, self.n_states], name='state_')
		self.action = tf.placeholder(tf.float32, [None,self.n_actions], name='action')
		self.action_ = tf.placeholder(tf.float32, [None,self.n_actions], name='action_')
		self.reward = tf.placeholder(tf.float32, [None,1], name='reward')
		self.y_batch = tf.placeholder(tf.float32, [None,1], name='y_batch')
		self.q_batch = tf.placeholder(tf.float32, [None,1], name='q_batch')

		self.critic_output = self.createCriticNetwork()
		self.network_params = tf.trainable_variables()[trainable_var_count:]
		self.critic_target_output = self.createCriticNetworkTarget()
		self.target_network_params = tf.trainable_variables()[ \
            trainable_var_count+len(self.network_params):]
		self.action_gradients, self.optimizer = self.trainNetwork()

		self.saver = tf.train.Saver()

		#self.params = load_params(path) # ???
		self.learning_rate = LEARNING_RATE
		self.gamma = GAMMA
		self.tau = TAU

	def createCriticNetwork(self):
		# self.state = tf.placeholder(tf.float32, [None,self.n_states], name='state')
		# conv1 = tf.layers.conv2d(state, 32, 8, 4, padding='same', activation=tf.nn.relu)
		# conv2 = tf.layers.conv2d(conv1, 64, 4, 2, padding='same', activation=tf.nn.relu)
		# flattened = tf.layers.flatten(conv2)
		critic_nw = tf.layers.dense(self.state, 256, activation=tf.nn.relu)
		critic_nw_actions = tf.layers.dense(self.action, 256)
		print(tf.shape(critic_nw) , " ----------- ", tf.shape(critic_nw_actions))
		critic_nw = tf.concat([critic_nw, critic_nw_actions], axis=1)
		critic_output = tf.layers.dense(critic_nw, 1)
		return critic_output

	def createCriticNetworkTarget(self):
		# self.state = tf.placeholder(tf.float32, [None,self.n_states], name='state')
		# conv1 = tf.layers.conv2d(state, 32, 8, 4, padding='same', activation=tf.nn.relu)
		# conv2 = tf.layers.conv2d(conv1, 64, 4, 2, padding='same', activation=tf.nn.relu)
		# flattened = tf.layers.flatten(conv2)
		critic_nw = tf.layers.dense(self.state_, 256, activation=tf.nn.relu)
		critic_nw_actions = tf.layers.dense(self.action_, 256)
		critic_nw = tf.concat([critic_nw, critic_nw_actions], axis=1)
		critic_target_output = tf.layers.dense(critic_nw, 1)
		return critic_target_output

	def trainNetwork(self):
		q_batch = self.critic_output
		loss = tf.reduce_mean(tf.square( \
			self.y_batch - q_batch)) # axis???
		optimizer = \
			tf.train.AdamOptimizer(0.01).minimize(loss) #params[learning_rate]???
		action_gradients = tf.gradients(self.critic_output,self.action)
		return action_gradients, optimizer

	def update(self, y_batch, states, actions):
		self.sess.run(self.optimizer, \
			feed_dict={self.y_batch: y_batch, self.state: states, self.action: actions})
		self.soft_update_target()

	def soft_update_target(self):
		self.sess.run([self.target_network_params[i].assign(tf.multiply( \
			self.network_params[i], self.tau) + \
            tf.multiply(self.target_network_params[i], 1. - self.tau)) \
                for i in range(len(self.target_network_params))])

	def predict(self, state, action):
		return self.sess.run(self.critic_output, \
			feed_dict={self.state: state, self.action: action})

	def predict_target(self, state_, action_):
		return self.sess.run(self.critic_target_output, \
			feed_dict={self.state_: state_, self.action_: action_})

	def get_action_gradient(self, states, actions):
		return self.sess.run(self.action_gradients, \
			feed_dict={self.state: states, self.action: actions})

	def save(self):
		self.saver.save(self.sess, 'saved_model/trained_model')

