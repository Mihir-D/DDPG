import tensorflow as tf

BUFFER_SIZE = 10
BATCH_SIZE = 3
GAMMA = 0.99
LEARNING_RATE = 1e-3
TAU = 1e-3

class ActorNetwork:
	def __init__(self, sess, n_states, n_actions):
		self.sess = sess
		self.n_actions = n_actions
		self.n_states = n_states

		self.batch_size = BATCH_SIZE
		self.learning_rate = 0.01
		self.tau = 0.001

		self.input_state = tf.placeholder(tf.float32, [None,self.n_states])
		self.state_ = tf.placeholder(tf.float32, [None,self.n_states])
		self.actor_output = self.createActorNetwork()
		self.network_params = tf.trainable_variables()
		self.actor_target_output = self.createActorNetworkTarget()
		self.target_network_params = tf.trainable_variables()[ \
			len(self.network_params):]
		self.trainable_var_count = len(tf.trainable_variables())
		self.optimize = self.trainNetwork()

	def createActorNetwork(self):
		# conv1 = tf.layers.conv2d(self.input_state, 32, 8, 4, padding='same', activation=tf.nn.relu)
		# conv2 = tf.layers.conv2d(conv1, 64, 4, 2, padding='same', activation=tf.nn.relu)
		# flattened = tf.layers.flatten(conv2)
		actor_nw = tf.layers.dense(self.input_state, 256, activation=tf.nn.relu)
		actor_nw = tf.layers.dense(actor_nw, 128, activation=tf.nn.relu, )
		actor_output = tf.layers.dense(actor_nw, self.n_actions, activation=tf.nn.tanh)
		return actor_output


	def createActorNetworkTarget(self):
		# conv1 = tf.layers.conv2d(self.input_state, 32, 8, 4, padding='same', activation=tf.nn.relu)
		# conv2 = tf.layers.conv2d(conv1, 64, 4, 2, padding='same', activation=tf.nn.relu)
		# flattened = tf.layers.flatten(conv2)
		actor_nw = tf.layers.dense(self.state_, 256, activation=tf.nn.relu)
		actor_nw = tf.layers.dense(actor_nw, 128, activation=tf.nn.relu, )
		actor_target_output = tf.layers.dense(actor_nw, self.n_actions, activation=tf.nn.tanh)
		return actor_target_output

	def trainNetwork(self):
		self.q_gradient_input = tf.placeholder(tf.float32,[None,self.n_actions])
		self.actor_gradients = tf.gradients(self.actor_output, \
			self.network_params, -self.q_gradient_input)
		optimize = tf.train.AdamOptimizer(self.learning_rate). \
			apply_gradients(zip(self.actor_gradients, self.network_params))
		return optimize

	def update(self, state, q_gradient_batch):
		self.sess.run(self.optimize, \
			feed_dict={self.input_state: state, \
				self.q_gradient_input: q_gradient_batch})
	def soft_update_target():
		self.sess.run([self.target_network_params[i].assign(tf.multiply( \
			self.network_params[i], self.tau) + \
            tf.multiply(self.target_network_params[i], 1. - self.tau)) \
                for i in range(len(self.target_network_params))])

	def predict(self, states):
		return self.sess.run(self.actor_output, feed_dict={self.input_state: states})

	def predict_target(self, states_):
		return self.sess.run(self.actor_target_output, feed_dict={self.state_: states_})

	def get_trainable_var_count(self):
		return self.trainable_var_count
