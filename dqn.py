import tensorflow as tf
import numpy as np



class DQN():

	def __init__(self, input_shape, num_actions, optimizer, use_target_network=True, use_double_dqn=False, gamma=0.99):
	
		input_shape = (None,) + input_shape
		output_shape = (None, num_actions)
		
		self.optimizer = optimizer
		self.use_target_network = use_target_network
		self.use_double_dqn = use_double_dqn
		self.gamma = 0.99
		
		self.qvars = None
		self.tvars = None
		
		self.input = tf.placeholder(tf.float32, shape=input_shape) 
		normalized_input = self.input / 127.5 - 1.0 # faster to do here on gpu instead on cpu during observation
		
		# q network graph definition
		with tf.variable_scope("qnet"):
			self.output = self._network(normalized_input, num_actions)
			current_scope = tf.get_default_graph().get_name_scope()
			self.qvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=current_scope)
			self.qvars.sort(key=lambda x: x.name)
		
		if self.use_target_network or self.use_double_dqn:
			# target network graph definition
			with tf.variable_scope("tnet"):
				self.target = self._network(normalized_input, num_actions)
				current_scope = tf.get_default_graph().get_name_scope()
				self.tvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=current_scope)
				self.tvars.sort(key=lambda x: x.name)
			
			# target network weights update operations
			self.update_target_op = [var[0].assign(var[1]) for var in zip(self.tvars, self.qvars)]
		
		# training operations definition
		self.yt_loss = tf.placeholder(tf.float32, shape=(None))
		self.actions = tf.placeholder(tf.int32, shape=(None))
		actions_onehot = tf.one_hot(self.actions, num_actions)
		q_actions = tf.multiply(actions_onehot, self.output)
		
		self.loss = tf.losses.huber_loss(self.yt_loss, tf.reduce_sum(q_actions, axis=1))
		self.train_op = self.optimizer.minimize(loss=self.loss)
		
		
	
	def _network(self, input, num_actions):
		conv1 = tf.layers.conv2d(inputs=input, filters=32, kernel_size=(8,8), strides=(4,4), activation=tf.nn.relu)
		conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=(4,4), strides=(2,2), activation=tf.nn.relu)
		conv3 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=(3,3), strides=(1,1), activation=tf.nn.relu)
		flat = tf.contrib.layers.flatten(conv3)
		fc1 = tf.layers.dense(inputs=flat, units=512, activation=tf.nn.relu)
		output = tf.layers.dense(inputs=fc1, units=num_actions, activation=None)
		return output
	
	
	# update target network weights 
	def updateTargetNetwork(self, sess):
		if self.use_target_network or self.use_double_dqn:
			sess.run(self.update_target_op)
			
	
	def updateModel(self, sess, batch):
		states, actions, rewards, new_states, endgames = batch
		
		qtarget = None
		
		if self.use_double_dqn:
			# computing target Q value using target network and double deep q-network algorithm
			[n_out, t_out] = sess.run([self.output, self.target], feed_dict={self.input: np.array(new_states)})
			target_action = np.argmax(n_out, axis=1)
			qtarget = np.array([output_sample[target_action[sample]] for sample, output_sample in enumerate(t_out)])
		elif self.use_target_network:
			# computing target Q value using target network
			qtarget = np.amax( sess.run(self.target, feed_dict={self.input: np.array(new_states)}), axis=1)
		else:
			qtarget = np.amax( sess.run(self.output, feed_dict={self.input: np.array(new_states)}), axis=1)
			
		yt = rewards + self.gamma * (np.logical_not(endgames) * qtarget)
		
		# computing loss and update weights of  Q network
		sess.run([self.loss, self.train_op], feed_dict={self.input: np.array(states), self.yt_loss: yt, self.actions: np.array(actions)})
	
	
	def predict(self, sess, X):
		return sess.run(self.output, feed_dict={self.input: X})

	def takeAction(self, sess, X):
		return np.argmax(self.predict(sess, np.array([X]))[0])

		
		
		
class DuelingDQN(DQN):
	
	# just redefine the network architecture
	def _network(self, input, num_actions):
		conv1 = tf.layers.conv2d(inputs=input, filters=32, kernel_size=(8,8), strides=(4,4), activation=tf.nn.relu)
		conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=(4,4), strides=(2,2), activation=tf.nn.relu)
		conv3 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=(3,3), strides=(1,1), activation=tf.nn.relu)
		conv3 = tf.contrib.layers.flatten(conv3)
		# advantage stream
		fc1a = tf.layers.dense(inputs=conv3, units=512, activation=tf.nn.relu)
		fc1v = tf.layers.dense(inputs=conv3, units=512, activation=tf.nn.relu)
		# value stream
		advantage = tf.layers.dense(inputs=fc1a, units=num_actions, activation=tf.nn.relu)
		value = tf.layers.dense(inputs=fc1v, units=1, activation=tf.nn.relu)
		output = tf.reshape(value, [-1, 1]) + (advantage - tf.reduce_mean(advantage, axis=1, keep_dims=True))
		
		return output
		
		

		
def get_network(type, **kargs):
	
	net = None
	if type == 'dqn' or type == 'doubledqn':
		net = DQN(**kargs)
	elif type == 'duelingdqn' in type:	
		net = DuelingDQN(**kargs)
	
	return net


	
