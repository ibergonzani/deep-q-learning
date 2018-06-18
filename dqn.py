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
			self.update_target_op = [vars[0].assign(vars[1]) for vars in zip(self.tvars, self.qvars)]
		
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
			print("Updating target network")
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
	def _network(self, input_shape, num_actions):
		
		input = tf.placeholder(tf.float32, shape=input_shape)
		conv1 = tf.layers.conv2d(inputs=input, filters=16, kernel_size=[8,8], strides=[4,4], activation=tf.nn.relu)
		conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[4,4], strides=[2,2], activation=tf.nn.relu)
		fc1 = tf.layers.dense(inputs=conv2, units=50, activation=tf.nn.relu)
		fc2a = tf.layers.dense(inputs=fc1, units=25, activation=tf.nn.relu)
		fc2b = tf.layers.dense(inputs=fc1, units=25, activation=tf.nn.relu)
		fc3a = tf.layers.dense(inputs=fc2a, units=1, activation=tf.nn.relu)
		fc3b = tf.layers.dense(inputs=fc2b, units=num_actions, activation=tf.nn.relu)
		output = tf.scalar_sum(fc3a, fc3b)
		
		return output
		
		

		
# class DoubleNetwork():
	
	# def __init__(self, networkclass, **kargs):
		
		# with tf.variable_scope("dnet1"):
			# self.network1 = networkclass(**kargs)
		# with tf.variable_scope("dnet1"):
			# self.network2 = networkclass(**kargs)
	
	
	# def updateModel(self, sess, batch):
		# states, actions, rewards, new_states, endgames = batch
		
		# qtarget = None
		# if self.use_target_network:
			# # computing target Q value using target network
			# qtarget = sess.run(self.target, feed_dict={self.input: np.array(new_states)})
		# else:
			# qtarget = sess.run(self.output, feed_dict={self.input: np.array(new_states)})
		# y = rewards + self.gamma * (np.logical_not(endgames) * np.amax(qtarget, axis=1))
		# # computing loss and update weights of  Q network
		# sess.run([self.loss, self.train_op], feed_dict={self.input: np.array(states), self.ground_truth: y, self.actions: np.array(actions), self.learning_rate: learning_rate})
	
	
	# def predict(self, sess, X):
		# return self.network1.predict(sess, X)

	# def takeAction(self, sess, X):
		# return self.network1.takeAction(sess, X)
		
	# # def __init__(self, input_shape, num_action):
		# # base_network.__init__(self)
		
		# # input = tf.placeholder(tf.float32, shape=(None,) + input_shape)
		# # conv1 = tf.layers.conv2d(inputs=input_std, filters=16, kernel_size=[8,8], strides=[4,4], activation=tf.nn.relu)
		# # conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[4,4], strides=[2,2], activation=tf.nn.relu)
		
		# # # first set of parameters for the Q function
		# # fc1 = tf.layers.dense(inputs=conv2, units=256, activation=tf.nn.relu)
		# # self.output1 = tf.layers.dense(inputs=fc1, units=8, activation=None)
		
		# # # second set of parameters for the Q function
		# # fc2 = tf.layers.dense(inputs=conv2, units=256, activation=tf.nn.relu)
		# # self.output2= tf.layers.dense(inputs=fc2, units=num_action, activation=None)
		
		# # ground_truth = tf.placeholder(tf.float32, shape=(None, num_action))
		# # loss1 = tf.losses.huber_loss(ground_truth, self.output1)
		# # loss2 = tf.losses.huber_loss(ground_truth, self.output2)
		
		# # learning_rate = tf.placeholder(tf.float32)
		# # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
		# # self.train_op1 = optimizer.minimize(loss=loss1, global_step=tf.train.get_global_step())
		# # self.train_op2 = optimizer.minimize(loss=loss2, global_step=tf.train.get_global_step())

		
		
	# # def updateModel(self, sess, batch, learning_rate=0.01, parameters_set=0):
		# # train_op = [self.train_op1, self.train_op2][parameters_set]
		# # sess.run([train_op], feed_dict={'input':X, 'ground_truth':y, 'learning_rate':learning_rate})
	
	
	# # def predict(self, sess, X, parameters_set=0):
		# # output = [self.output1, self.output2][parameters_set]
		# # return sess.run([output], feed_dict={'x':X})
		
	# # def takeAction(self, sess, X, parameters_set=0):
		# # return np.argmax(self.predict(sess, X, parameters_set))
		

		
def get_network(type, **kargs):
	
	net = None
	if type == 'dqn' or type == 'doubledqn':
		net = DQN(**kargs)
	elif type == 'duelingdqn' in type:	
		net = DuelingDQN(**kargs)
	# elif type == 'doubledqn':
		# net = DoubleNetwork(DQN, **kargs)
	# elif type == 'doubleduelingdqn':
		# net = DoubleNetwork(DuelingDQN, **kargs)
	
	return net

		
# def get_network(type, input_shape, num_actions):
	# net = None
	
	# if type == 'dqn':
		# net = DQN(input_shape=input_shape, num_actions=num_actions)
	# elif type == 'doubledqn':
		# net = DoubleNetwork(DQN, input_shape=input_shape, num_actions=num_actions)
	# elif type == 'duelingdqn':	
		# net = DuelingDQN(input_shape=input_shape, num_actions=num_actions)
	# elif type == 'doubleduelingdqn':
		# net = DoubleNetwork(DuelingDQN, input_shape=input_shape, num_actions=num_actions)
	
	# return net