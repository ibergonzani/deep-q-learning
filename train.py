import tensorflow as tf
import argparse
import random
import gym
import util
import atari_wrappers
import numpy as np
import os
import dqn
import cv2


# Class storing the training information (class -> easy to save on file)
class TrainingInfo():
	
	def __init__(self, experience_replay_size=100000):
		
		# buffer containing past transitions used for training
		self.experience_replay = util.ExperienceReplay(capacity=experience_replay_size)
		
		# game statistics: #ìnumber of episodes, rewards...
		self.game_stats = util.GameStats()
		
		self.batch_size = 32				# number of transitions used for each model update
		
		self.train_steps = 2000000			# total number of frames in the training
		self.pre_train_steps = 50000		# number of initial random steps (actions)
		self.target_network_steps = 10000	# number of frames between each update of the target network
		
		self.current_step = 0				# current step in training (Here just used for checkpoint saves) 
		
		self.epsilon_start = 1.0			# initial value for epsilon
		self.epsilon_end = 0.1				# final value for epsilon
		self.epsilon_start_frame = 0		# starting frame for epsilon linear decrease
		self.epsilon_end_frame = 1000000	# endign frame for epsilon linear decrease
		
	
	# just increase the total number of steps(frames) of the training
	# used for example if training resume from the end of a previous training
	def addSteps(self, steps):
		self.train_steps += steps
	
	# return the correct epsilon value fot the given step(frame)
	# considering starting and ending frame for the linear decrease of epsilon
	def getEpsilon(self, step):
		bounded_step = min(max(step, self.epsilon_start_frame), self.epsilon_end_frame)
		movement = (bounded_step - self.epsilon_start_frame) / (self.epsilon_end_frame - self.epsilon_start_frame)
		return self.epsilon_start + movement * (self.epsilon_end - self.epsilon_start)


		

if __name__ == "__main__":

	################## ARGUMENTS PARSING ###################
	parser = argparse.ArgumentParser(description='Beating OpenAI envs with Reinforcement Learning: Training script')

	parser.add_argument('--environment', dest='env', type=str, default='Alien-ram-v0', help='environment to be used in the simulation')
	parser.add_argument('--model_folder', dest='model_folder', type=str, default='./model/', help='path where models are saved')
	parser.add_argument('--model_name', dest='model_name', type=str, default='model.ckpt', help='Name of the model to be saved')
	parser.add_argument('--checkpoint', dest='checkpoint', default=None, type=str, help='path of the model\'s checkpoint')
	parser.add_argument('--training_info', dest='training_info', default=None, type=str, help='path of saved training info (pickle)')
	parser.add_argument('--network', dest='network', type=str, default='dqn', choices=['dqn', 'doubledqn', 'duelingdqn', 'doubleduelingdqn'], help='Type of network to be used')
	parser.add_argument('--add_training_steps', dest='add_training_steps', type=int, default=None, help='Number of steps between each checkpoint')
	parser.add_argument('--checkpoint_steps', dest='checkpoint_steps', type=int, default=100000, help='Number of steps between each checkpoint')

	parser.set_defaults(render_training=False)
	args = parser.parse_args()

	ENVIRONMENT = args.env
	MODEL_FOLDER = args.model_folder
	MODEL_NAME = args.model_name
	CHECKPOINT = args.checkpoint
	TRAINING_INFO = args.training_info
	NETWORK = args.network
	CHECKPOINT_STEPS = args.checkpoint_steps
	additional_training_steps = args.add_training_steps


	########## LOAD ENVIRONMENT AND BUILD NETWORK ##########
	env = gym.make(ENVIRONMENT)
	env = atari_wrappers.wrap_deepmind(env, frame_stack=True, clip_rewards=True)
	env = atari_wrappers.MaxAndSkipEnv(env, skip=3)
	#env = atari_wrappers.CenteredScaledFloatFrame(env)
	
	optimizer = tf.train.RMSPropOptimizer(learning_rate=0.00025, momentum=0.95, epsilon=0.01)
	
	#optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
	use_target_network = True #False if NETWORK.startswith('double') else True
	use_double_dqn = True if NETWORK.startswith('double') else False
	
	net = dqn.get_network(type=NETWORK, input_shape=env.observation_space.shape, num_actions=env.action_space.n,
									use_target_network=use_target_network, use_double_dqn=use_double_dqn, optimizer=optimizer)

	
	#################### TRAINING AGENT ####################
	saver = tf.train.Saver(max_to_keep=10)

	with tf.Session() as sess:
		
		tf.global_variables_initializer().run()
		
		# Restoring network weight and experience replay buffer from previous sessions
		if CHECKPOINT != None:
			print("Restoring model from checkpoint")
			saver.restore(sess, CHECKPOINT)
		
		# training informations
		trn = TrainingInfo()
		
		#restoring training informations from previus sessions
		if TRAINING_INFO != None:
			print("Restoring training informations: experience replay buffer, games statistics")
			util.PickleSerializer.load(trn, TRAINING_INFO)
			trn.game_stats.deleteCurrentGame()			
		
		if additional_training_steps != None:
			trn.addSteps(additional_training_steps)
		
		endgame = True
		
		# Training procedure
		for step in range(trn.current_step, trn.train_steps):
			print("Training step %d/%d\t%f" % (step+1, trn.train_steps, trn.getEpsilon(step)), end="\r")
			
			if endgame:
				print("Game %d completed. Reward: %d" % (trn.game_stats.totalGames(), trn.game_stats.lastGameReward()))
				state = env.reset()
			
			if np.random.uniform() < trn.getEpsilon(step):
				action = env.action_space.sample()
			else:
				action = net.takeAction(sess, state)
				
			new_state, reward, endgame, info = env.step(action)
			trn.experience_replay.insert([state, action, reward, new_state, endgame])
			state = new_state
			
			trn.game_stats.addReward(reward, endgame)
			
			if step >= trn.pre_train_steps:
				batch = trn.experience_replay.sample(trn.batch_size)
				net.updateModel(sess, batch)
			
			if step % trn.target_network_steps == 0:
				net.updateTargetNetwork(sess)
			
			trn.current_step = step + 1 	# in order to resume to next step and not this one
						
			if step % CHECKPOINT_STEPS == 0 and step != 0:
				print("\nSave checkpoint:", step, "steps")
				checkpoint_path = MODEL_FOLDER + MODEL_NAME			
				saver.save(sess, checkpoint_path, global_step=step)
				
				experience_buffer_path = MODEL_FOLDER + 'training_info_' + str(step) + '.pkl'
				util.PickleSerializer.save(trn, experience_buffer_path)
		
		# Saving final network and experience replay buffer
		model_path = MODEL_FOLDER + MODEL_NAME
		saver.save(sess, model_path)
		
		experience_buffer_path = MODEL_FOLDER + 'training_info.pkl'
		util.PickleSerializer.save(trn, experience_buffer_path)
		training_rewards = MODEL_FOLDER + 'training_rewards.pkl'
		util.PickleSerializer.save(trn.game_stats, training_rewards)
		
		util.plotGameStats([trn.game_stats], NETWORK, episodes_span=25, labels=[NETWORK])
