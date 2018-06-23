import argparse
import tensorflow as tf
import time
import gym
import atari_wrappers
import dqn
import util


################## ARGUMENTS PARSING ###################
parser = argparse.ArgumentParser(description='Beating OpenAI envs with Reinforcement Learning')

parser.add_argument('--environment', dest='env', type=str, default='Alien-ram-v0', help='environment to be used in the simulation', required=True)
parser.add_argument('--model', dest='model', type=str, help='path of the saved network model', required=True)
parser.add_argument('--num_games', dest='num_games', type=int, default=10, help='number of games to be played')
parser.add_argument('--steps_sleep', dest='steps_sleep', type=int, default=0, help='sleep time between steps')
parser.add_argument('--network', dest='network', type=str, default='dqn', choices=['dqn', 'doubledqn', 'duelingdqn'], help='Type of network used')
args = parser.parse_args()

ENVIRONMENT = args.env
MODEL = args.model
NUM_GAMES = args.num_games
NETWORK = args.network
STEPS_SLEEP = args.steps_sleep


########## LOAD ENVIRONMENT AND BUILD NETWORK ##########
env = gym.make(ENVIRONMENT)
env = atari_wrappers.MaxAndSkipEnv(env, 3)
env = atari_wrappers.wrap_deepmind(env, frame_stack=True, clip_rewards=False)


#optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
use_target_network = False if NETWORK.startswith('double') else True
use_double_dqn = True if NETWORK.startswith('double') else False
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.00025, momentum=0.95, epsilon=0.01)

net = dqn.get_network(type=NETWORK, input_shape=env.observation_space.shape, num_actions=env.action_space.n,
						use_target_network=use_target_network, use_double_dqn=use_double_dqn, optimizer=optimizer)

#################### TESTING AGENT #####################
saver = tf.train.Saver()
stats = util.GameStats()

with tf.Session() as sess:

	print("Testing model\n")
	saver.restore(sess, MODEL)
	
	for game in range(NUM_GAMES):
		endgame = False
		observation = env.reset()
		treward = 0
		
		while not endgame:
			env.render()
			action = net.takeAction(sess, observation)
			observation, reward, endgame, info = env.step(action)
			treward += reward
			stats.addReward(reward, endgame)
			#time.sleep(STEPS_SLEEP)
			
		print('Game', game+1, 'ended. Total Reward:', treward)

training_rewards = './test/test_rewards.pkl'
util.PickleSerializer.save(stats, training_rewards)

util.plotGameStats([stats], "test_duelingdqn_1M", episodes_span=25, labels=["dueling dqn"])