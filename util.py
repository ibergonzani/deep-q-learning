import numpy as np
import matplotlib.pyplot as plt
import random
import json
import pickle
import os


class PickleSerializer():
	
	@staticmethod
	def save(obj, path):
		with open(path, 'wb') as outfile:
			pickle.dump(obj.__dict__, outfile, pickle.HIGHEST_PROTOCOL)
			
	@staticmethod
	def load(obj, path):
		with open(path, 'rb') as infile:
			data = pickle.load(infile)
		for key, value in data.items():
			obj.__dict__[key] = value
				


# Buffer with a defined capacity that can be preallocated using a default value or .
# If the buffer is saturated, adding an element will delete a stored value following a FIFO arrengement.
# There isn't any check on the access.
# In case of a not fully allocated buffer be sure to access over the current the inserted elements.
class MultipleRoundBuffer():
	
	def __init__(self, number_of_buffers=1, capacity=100):
		assert number_of_buffers > 0
		assert capacity > 0
		
		self.number_of_buffers = number_of_buffers
		self.current_capacity = 0		
		self.capacity = capacity
		self.current = 0
		self.count = 0
		
		self.items = [[] for i in range(self.number_of_buffers)]
	
	
	def __getitem__(self, index):
		return self.items[index]
	
	
	def insert(self, item):
		if self.current >= self.current_capacity:
			for i in range(self.number_of_buffers):
				self.items[i].append(item[i])
			self.current_capacity += 1
		else:
			for i in range(self.number_of_buffers):
				self.items[i][self.current] = item[i]
		
		self.current = (self.current + 1) % self.capacity
		self.count = min([self.count + 1, self.capacity])

		
	def sample(self, size):
		ids = np.random.randint(self.count, size=min(size, self.count))
		return [[self.items[buffer][id] for id in ids] for buffer in range(self.number_of_buffers)]

		

class GameStats():
	
	def __init__(self):
		self.games_rewards = []
		self.current_game_rewards = []
	
	def addReward(self, reward, endgame):
		self.current_game_rewards.append(reward)
		
		if endgame:
			self.games_rewards.append(self.current_game_rewards)
			self.current_game_rewards = []		
	
	def deleteCurrentGame(self):
		self.current_game_rewards = []
	
	def totalGames(self):
		return len(self.games_rewards)
		
	def lastGameReward(self):
		if len(self.games_rewards) == 0:
			return 0
		return sum(self.games_rewards[-1])
		
	def meanReward(self):
		return np.mean(np.array(self.games_rewards), axis=1)
	
	def cumulativeReward(self):
		return [sum(self.games_rewards[i]) for i in range(len(self.games_rewards))]
	
	def stepsPerGame(self):
		return [len(self.games_rewards[i]) for i in range(len(self.games_rewards))]
		
		
		
		
def plotGameStats(game_stats, path, episodes_span=25, labels=None, colours=None):
	
	
	fig, ax = plt.subplots()
	
	for n, gs in enumerate(game_stats):
		missing_episodes = episodes_span - (gs.totalGames() % episodes_span)
		
		rewards = gs.cumulativeReward() + [0] * missing_episodes
		rewards = np.reshape(rewards, [-1, episodes_span])[:-1] #last games is not considered because of padding
		mean_span_rewards = np.mean(rewards, axis=1)
		
		games_steps = gs.stepsPerGame() + [0] * missing_episodes
		games_steps = np.reshape(games_steps, [-1, episodes_span])[:-1]
		games_steps = np.sum(games_steps, axis=1)
		span_steps = [np.sum(games_steps[:i+1]) for i in range(len(games_steps))] #np.sum(games_steps, axis=1)
		
		label = labels[n] if labels != None else ""
		colour = colours[n] if colours != None else "#0F0F0F"
		ax.plot(span_steps, mean_span_rewards, label=label, color=colour, linewidth=2-(n))
		
	ax.set(xlabel='frames', ylabel='reward (clipped)', title='Mean rewards over {:d} episodes'.format(episodes_span))
	ax.grid()
	if labels != None:
		ax.legend()
	
	fig.savefig(path)
	plt.show()
	
	