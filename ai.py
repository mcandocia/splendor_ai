from __future__ import print_function
import keras
from keras.layers import Input, Dense
from keras.models import Model 
from keras.models import load_model

from copy import copy, deepcopy
from itertools import chain
import os

NETWORK_DIRECTORY = 'keras_networks'
os.makedirs(NETWORK_DIRECTORY, exist_ok=True)

# funnel layers are layers that can be shared that convert a block of data to some other representation
# it is similar to word2vec

# constants to define 
NETWORK_HYPERPARAMETERS = {
	# player input
	'player_funnel_layers': [15,12,10],
	'reserved_funnel_layers': [12,10,],
	'inject_reserved_at_player_funnel_index': 1, # 0 same as input, 1 = as first layer, etc.
	'card_funnel_layers': [12,12,8],
	# game input
	'game_funnel_layers': [15, 12, 10],
	'game_objective_funnel_layers': [10, 8],
	'game_card_funnel_layers': [15, 12, 10],
	# overall
	'main_dense_layers': [72, 32, 12], #this is when everything is combined

	# output layers
	# this does not include the win layer
	'output_layers': [
		{
			'name': 'Q1',
			'lag': 1,
			'score': 1,
			'discount': 0.1,
			'gems': 0.01,

		},
		{
			'name': 'Q3',
			'lag':  3,
			'score': 1,
			'discount': 0.1,
			'gems': 0,
		},
		{
			'name': 'Q5',
			'lag': 5,
			'score': 1,
			'discount': 0.05,
			'gems': 0,
		},
	],
}

class SplendorAI(object):
	def __init__(self, id, game, load_filename=None, **hyperparameters):
		"""
		TODO: generate a keras network that will be used to make predictions for players using this AI
		if the 
		"""
		self.id = id
		self.hyperparameters = deepcopy(NETWORK_HYPERPARAMETERS)
		self.hyperparameters.update(hyperparameters)

		# define universal inputs inputs
		self.player_inputs = [Input(shape=(12 + self.n_players,)) for i in range(self.n_players)]
		self.reserved_inputs = [[Input(shape=(15,)) for _ in range(3)] for i in range(self.n_players)]

		self.game_inputs = Input(shape=(8,))
		self.game_objective_inputs = [Input(shape=(5,)) for _ in range(self.n_players+1)]
		self.game_cards_inputs = [[Input(shape=(15,)) for position in range(4)] for tier in range(3)]

		self.model_inputs =  (
			self.player_inputs + 
			self.reserved_inputs + 
			[self.game_inputs] +
			self.game_objective_inputs + 
			list(chain(*self.reserved_inputs)) +
			list(chain(*self.game_cards_inputs))
		)

		self.n_players = len(game.players)
		self.q_networks = []
		for q_network_config in self.hyperparameters['output_layers']:
			q_network = self.initialize_network_layers()
			q_network = Dense(1, activation='relu')(q_network)
			q_model = Model(inputs=self.model_inputs, outputs=q_network)
			q_model.compile(
				optimizer='rmsprop',
				loss='mse',
			)
			self.q_networks.append(q_model)

		# win network

		self.w_network = self.initialize_network_layers()
		self.w_network = Dense(1, activation='sigmoid')(self.w_network)

		self.win_model = Model(inputs = self.model_inputs, outputs=self.w_network)
		self.win_model.compile(optimizer='rmsprop',
			loss='binary_crossentropy',
			metrics=['accuracy']
		)


		if load_filename is None:
			pass
		else:
			pass

	def save_win_network(self):
		pass

	def load_win_network(self, name=None):
		pass

	def save_q_network(self, q_index):
		pass

	def load_q_network(self, q_index, name=None):
		pass

	def initialize_network_layers(self):
		"""
		this will utilize the inputs given to the network
		in order to create a final dense layer that should be connected to some sort of output
		"""

		## player shared networks
		# this ensures that each player's state is transformed to something more representative of their current position
		player_funnel_layers = [Dense(n, activation='relu') for n in self.hyperparameters['player_funnel_layers']]
		reserved_funnel_layers = [Dense(n, activation='relu') for n in self.hyperparameters['reserved_funnel_layers']]

		reserved_networks = []

		for i in range(self.n_players):
			next_layers = self.reserved_inputs[i]
			for j, funnel_layer in enumerate(reserved_funnel_layers):
				next_layers = [funnel_layer(next_layer) for next_layer in next_layers]
			reserved_networks.append(keras.layers.concatenate(next_layers))
				
		player_networks = []

		injection_index = self.hyperparameters['inject_reserved_at_player_funnel_index']

		for i in range(self.n_players):
			next_layer = self.player_inputs[i]
			for j, funnel_layer in enumerate(player_funnel_layers):
				if j==injection_index:
					next_layer = funnel_layer(keras.layers.concatenate(next_layer, reserved_networks[i]))
				else:
					next_layer = funnel_layer(next_layer)
			player_networks.append(next_layer)

		## game network
		# note that there will be no intermediate injection here
		

		game_funnel_layers = [Dense(n, activation='relu') for n in self.hyperparameters['game_funnel_layers']]
		game_objective_funnel_layers = [Dense(n, activation='relu') for n in self.hyperparameters['game_objetive_funnel_layers']]
		game_card_funnel_layers = [Dense(n, activation='relu') for n in self.hyperparameters['game_card_funnel_layers']]

		# regular game input
		next_layer = self.game_inputs
		for funnel_layer in game_funnel_layers:
			next_layer = funnel_layer(next_layer)

		game_network = next_layer 

		#objective game input
		objective_layers = []
		for i in range(self.n_players+1):
			next_layer = self.game_objective_inputs[i]
			for funnel_layer in game_objective_funnel_layers:
				next_layer = funnel_layer(next_layer)
			objective_layers.append(next_layer)

		objective_network = keras.layers.concatenate(objective_layers)

		#card inputs
		card_layers = []
		for tier in range(3):
			for position in range(4):
				next_layer = self.game_cards_inputs[tier][position]
				for funnel_layer in self.game_card_funnel_layers:
					next_layer = funnel_layer(next_layer)
				card_layers.append(next_layer)

		card_network = keras.layers.concatenate(card_layers)

		# build main network from player_networks, game_network, card_network, and objective_network

		main_dense_layers = [Dense(n, activation='relu') for n in self.hyperparameters['main_dense_layers']]

		# add player inputs for skip-layer connections

		main_dense_input = keras.layers.concatenate(player_networks + [game_network] + [objective_network] + [card_network] +
			self.player_inputs
		)


		next_layer = main_dense_input
		for layer in main_dense_layers:
			next_layer = layer(next_layer)

		return next_layer



	def create_self_network(self):

		# gems input, 6

		# discount input, 5

		# points, 1

		# order, 2-4

		# reserved cards, 3x15 shared

	def create_other_player_network(self):
		
		# gems input, 6

		# discount input, 5

		# points, 1

		# order, 2-4

		# reserved_cards, 3x15 shared


	def create_game_network(self):
		# gems available, 6

		# cards on board, 12 x 15

		# objectives available, (3-5) x 5

		# turn, 1

		# is last turn, 1

	def determine_winning_probability(self, state):
		"""
		takes in a matrix of possible game states and predicts the probability of winning
		for each of them

		returns a numpy array of the probabilities
		"""

	def predict_q_score(self, state, q_lag):
		"""
		takes in a matrix of possible game states and estimates the q-score given a certain lag
		"""

	def save(self, filename):
		"""
		saves the neural network configuration to a file
		"""