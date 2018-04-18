from __future__ import print_function
import keras
from keras.layers import Input, Dense
from keras.models import Model 
from keras.models import load_model

import numpy as np 

from copy import copy, deepcopy
from itertools import chain
import os

NETWORK_DIRECTORY = 'keras_networks'
os.makedirs(NETWORK_DIRECTORY, exist_ok=True)

# funnel layers are layers that can be shared that convert a block of data to some other representation
# it is similar to word2vec

def lchain(x):
	return list(chain(*x))

# constants to define; default
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
	def __init__(self, id, game, load_params=None, **hyperparameters):
		"""
		TODO: generate a keras network that will be used to make predictions for players using this AI
		if the 
		"""
		self.id = id
		self.n_players = game.n_players

		self.hyperparameters = deepcopy(NETWORK_HYPERPARAMETERS)
		self.hyperparameters.update(hyperparameters)

		# define universal inputs inputs
		self.player_inputs = [Input(shape=(12 + self.n_players,)) for i in range(self.n_players)]
		self.reserved_inputs = [[Input(shape=(15,)) for _ in range(3)] for i in range(self.n_players)]

		self.game_inputs = Input(shape=(8,))
		self.game_objective_inputs = [Input(shape=(5,)) for _ in range(self.n_players+1)]
		self.game_cards_inputs = [[Input(shape=(15,)) for position in range(4)] for tier in range(3)]

		self.model_inputs =  (
			self.player_inputs + # 4
			[self.game_inputs] + # 1
			self.game_objective_inputs + # 1 + n_players (3-5) 
			lchain(self.reserved_inputs) + # 12
			lchain(self.game_cards_inputs) # 12
		)

		if load_params is not None:
			model_index = load_params.get('index', 0)
			model_main_name = load_params.get('name', 'default')
			self.load_models(main_name = model_main_name, model_index=model_index)

		else:
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
					next_layer = funnel_layer(keras.layers.concatenate([next_layer, reserved_networks[i]]))
				else:
					next_layer = funnel_layer(next_layer)
			player_networks.append(next_layer)

		## game network
		# note that there will be no intermediate injection here
		

		game_funnel_layers = [Dense(n, activation='relu') for n in self.hyperparameters['game_funnel_layers']]
		game_objective_funnel_layers = [Dense(n, activation='relu') for n in self.hyperparameters['game_objective_funnel_layers']]
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
				for funnel_layer in game_card_funnel_layers:
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

	def map_game_input_to_network_inputs(self, input):
		"""
		will align the numpy/dict input returned by Player.full_serialization() 
		to the inputs used by the neural network
		"""

		# player inputs, game input, game objectives, player reserved, game cards
		# print(input)
		"""
			self.model_inputs =  (
			self.player_inputs + # 4
			[self.game_inputs] + # 1
			self.game_objective_inputs + # 1 + n_players (3-5) 
			lchain(self.reserved_inputs) + # 12
			lchain(self.game_cards_inputs) # 12
		)
		"""
		self_raw_input = [np.concatenate([input['self'][k] for k in ['gems','discount','points','order']])]

		self_reserved_input = input['self']['reserved_cards']

		player_raw_inputs = [
			
			np.concatenate(
				[
					serializations[k] for k in ['gems','discount','points','order']
				]
			)
			for serializations in input['other_players']
				
		]
		

		player_reserved_inputs = lchain([x['reserved_cards'] for x in input['other_players']])

		game_raw_input = [np.concatenate([
			input['game'][k] for k in ['gems','turn']
		])
		] # note that the turn number + last turn is in the 'turn' key
		

		game_objective_input = input['game']['objectives']

		game_card_input = lchain(input['game']['available_cards'])

		return (
			self_raw_input + player_raw_inputs +
			game_raw_input +
			game_objective_input + 
			self_reserved_input + player_reserved_inputs +
			game_card_input
		)


	def make_predictions(self, inputs):
		unstacked_network_inputs = [self.map_game_input_to_network_inputs(input) for input in inputs]
		# each entry of the above list is a list of numpy arrays
		stacked_network_inputs = [np.vstack(input_array) for input_array in zip(*unstacked_network_inputs)]

		win_prediction = self.win_model.predict(stacked_network_inputs)[:,0]
		q_predictions = [None for _ in self.q_networks]

		for i, q_index in enumerate(self.hyperparameters['output_layers']):
			q_predictions[i] = self.q_networks[i].predict(stacked_network_inputs)[:,0]

		return({'win_prediction': win_prediction, 'q_predictions': q_predictions})

	def save_models(self, main_name,  index=0):
		"""
		saves the neural network configuration to a file
		"""
		for model_name in ['win','Q1','Q3','Q5']:
			if model_name == 'win':
				model = self.win_model
			elif model_name=='Q1':
				model = self.q_networks[0]
			elif model_name=='Q3':
				model = self.q_networks[1]
			elif model_name=='Q5':
				model = self.q_networks[2]
			filename = '{main_name}_{model_name}_{index}.h5'.format(
				main_name=main_name,
				model_name=model_name,
				index=index
			)
			print('saving {filename}'.format(filename=filename))
			model.save(os.path.join(NETWORK_DIRECTORY, filename))

	def load_models(self, main_name, index=0):
		"""
		loads model from file
		"""
		self.q_networks = [None, None, None]

		for model_name in ['win','Q1','Q3','Q5']:
			filename = '{main_name}_{model_name}_{index}.h5'.format(
				main_name=main_name,
				model_name=model_name,
				index=index
			)
			print('loading {filename}'.format(filename=filename))
			model = load_model(os.path.join(NETWORK_DIRECTORY, filename))
			if model_name == 'win':
				self.win_model = model
			elif model_name=='Q1':
				self.q_networks[0] = model
			elif model_name=='Q3':
				self.q_networks[1] = model
			elif model_name=='Q5':
				self.q_networks[2] = model

	def train_models(self, n_epochs=10, batch_size=1000, verbose=0):
		for model_name in ['win', 'Q1', 'Q3', 'Q5']:
			print('training %s model' % model_name)
			x, y = self.prepare_data(model_name)
			if model_name == 'win':
				model = self.win_model
			elif model_name == 'Q1':
				model = self.q_networks[0]
			elif model_name == 'Q3':
				model = self.q_networks[1]
			elif model_name == 'Q5':
				model = self.q_networks[2]
			model.fit(x, y, epochs=n_epochs, batch_size=batch_size, verbose=verbose)

	def load_extended_history_from_player(self, player):
		"""
		note: this is not thread-safe, so this would need to be changed to parallelize in future
		"""
		self.extended_serialized_history = player.extended_serialized_action_history
		self.lagged_q_state_history = player.extended_lagged_q_state_history

	def prepare_data(self, model_name):
		x_unstacked = [self.map_game_input_to_network_inputs(row) for row in self.extended_serialized_history] #np.vstack(self.extended_serialized_history)
		x = [np.vstack(input_array) for input_array in zip(*x_unstacked)]
		y = np.asarray([row[model_name] for row in self.lagged_q_state_history])
		return x, y
			