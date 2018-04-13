from __future__ import print_function
from player import Player 

from constants import *
from copy import copy, deepcopy
from random import shuffle
from itertools import chain

import numpy as np

class Game(object):
	"""
	TODO: add some game-based hyperparameters that determine 
	how players use the posterior probabilities of their decisions
	in order to make an action

	"""
	def __init__(self, id, n_players=4, players=None, shuffle_players=True):
		self.id = id 
		if players is not None:
			self.n_players = len(players)
			self.players=players 
			if shuffle_players:
				self.shuffle_players()
		else:
			self.n_players = n_players
			self.players = [Player(self, id=i, order=i) for i in range(n_players)]
		self.turn = -1

		# make sure each player knows where other players are relative to them
		for player in self.players:
			player.get_other_players()

		self.generate_initial_cards()
		#self.gems = {color:COLOR_STOCKPILE_AMOUNT - (4-self.n_players) for color in COLOR_ORDER}
		#self.gems['gold'] = GOLD_STOCKPILE_AMOUNT
		self.gems = GEMS_PILE - (4-self.n_players) * EACH_COLOR
		self.last_turn = False

	#TODO: add methods to append 
	def run(self):
		while not self.last_turn:
			for player in self.players:
				self.turn += 1
				player.take_turn()
		self.assign_winner()

	def shuffle_players(self):
		"""
		before starting a new game when you get a previous list of players
		"""
		shuffle(self.players)
		for i, player in enumerate(self.players):
			player.order = i 

	def generate_initial_cards(self):
		self.tier_1_cards = shuffle(deepcopy(TIER_1_CARDS))
		self.tier_2_cards = shuffle(deepcopy(TIER_2_CARDS))
		self.tier_3_cards = shuffle(deepcopy(TIER_3_CARDS))

		self.objectives = shuffle(deepcopy(OBJECTIVE_CARDS))[:self.n_players + 1]

		self.available_tier_1_cards = [self.tier_1_cards.pop() for _ in range(4)]
		self.available_tier_2_cards = [self.tier_2_cards.pop() for _ in range(4)]
		self.available_tier_3_cards = [self.tier_3_cards.pop() for _ in range(4)]

	def add_top_card_to_available(self, tier):
		"""
		after a card is purchased, a new card from the deck should be added
		returns False if deck run empty (probably only happens in tier 1)
		"""
		cards = self.get_deck(tier)
		if len(cards)==0:
			return False
		self.get_available_cards(tier).append(cards.pop())
		return True

	def get_deck(self, tier):
		if tier==1:
			return self.tier_1_cards
		elif tier==2:
			return self.tier_2_cards
		else:
			return self.tier_3_cards 

	def get_available_cards(self, tier):
		if tier==1:
			return self.available_tier_1_cards
		elif tier==2:
			return self.available_tier_2_cards
		else:
			return self.available_tier_3_cards

	def assign_winner(self):
		"""
		TODO: update player extended histories and assign the proper win (1 or 0) value to the second element of each row
		"""
		# score primary objective
		max_score = max([player.score for player in self.players])
		highest_scoring_players = [player for player in self.players if player.score==max_score]

		# nobles secondary objective
		max_objectives = max([len(player.objectives) for player in highest_scoring_players])
		highest_scoring_players = [player for player in highest_scoring_players if len(player.objectives)==max_objectives]

		# efficiency tertiary objective
		lowest_number_of_cards = min([player.n_cards for player in highest_scoring_players])
		winning_players = [player for player in highest_scoring_players if player.n_cards==lowest_number_of_cards]
		for player in winning_players:
			player.win = True 

	# length = 203-213 (188 + (3-5) * 5)
	def serialize(self, gem_change = None, available_card_change=None, reservation_change=None):
		"""
		describes the state of the cards and gems on the board in a numeric format using a numpy array

		IMPORTANT: the position of the cards on the board is unimportant within a tier. when implementing a 
		neural network, try to get the values of the connections from their nodes to be the same as each other 
		(at least within the same tier)

		available_card_change - {'tier': [1,2,3], 'position': [0,1,2,3], 'can_be_replaced': True/False, 'type': 'board'/'reserved'}
		"""

		# gem calculations
		if gem_change is not None:
			theoretical_gems = self.gems + gem_change
		else:
			theoretical_gems = self.gems 
		gem_serialization = theoretical_gems.serialize()

		# available card change
		
		tier_1_cards_serialization = [serialize_card(card) for card in self.available_tier_1_cards]
		tier_2_cards_serialization = [serialize_card(card) for card in self.available_tier_2_cards]
		tier_3_cards_serialization = [serialize_card(card) for card in self.available_tier_3_cards]

		# make an adjustment to one of the row serializations if the reservation change is not null
		if reservation_change is not None:
			if reservation_change['type'] == 'board':
				tier = reservation_change['tier']
				position = reservation_change['position']
				if tier==1:
					target_serialization = tier_1_cards_serialization
					blank_value = (len(self.tier_1_cards) > 0) * 1
				elif tier==2:
					target_serialization = tier_2_cards_serialization
					blank_value = (len(self.tier_2_cards) > 0) * 1
				elif tier==3:
					target_serialization = tier_3_cards_serialization
					blank_value = (len(self.tier_3_cards) > 0) * 1

				target_serialization[position] = serialize_card(make_blank_card(tier=tier, blank_value=blank_value))

		available_cards_serializations = [tier_1_cards_serialization, tier_2_cards_serialization, tier_3_cards_serialization]

		if available_card_change is not None:
			if available_card_change['type'] == 'board':
				tier = available_card_change['tier']
				position = available_card_change['position']
				can_be_replaced = available_card_change.get('can_be_replaced', True)
				available_cards_serializations[tier-1][position] = serialize_card(
					make_blank_card(tier,can_be_replaced )
				)

		# objectives serializations
		objectives_serializations = [serialize_objective(objective) for objective in self.objectives]

		# turn and last turn
		turn_serialization = [np.asarray(self.turn)]
		last_turn_serialization = [np.asarray(1*self.last_turn)]

		return {
			'gems': gem_serialization, # 6
			'available_cards': available_cards_serializations, # 12 x 15
			'objectives': objectives_serializations, # (3-5) x 5
			'turn': turn_serialization + last_turn_serialization, #2
		}
		




	def save_state(self):
		"""
		to be used in conjunction with save_state() from players to save a state and then undo it as necessary
		"""

	def load_state(self):
		"""
		to revert to previous state stored by self.save_state()
		"""