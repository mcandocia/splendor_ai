from __future__ import print_function
from player import Player 

from constants import *
from copy import copy, deepcopy
from random import shuffle

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

		self.objectives = shuffle(deepcopy(OBJECTIVE_CAREDS))[:self.n_players]

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
		max_score = max([player.score for player in self.players])
		highest_scoring_players = [player for player in self.players if player.score==max_score]
		lowest_number_of_cards = min([player.n_cards for player in highest_scoring_players])
		winning_players = [player for player in highest_scoring_players if player.n_cards==lowest_number_of_cards]
		for player in winning_players:
			player.win = True 

	def serialize(self, gem_change = None, available_card_change=None):
		"""
		describes the state of the cards and gems on the board in a numeric format using a numpy array

		IMPORTANT: the position of the cards on the board is unimportant within a tier. when implementing a 
		neural network, try to get the values of the connections from their nodes to be the same as each other 
		(at least within the same tier)
		"""

	def save_state(self):
		"""
		to be used in conjunction with save_state() from players to save a state and then undo it as necessary
		"""

	def load_state(self):
		"""
		to revert to previous state stored by self.save_state()
		"""