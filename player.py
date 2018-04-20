from __future__ import print_function
from constants import * 
from itertools import combinations
from itertools import zip_longest
from itertools import product
from collections import Counter

from copy import copy, deepcopy

import numpy as np 

from ai import NETWORK_HYPERPARAMETERS
from ai import SplendorAI

q_loadings = NETWORK_HYPERPARAMETERS['output_layers']

"""
uncommon strategies not supported:

* using gold instead of a regular color in order to prevent a player from grabbing one or two gems of a particular color
on their turn (fairly uncommon)

* telling a player what your hidden card is so that it is more obvious to them that you can
purchase something/are more likely to purchase something that may further their interests in some way (very rare)

* take 1 gem each from 2 separate piles to allow another player to take a certain color or to allow yourself to take 3 next turn (never seen it)

"""

# TODO : test run

def wrap_if_not_list(x):
	if isinstance(x, (list, tuple)):
		return x
	else:
		return [x]

# used to convert score weighting to positive weights
def elu(x):
	if x < 500:
		return np.log1p(np.exp(x))
	else:
		return x

def normalize(x):
	return np.asarray(x)/np.sum(x)

def get_phase_parameters(phase):
	"""
	training will be divided into 5 phases

	"""
	if phase==1:
		return {
			'Q1': 0.5,
			'Q3': 0.3,
			'Q5': 0.15,
			'win': 0.05,
		}
	elif phase==2:
		return {
			'Q1': 0.4,
			'Q3': 0.25,
			'Q5': 0.2,
			'win': 0.15,
		}
	elif phase==3:
		return {
			'Q1': 0.25,
			'Q3': 0.25,
			'Q5': 0.25,
			'win': 0.25,
		}
	elif phase==4:
		return {
			'Q1': 0.15,
			'Q3': 0.2,
			'Q5': 0.35,
			'win': 0.3,
		}
	elif phase==5:
		return {
			'Q1': 0.05,
			'Q3': 0.1,
			'Q5': 0.35,
			'win': 0.50,
		}

class Player(object):
	def __init__(self, game, id, order, ai=None, decision_weighting=None, temperature=1, record_plain_history=False, hyperparameters=None):
		self.game = game 
		# this is important for keeping track of which player has which id
		self.id = id
		self.order = order 
		if ai is not None:
			self.ai = ai
		else:
			if hyperparameters is not None:
				self.ai = SplendorAI(id=id, game=game, **hyperparameters)
			else:
				self.ai = SplendorAI(id=id, game=game)

		self.points = 0
		# this should be retrieved via get_phase_parameters()
		if decision_weighting is None:
			self.decision_weighting = get_phase_parameters(1)
		else:
			self.decision_weighting=decision_weighting

		self.temperature = temperature

		self.record_plain_history = record_plain_history
		#cards that contribute to cost reduction and points
		self.owned_cards = []
		#cards that can be purchased only by player
		self.reserved_cards = []
		#faster way of keeping track of cards
		self.n_cards = 0
		self.n_reserved_cards = 0
		# updating this isn't implemented yet
		self.n_reserved_cards_tiers = [0 for _ in range(3)]
		self.gems = ColorCombination(True, **{color:0 for color in COLOR_ORDER})
		self.n_gems = 0
		self.discount = ColorCombination(**{color:0 for color in COST_COLOR_ORDER})
		self.objectives = []
		self.win = False
		#self.draw = False#will allow multiple victories in rare instances

		## describes the history for the current game
		# this should have a dict of basic game information
		# describes q state at the beginning of a turn
		self.q_state_history = []
		self.lagged_q_state_history = []
		# describes the actions in plain terms
		self.plain_action_history = []
		# these are the raw serializations used for each player in a turn
		self.serialized_action_history = []

		## describes serialized history for all games; this will be used to train the neural network
		#this should be in [[data, win, game_id], ...] format
		self.extended_serialized_action_history = []
		self.extended_plain_action_history = []
		self.extended_lagged_q_state_history = []

		# this describes the response variable at each time step
		# this should be of the format [{'win': value, 'Q1': value, 'Q3': value, 'Q5': value}, ...]
		# the reason they aren't all put into one is that extending the array requires copying and that's very expensive to
		# do over thousands of simulations
		# this is only calculated at the end of the game when all values of Q1, Q3, Q5, and win state can be certain
		self.extended_output = []

	def set_game(self, game):
		self.game = game 
		self.ai.game = game 


	def set_decision_weighting(self, weights):
		self.decision_weighting = weights

	def set_temperature(self, temperature):
		self.temperature=temperature

	def get_other_players(self):
		"""
		run when initializing the game to make it easy to access other players 
		for serializing
		"""
		n_players = len(self.game.players)
		if self.order==0:
			self.other_players = self.game.players[1:]
		elif self.order < n_players - 1:
			self.other_players = self.game.players[self.order+1:] + self.game.players[:self.order]
		else:
			self.other_players = self.game.players[:self.order]

	def take_turn(self):
		"""
		TODO: record player state at end of turn after action is made

		the player takes a turn

		"""
		self.record_q_state()

		self.make_turn_action()

		self.objective_check()
		if not self.game.last_turn:
			self.victory_check()

	def make_choice(self, score):
		"""
		uses temperature to randomly decide on decision index
		based on Boltzmann Distribution
		"""
		# at T=1, minimum probability for action is about 1/20,000
		score_ = np.maximum(score - np.max(score), -10)
		score = score_ + -1e100 * (score == -1e100)
		# condition prevents underflow errors
		if self.temperature < 0.01:
			choice = np.argmax(score)
		else:
			score = np.exp(score/self.temperature)
		choice = np.random.choice(
			np.arange(len(score)),
			p=score/np.sum(score)
		)
		# worst-case scenario with probability fudging
		if score[choice] < -1e50:
			for i in range(3):
				if score[i] > -1e50:
					return i
		return choice



	def decide_on_action(self, purchasing_options, reserving_options, gem_taking_options):
		# determine parameters used to weight the 3 categories amongst themselves, then the options among the categories
		# first, use group averaging to decide among 3 possible options (if valid) among outcome

		if False:
			print(type(purchasing_options))
			print(type(reserving_options))
			print(type(gem_taking_options))
		if purchasing_options is None:
			purchasing_weight = -1e100
		else:
			purchasing_scores = purchasing_options['score'] # [option['score'] for option in purchasing_options]
			purchasing_weight = np.max(purchasing_scores)

		if reserving_options is None:
			reserving_weight = -1e100
		else:
			reserving_scores = reserving_options['score'] # [option['score'] for option in reserving_options]
			reserving_weight = np.max(reserving_scores)

		if gem_taking_options is None:
			gem_taking_weight = -1e100
		else:
			gem_scores = gem_taking_options['score'] # [option['score'] for option in gem_taking_options]
			gem_taking_weight = np.max(gem_scores)

		if purchasing_weight + reserving_weight + gem_taking_weight==0:
			print("WARNING: NO ACTIONS CAN BE TAKEN!")
			return None

		action = ['purchase','reserve','take_gems'][self.make_choice(
			np.asarray([purchasing_weight, reserving_weight, gem_taking_weight])
		)]
		#print(action)

		if action=='purchase':
			which_purchase = self.make_choice(purchasing_scores)
			serialization = purchasing_options['serializations'][which_purchase]
			return (
				action, 
				{
					'card_changes': purchasing_options['actions']['card_changes'][which_purchase],
					'gem_changes': purchasing_options['actions']['gem_changes'][which_purchase],
				}, 
				serialization
			)
		elif action=='reserve':
			which_reserve = self.make_choice(reserving_scores)
			serialization = reserving_options['serializations'][which_reserve]
			return (
				action, 
				{
					'reservation_changes': reserving_options['actions']['reservation_changes'][which_reserve],
					'gem_changes': reserving_options['actions']['gem_changes'][which_reserve],
				}, 
				serialization,
			)
		elif action=='take_gems':
			which_gems = self.make_choice(gem_scores)
			serialization = gem_taking_options['serializations'][which_gems]
			return (
				action, 
				{
					'gem_changes': gem_taking_options['actions']['gem_changes'][which_gems]
				}, 
				serialization
			)


	def make_turn_action(self):
		"""
		
		"""
		purchasing_options = self.simulate_purchasing_options()
		reserving_options = self.simulate_reserving_options()
		gem_taking_options = self.simulate_gem_taking_options()

		# now determine best course of action based on q-weighting
		action_type, action_params, serialization = self.decide_on_action(
			purchasing_options=purchasing_options,
			reserving_options=reserving_options,
			gem_taking_options=gem_taking_options
		)
		if action_type is None:
			print(':-(')
		elif action_type == 'purchase':
			self.purchase_card(action_params)
		elif action_type == 'reserve':
			self.reserve_card(action_params)
		elif action_type == 'take_gems':
			self.take_gems(action_params['gem_changes'])


		# HISTORY UPDATED
		self.serialized_action_history.append(serialization)
		if self.record_plain_history:
			self.plain_action_history.append({'action_type': action_type, 'action_params': action_params})

		
		

	"""
	BELOW ARE SIMULATION FUNCTIONS
	for each of these functions, the following should be done:
		1. all possible moves should be determined
		2. for each possible move, a semi-simulation should be made of the resulting board state. 
		   I recommend implementing & using save_state()/load_state() for both the Player and Game classes
		   so that you can directly modify these; I have put a "move_cards" argument to some functions so that they won't 
		   shift around cards if you want to avoid that; you should use copy/deepcopy to copy lists/nested objects 
		   when creating these states
		3. for each state, there should be a serialization of the entire game from that player's point of view. For example,
		   a game of texas hold 'em would have a binary (1s and 0s) vector of length 52 describing  your hand, a binary vector
		   describing the state of the cards in the center, and perhaps some other vectors describing betting/folding of 
		   the other players (since you can't see their cards). In this game, you may be able to use integer values for gem costs.
		4. the SplendorAI class needs to be further developed, but it will take in these serializations and output the probabilities;
		   With these probabilities, a decision should be made by processing them through determine_best_option(). Initially I want the
		   actual probability to have less impact on the chosen decision in order to increase randomness, but as the network becomes
		   trained, I want the better probabilities to have a much higher chance of being chosen
		5. The chosen action will be taken with the keyword arguments provided in the option, and the turn will continue.


	"""

	def calculate_score(self, predictions):
		win_prediction = predictions['win_prediction']
		Q1_prediction = predictions['q_predictions'][0]
		Q3_prediction = predictions['q_predictions'][1]
		Q5_prediction = predictions['q_predictions'][2]
		score = (
			(win_prediction * 15) * self.decision_weighting['win'] + 
			Q1_prediction * self.decision_weighting['Q1'] + 
			Q3_prediction * self.decision_weighting['Q3'] + 
			Q5_prediction * self.decision_weighting['Q5']
		)
		return score

	def simulate_purchasing_options(self):
		"""
		creates all possible purchasing options and returns corresponding probabilities and action types

		FORMAT:
		 list of
		  [prob, action_kwargs, action_type]
		"""

		purchasing_options = []
		payment_options = []

		# cards on board
		for tier in [1,2,3]:
			available_cards = self.game.get_available_cards(tier=tier)
			deck_cards = self.game.get_deck(tier=tier)
			can_be_replaced = len(deck_cards) > 0
			for i, card in enumerate(available_cards):
				net_cost = (card['cost'] - self.discount).truncate_negatives()
				if self.gems.can_pay_for(net_cost):
					payment_options.append(self.gems.calculate_actual_cost(net_cost))
					card_purchase = {
						'position': i,
						'tier': tier,
						'can_be_replaced': can_be_replaced,
						'type': 'board',
						'card': card
					}
					purchasing_options.append(card_purchase)

		# reserved cards
		for i, card in enumerate(self.reserved_cards):
			if card is not None:
				net_cost = (card['cost'] - self.discount).truncate_negatives()
				if self.gems.can_pay_for(net_cost):
					payment_options.append(self.gems.calculate_actual_cost(net_cost))
					card_purchase = {
						'position': i,
						'tier': tier,
						'can_be_replaced': False,
						'type': 'reserved',
						'card': card

					}
					purchasing_options.append(card_purchase)

		if len(purchasing_options) > 0:
			purchasing_serializations = self.full_serializations(
				gem_changes=payment_options,
				card_changes=purchasing_options,
			)
			predictions = self.ai.make_predictions(purchasing_serializations)
			return {
				'predictions': predictions,
				'score': self.calculate_score(predictions),
				'serializations': purchasing_serializations,
				'actions': {
					'gem_changes': payment_options,
					'card_changes': purchasing_options,
				}
			}
		else:
			return None

	def simulate_reserving_options(self):
		"""
		creates all possible reserving options and returns corresponding probabilities and action types

		FORMAT:
		 list of
		  [prob, action_kwargs, action_type]
		"""
		# skip if 3 cards already reserved
		if len(self.reserved_cards) == 3:
			return None

		# cannot get gold if none exists or you have 10 gems
		if self.game.gems.gold==0 or self.gems.count() == 10:
			gem_change = ColorCombination(uses_gold=True)
		else:
			gem_change = ColorCombination(gold=1,uses_gold=True)

		reservation_options = []
		# cards on board
		for tier in [1,2,3]:
			tier_cards = self.game.get_available_cards(tier=tier)
			for position, card in enumerate(tier_cards):
				if card is not None:
					reservation_options.append({'tier':tier, 'position':position, 'type':'board', 'card': card})

		# cards on top of deck
		for tier in [1,2,3]:
			if len(self.game.get_deck(tier=tier)) > 0:
				reservation_options.append({'tier':tier, 'position': 0, 'type':'topdeck', 'card':make_blank_card(tier)})

		if len(reservation_options) > 0:
			gem_changes = [gem_change] * len(reservation_options)
			reservation_serializations = self.full_serializations(
				gem_changes=gem_changes,
				reservation_changes = reservation_options
			)
			# print(reservation_serializations)

			predictions = self.ai.make_predictions(reservation_serializations)
			return {
				'predictions':predictions, 
				'score': self.calculate_score(predictions),
				'serializations': reservation_serializations,
				'actions': {
					'gem_changes': gem_changes, 
					'reservation_changes':reservation_options
				}
			}
		else:
			return None


	def simulate_gem_taking_options(self):
		"""
		creates all possible gem-taking options and returns corresponding probabilities and action types

		FORMAT:
		 list of
		  [prob, action_kwargs, action_type]
		"""
		# if no gems, provide an out in case no moves can be made
		if self.game.gems.count_nongold() == 0:
			gem_combinations = [ColorCombination(True)]
		else:
			# determine all combinations
			gem_combinations = self.take_gems_options()
		if len(gem_combinations) > 0:
			gem_serializations = self.full_serializations(gem_changes=gem_combinations)
			predictions = self.ai.make_predictions(gem_serializations)
			return {
				'predictions': predictions,
				'score': self.calculate_score(predictions),
				'serializations': gem_serializations,
				'actions': {
					'gem_changes': gem_combinations,
				}
			}
		else:
			return None

	def save_state(self):
		"""
		I recommend using this for storing a few attributes when simulating so you 
		can directly modify some of the player attribute variables;
		you can also have game.save_state() called from here
		"""

	def load_state(self):
		"""
		loads previous state after self.save_state() has been called;
		you can alsoe have game.load_state() called from here
		"""

	#GENERAL CHECKING AND ACTION FUNCTIONS
	def get_available_cards(self, tier):
		"""
		does not include reserved cards
		"""
		if tier==1:
			cards = self.game.available_tier_1_cards
		elif tier==2:
			cards = self.game.available_tier_2_cards
		elif tier==3:
			cards = self.game.available_tier_3_cards
		elif tier=='all':
			# only for recordkeeping
			return {
			    'tier 1': self.game.available_tier_1_cards,
			    'tier 2': self.game.available_tier_2_cards,
			    'tier 3': self.game.available_tier_3_cards
			}
		return cards 

	def card_in_position_purchasing_cost(self, tier, position):
		card = self.get_available_cards(tier)[position]
		return self.card_purchasing_cost(card)

	def card_purchasing_cost(self, card):
		"""
		this doubles as a boolean check
		"""
		cost = card['cost'] - self.discount
		

		#calculates the surplus of each color for each part of the cost
		difference = self.gems - cost 
		#checks to see if extra gold is enough to purchase 
		if not self.gems.can_pay_for(cost):
			return None
		else:
			return cost.truncate_negatives() #new_gems = self.gems.make_payment(cost)

	def purchase_card(self, action_data):
		"""
		this will translate the card input to purchase either available or reserved card functions below
		"""
		card = action_data['card_changes']['card']
		tier = card['tier']
		position = action_data['card_changes']['position']
		purchase_type = action_data['card_changes']['type']
		if purchase_type == 'reserved':
			self.purchase_reserved_card(position=position)
		elif purchase_type == 'board':
			self.purchase_available_card(tier=tier, position=position)

	def reserve_card(self, action_data):
		"""
		this will translate the reserving card input to reserve a card eitehr on board on on top of a deck
		"""
		card = action_data['reservation_changes']['card']
		tier = card['tier']
		position = action_data['reservation_changes']['position']
		reserve_type = action_data['reservation_changes']['type']
		if reserve_type == 'topdeck':
			self.reserve_card_on_top(tier=tier)
		elif reserve_type == 'board':
			self.reserve_card_on_board(tier=tier, position=position)
		self.n_reserved_cards += 1
		self.n_reserved_cards_tiers[card['tier']-1] += 1

	def purchase_available_card(self, tier, position, move_cards=True):
		"""
		note: I am unsure why I have a move_cards parameter here...
		"""
		card = self.get_available_cards(tier).pop(position)
		card_cost = self.card_purchasing_cost(card)
		actual_card_cost = ((self.gems + self.discount).calculate_actual_cost(card['cost'])-self.discount).truncate_negatives()
		original_discount_and_gems = self.discount + self.gems
		#add points
		self.points+= card['points']
		#add color # still technically valid with new class
		self.discount[card['color']] += 1
		#subtract gems
		original_gems = self.gems.__copy__()
		self.gems = self.gems-actual_card_cost#self.gems.make_payment(card_cost)
		self.game.gems+=actual_card_cost
		if self.gems.has_any_negatives() or self.game.gems.has_any_negatives():
			print(card)
			print('current')
			print(self.gems)
			print('original')
			print(original_gems)
			print('original gems and discount')
			print(original_discount_and_gems)
			#print('game')
			#print(self.game.gems)
			print('cost')
			print(card_cost)
			print('actual cost')
			print(actual_card_cost)
			raise ValueError('unexpected negative')

		self.n_gems = self.gems.count()
		'''
		for color, amount in card_cost.iteritems():
			self.gems[color] -= amount 
			self.n_gems -= amount
		'''
		#add card to inventory
		self.n_cards += 1
		if move_cards:
			self.owned_cards.append(card)
			#add new card, display warning if deck is empty
			new_card_added = self.game.add_top_card_to_available(tier)
			if not new_card_added and False:
				print("WARNING: tier %s deck ran empty!" % tier)

	def purchase_reserved_card(self,  position, move_cards=True):
		# dunno why I have move_cards variable
		if move_cards:
			card = self.reserved_cards.pop(position)
		else:
			card = self.reserved_cards[position]
		card_cost = self.card_purchasing_cost(card)
		actual_card_cost = ((self.gems + self.discount).calculate_actual_cost(card['cost'])-self.discount).truncate_negatives()
		#add points
		self.points+= card['points']
		#add color
		self.discount[card['color']] += 1
		#subtract gems
		self.gems = self.gems-actual_card_cost
		self.game.gems+=actual_card_cost
		self.n_gems = self.gems.count()
		
		#add card to inventory
		self.n_cards += 1
		if move_cards:
			self.owned_cards.append(card)
		self.n_reserved_cards -= 1
		self.n_reserved_cards_tiers[card['tier']-1] -= 1

	def reserve_card_on_top(self, tier, move_cards=True):
		# self.n_reserved_cards += 1
		if move_cards:
			self.reserved_cards.append(self.game.get_deck(tier).pop())
		self.n_reserved_cards_tiers[tier-1] += 1
		if self.n_gems < 10 and self.game.gems['gold'] > 0:
			self.take_gems(ColorCombination(True, **{'gold':1}))
		return 0

	def reserve_card_on_board(self, tier, position):
		# self.n_reserved_cards += 1
		self.reserved_cards.append(self.game.get_available_cards(tier).pop(position))
		self.game.add_top_card_to_available(tier)
		self.n_reserved_cards_tiers[tier-1] += 1
		if self.n_gems < 10 and self.game.gems['gold'] > 0:
			self.take_gems(ColorCombination(True, **{'gold':1}))
		return 0 

	def objective_check(self):
		"""
		at end of turn, checks to see if any objectives are met
		if no, then returns 0
		if only one, then it adds that and returns a 1
		if more than one, decides which one, then adds it, then returns 1
		"""
		possible_objective_ids = []
		for i, objective in enumerate(self.game.objectives):
			if self.discount.can_pay_for(objective):
				possible_objective_ids.append(i)
		if len(possible_objective_ids) == 0:
			return 0
		elif len(possible_objective_ids) == 1:
			self.points+=3 
			self.objectives.append(self.game.objectives.pop(possible_objective_ids[0]))
		else:
			objective_id = self.decide_on_objective(possible_objective_ids)
			self.points+=3
			self.objectives.append(self.game.objectives.pop(objective_id))	
		return 1

	def decide_on_objective(self, possible_objectives):
		"""
		MEH METHOD: randomly choose an objective
		AI METHOD: forecast all possibilities (3 at most, but most likely 2) and choose the one that most likely results in winning
		DERIVED METHOD: check to see if any other players are within a turn of getting the other card; not 100% exact bc of reserved cards, and might be tedious
		"""
		return np.random.choice(possible_objectives)

	def victory_check(self):
		"""
		if premature, will not compare number of cards and will 
		"""
		if self.points >= 15:
			self.game.last_turn = True

	def take_gems(self, gems):
		"""
		adds gems to inventory
		will not update code for ColorCombination class
		"""
		self.gems = self.gems + gems
		self.game.gems = self.game.gems - gems 
		self.n_gems += gems.count()
		'''
		for color, amount in gems.iteritems():
			self.gems[color] += amount
			self.game.gems[color] -= amount
			self.n_gems +=1
		return 0
		'''

	def take_gems_options(self):


		# returns a list of ColorCombination objects
		single_colors = set()
		double_colors = set()
		for color in COST_COLOR_ORDER:
			if self.game.gems[color] > 0:
				if self.game.gems[color] > 3:
					double_colors.add(color)
				single_colors.add(color)

		positive_possibilities = []

		for color in double_colors:
			positive_possibilities.append(ColorCombination(True, **{color:2}))

		if len(single_colors)==1:
			color = list(single_colors)[0]
			positive_possibilities.append(ColorCombination(**{color:1}))
		elif len(single_colors)==2:
			positive_possibilities += color_combinations(single_colors, 2)
		else:
			positive_possibilities += color_combinations(single_colors, 3)

		# determines returning requirements
		if self.n_gems <= 7:
			return positive_possibilities
		else:
			actual_possibilities = self.calculate_gem_returns(positive_possibilities)
			return actual_possibilities

	def calculate_gem_returns(self, possibilities):
		# I am using a set to remove duplicates
		possibility_tuples = set()
		n_gems = self.n_gems
		for possibility in possibilities:
			n = possibility.count()
			number_to_return = n + n_gems - 10
			if number_to_return <= 0:
				possibility_tuples.add(possibility.as_tuple())
			else:
				return_combos = [ColorCombination(True, **Counter(x)) for x in combinations((self.gems + possibility).expand(), number_to_return)]
				net_changes = [possibility-combo for combo in return_combos]
				for change in net_changes:
					possibility_tuples.add(change.as_tuple())
		# convert hashable tuples back to ColorCombination objects
		return [convert_tuple_to_color_combination(x) for x in possibility_tuples]


	def reset(self, reset_extended_history=False):
		"""
		used when you want to keep the same players (saves some AI loading, possibly) 
		and start a new game; extended history is kept unless the flag is set to true to
		reset it
		"""
		self.points = 0
		#cards that contribute to cost reduction and points
		self.owned_cards = []
		#cards that can be purchased only by player
		self.reserved_cards = []
		#faster way of keeping track of cards
		self.n_cards = 0
		self.n_reserved_cards = 0
		self.n_reserved_cards_tiers = [0 for i in range(3)]
		self.gems = ColorCombination(uses_gold=True, **{color:0 for color in COLOR_ORDER})
		self.n_gems = 0
		self.discount = ColorCombination(**{color:0 for color in COST_COLOR_ORDER})
		self.objectives = []
		self.win = False
		#self.draw = False#will allow multiple victories in rare instances

		#describes the history for the current game
		# describes game state at the beginning of a turn
		self.q_state_history = []
		self.lagged_q_state_history = []
		# describes the actions in plain terms
		self.plain_action_history = []
		# these are the raw serializations used for each player in a turn
		self.serialized_action_history = []
		if reset_extended_history:
			self.extended_serialized_action_history = []
			self.extended_plain_action_history = []
			self.extended_lagged_q_state_history = []

	# length 59-61 (57 + number of players)
	def serialize(self, gem_change=None, card_change=None, reservation_change=None, from_own_perspective=True):
		"""
		# note: this is for purchasing a card, not reserving
		card_change - {'card': {...}, 'reserved_index': [0,1,2, None], 'position': [0,1,2,3, None]}
		  reserved_index is the index number of the reserved card in the player's inventory
		  position is the position of the card in its tier row if it's not reserved; does nothing in this function

		card_reservation - {'type': 'board'/'topdeck', 'tier': [1,2,3], 'position':[0,1,2,3]}
		  type is if itis on the board or if it is on top of a deck
		  tier corresponds to the row or deck tier
		  position corresponds to the position on the board; 0 if on top of the deck
		"""
		# gem calculations
		if gem_change is not None:
			theoretical_gems = self.gems + gem_change
		else:
			theoretical_gems = self.gems 

		gem_serialization = theoretical_gems.serialize()
		# print(self.gems.uses_gold)
		# print(self.gems.serialize())
		# card changes
		reserved_card_serializations = [
			serialize_card(card, not from_own_perspective)
			for card in self.reserved_cards
		]

		# hypothetical reservation
		if reservation_change is not None:
			tier = card_reservation['tier']
			if reservation_change['type'] == 'topdeck':
				reserved_card_serializations.append(
					serialize_card(make_blank_card(tier=tier))
				)
			else:
				position = reservation_change['position']
				reserved_card_serializations.append(
					serialize_card(self.game.get_available_cards(tier=tier)[position])
				)

		n_reserved = len(reserved_card_serializations)

		# fill in with blank reservation slots (they still need 0-filled serializations)
		if n_reserved < 3:
			reserved_card_serializations.extend((3-n_reserved) * [PURE_BLANK_CARD_SERIALIZATION])

		if card_change is not None:
			card = card_change['card']
			theoretical_points = card_change['card']['points'] + self.points 
			if card_change['type'] == 'reserved':
				reserved_index = card_change['position']
			else:
				reserved_index = None
			color = card['color']
			
			if reserved_index is not None:
				reserved_card_serializations[reserved_index] = PURE_BLANK_CARD_SERIALIZATION
			else:
				# do nothing because that doesn't affect the player, only the board state which is handled elsewhere
				pass
			theoretical_discount = self.discount + ColorCombination(**{color:1})
		else:
			theoretical_points = self.points
			theoretical_discount = self.discount 

		discount_serialization = theoretical_discount.serialize()
		points_serialization = np.asarray([theoretical_points])

		# order serialization to enforce symmetry requirements
		order_serialization = np.zeros(4)
		order_serialization[self.order] = 1.
		# print('-----')
		# print(gem_serialization)
		# print(discount_serialization)
		# print(points_serialization)
		# print(order_serialization)
		return {
			'gems': gem_serialization, #6
			'discount': discount_serialization, #5
			'points': points_serialization, # 1
			'reserved_cards': reserved_card_serializations, # 3 * 15
			'order': order_serialization, #2-4 (number of players in the game)
		}


	def full_serializations(self, gem_changes=None, card_changes=None, reservation_changes=None):
		# first get serializations of other players, since those are static
		other_player_serializations = [
			player.serialize(from_own_perspective=False)
			for player in self.other_players
		]

		# for each change combination, get own serialization and board serialization
		self_serializations = []
		game_serializations = []
		for gem_change, card_change, reservation_change in zip_longest(
			wrap_if_not_list(gem_changes), 
			wrap_if_not_list(card_changes), 
			wrap_if_not_list(reservation_changes)):

			self_serializations.append(self.serialize(gem_change=gem_change, card_change=card_change))

			game_serializations.append(
				self.game.serialize(
					gem_change=gem_change, 
					available_card_change=None, 
					reservation_change=reservation_change
				)
			)

		# return those values, which will be ready to be consumed by neural network input
		return [
		    {
		    	'other_players': other_player_serializations,
		    	'self': self_serialization,
		    	'game': game_serialization
		    } 
		    for self_serialization, game_serialization in zip(self_serializations, game_serializations)
		]
		'''
		return {
			'other_players': other_player_serializations,
			'self': self_serializations,
			'game': game_serializations,
		}
		'''

	def copy_plain_data(self):
		"""
		the game will use this when recording plain history
		"""
		return {
			'gems': deepcopy(self.gems),
			'discounts': deepcopy(self.discount),
			'cards': deepcopy(self.owned_cards),
			'n_cards': len(self.owned_cards),
			'objectives': deepcopy(self.objectives),
			'reserved_cards': deepcopy(self.reserved_cards),
			'points': copy(self.points),
			'order': self.order,
			'id': self.id,
			'n_reserved_cards': self.n_reserved_cards,
			'n_reserved_cards_tiers': self.n_reserved_cards_tiers,
			'win': self.win,
			'decision_weights': {
				'temperature': self.temperature,
				'decision_weighting': deepcopy(self.decision_weighting),
			},
			'n_gems': self.n_gems,
		}

	def record_q_state(self):

		q_state = {
			v['name']: v['score'] * self.points + v['discount'] * self.discount.count() + v['gems'] * self.gems.count()
			for v in q_loadings
		}
		self.q_state_history.append(q_state)



	def record_extended_history(self):
		# determine win status
		win_value = self.win * 1

		# retroactively apply Q-scores
		n_turns = len(self.q_state_history)
		q1_threshold = n_turns - 2
		q3_threshold = n_turns - 4
		q5_threshold = n_turns - 6
		for i in range(n_turns-1):
			data = {}
			if i <= q1_threshold:
				data['Q1'] = self.q_state_history[i+1]['Q1']
			if i <= q3_threshold:
				data['Q3'] = self.q_state_history[i+3]['Q3']
			else:
				data['Q3'] = self.q_state_history[n_turns-1]['Q3']
			if i <= q5_threshold:
				data['Q5'] = self.q_state_history[i+5]['Q5']
			else:
				data['Q5'] = self.q_state_history[n_turns-1]['Q5']
			data['win'] = win_value
			self.lagged_q_state_history.append(data)

		# write extended history
		self.extended_serialized_action_history.extend(deepcopy(self.serialized_action_history))

		self.extended_lagged_q_state_history.extend(deepcopy(self.lagged_q_state_history))

		# write plain history if applicable
		if self.record_plain_history:
			# this needs to be exported elsewhere in order to be useful for anything
			self.extended_plain_action_history.extend(deepcopy(self.plain_action_history))

	def transfer_history_to_ai(self):
		self.ai.load_extended_history_from_player(self)

	def copy_ai_from_other_player(self, player, index=0):
		self.ai.load_

def color_combinations(colors, n):
	"""
	returns the cost in the form of a dictionary 
	"""
	combos = combinations(colors, n)
	return [ColorCombination(use_gold=True, **{color:1 for color in combo}) for combo in combos]
