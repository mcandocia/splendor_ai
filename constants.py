from __future__ import print_function
import json
import csv

import numpy as np

#important when serializing color-dependent data
COLOR_ORDER = ['black','white','red','blue','green','gold']
#for card prices and requirements of objectives
COST_COLOR_ORDER = ['black','white','red','blue','green']

class ColorCombination(object):
	"""
	this represents color quantities to make arithmetic simpler
	"""
	def __init__(self, uses_gold=False, **colors):
		self.uses_gold = uses_gold
		if not uses_gold:
			self.possible_colors = COST_COLOR_ORDER
		else:
			self.possible_colors = COLOR_ORDER 
		for color in self.possible_colors:
			setattr(self, color, colors.get(color, 0))

	def __getitem__(self, key):
		return getattr(self, key)

	def __setitem__(self, attr, key):
		setattr(self, attr, key)

	def __add__(self, c2):
		colors = {}
		for color in self.possible_colors:
			colors[color] = getattr(self, color) + getattr(c2, color)

		return ColorCombination(self.uses_gold or c2.uses_gold, **colors)

	def __sub__(self, c2):
		colors = {}
		if c2.uses_gold:
			for color in self.possible_colors:
				colors[color] = getattr(self, color) - getattr(c2, color)
		else:
			for color in COST_COLOR_ORDER:
				colors[color] = getattr(self, color) - getattr(c2, color)
			if self.uses_gold:
				colors['gold'] = self.gold


		return ColorCombination(self.uses_gold, **colors)

	def __mul__(self, factor):
		colors = {color:getattr(self, color) * factor for color in self.possible_colors}
		return ColorCombination(self.uses_gold, **colors)

	def __rmul__(self, factor):
		return self.__mul__(factor)

	def __neg__(self):
		colors = {color:-getattr(self, color) for color in self.possible_colors}
		return ColorCombination(self.uses_gold, **colors)

	def __str__(self):
		return json.dumps(self.as_dict(), indent=4)

	def __repr__(self):
		return 'ColorCombination object: \n' + str(self)

	def as_dict(self):
		return {color:self[color] for color in self.possible_colors}

	def count(self):
		return sum([getattr(self, color) for color in self.possible_colors])

	def count_nonnegative(self):
		return sum([getattr(x, color) for x in self.possible_colors  if getattr(x, color) >= 0])

	def truncate_negatives(self):
		colors = {color:max(0, getattr(self, color)) for color in self.possible_colors}
		return ColorCombination(self.uses_gold, **colors)

	def keep_only_negatives(self):
		# used for calculating what needs to be converted to gold
		colors = {color:min(0, getattr(self, color)) for color in self.possible_colors}
		return ColorCombination(self.uses_gold, **colors)

	def __copy__(self):
		colors = {color:getattr(self, color) for color in self.possible_colors}
		return ColorCombination(self.uses_gold, **colors)

	def __deepcopy__(self, *args):
		return self.__copy__()

	def can_pay_for(self, c2):
		difference = c2 - self
		net_shortfall = sum([max(0, getattr(difference, color)) for color in COST_COLOR_ORDER])
		return self.gold >= net_shortfall

	def calculate_actual_cost(self, c2):
		# get actual cost; used after can_pay_for() returns True
		difference = c2 - self
		net_shortfall = sum([max(0, getattr(difference, color)) for color in COST_COLOR_ORDER])
		cost = ONE_GOLD * net_shortfall + (c2 + difference.keep_only_negatives())
		return cost

	def make_payment(self, c2):
		"""
		calculates difference, but will take out gold when needed
		this should be done after can_pay_for() checks that it's okay
		"""
		difference = self - c2
		#print(difference)
		net_shortfall = sum([max(0, -getattr(difference, color)) for color in COST_COLOR_ORDER])
		print('net shortfall ', net_shortfall)
		if net_shortfall > 0:
			difference.gold -= net_shortfall
			for color in difference.possible_colors:
				setattr(difference, color,
					max(getattr(difference, color), 0)
				)
		return difference

	def serialize(self):
		return np.asarray([getattr(self, color) for color in self.possible_colors])

#length = 15
def serialize_card(card, allow_hidden=False):
	# this will return a card that is unknown to other players instead of serializing it
	# requires allow_hidden=True so one doesn't hide their own cards from themselves
	if card.get('hidden', False) and allow_hidden:
		return serialize_card(make_blank_card(card['tier']))

	color_serialization = np.asarray([1*(card['color']==color) for color in COST_COLOR_ORDER])
	cost_serialization = card['cost'].serialize()
	point_serialization = np.asarray([card['points']])
	tier_serialization = np.asarray([1*card['tier']==x for x in range(1,4)])
	# this allows a blank card to input *some* value into the network
	blank_serialization = np.asarray([card.get('blank_value', 0)])
	return np.concatenate((
		color_serialization, 
		cost_serialization, 
		point_serialization, 
		tier_serialization, 
		blank_serialization), 
	)

# length=5
def serialize_objective(objective):
	if objective is not None:
		return objective.serialize()
	else:
		return ColorCombination().serialize()


# used so that it can be serialized easily
def make_blank_card(tier, blank_value=1):
	# specify blank_value=0 if it can't be replaced
	# OR use to indicate that is it not known to other players 
	BLANK_CARD = {
		'tier': tier,
		'color': '',
		'points': 0,
		'cost': ColorCombination(),
		'blank': True,
		# for cards on board, this indicates if it will be replaced or not
		# for cards in player reserve zone, this indicates if card data is known or not
		'blank_value': blank_value, 
	}
	return BLANK_CARD

PURE_BLANK_CARD_SERIALIZATION = serialize_card(make_blank_card(0, 0))

# a small test case
# v = ColorCombination(True, gold=3, red=1)
# v2 = ColorCombination(red=2, black=1)
# print(v.make_payment(v2))



# these are used for simple arithmetic with ColorCombination class
EACH_COLOR = ColorCombination(True, **{color:1 for color in COST_COLOR_ORDER})

ONE_GOLD = ColorCombination(True, gold=1)

CARD_CSV_FILENAME = 'card_data.csv' # '/ntfsl/workspace/splendor_ai/card_data.csv'

CARD_LIST = []
#IF POINTS OR COST IS 0, THEN THAT VALUE WILL BE INSERTED LATER
TIER_1_CARDS = []
TIER_2_CARDS = []
TIER_3_CARDS = []

def int_or_zero(x):
	if x=='':
		return 0
	else:
		return int(x)

def LOAD_CARDS():
	global TIER_1_CARDS
	global TIER_2_CARDS
	global TIER_3_CARDS
	global CARD_LIST
	with open(CARD_CSV_FILENAME, 'r') as f:
		reader = csv.reader(f)
		next(f)
		for row in reader:
			data = {
			'tier':row[0],
			'color':row[1],
			'points':row[2],
			'cost_':{
				color:int_or_zero(value) for color, value in 
				zip(['black','white','red','blue','green'],
					row[3:])
				}
			}
			data['cost'] = ColorCombination(**data['cost_'])
			tier = int(row[0])
			if tier==1:
				TIER_1_CARDS.append(data)
			elif tier==2:
				TIER_2_CARDS.append(data)
			else:
				TIER_3_CARDS.append(data)

#loads cards into respective lists
LOAD_CARDS()

# aka "nobles"
OBJECTIVE_CARDS_ = [
	{'green':4,'blue':4},
	{'black':4,'white':4},
	{'green':3,'red':3,'blue':3},
	{'red':4,'green':4},
	{'black':4,'red':4},
	{'black':3,'red':3,'white':3},
	{'blue':4,'white':4},
	{'black':3,'red':3,'green':3},
	{'green':3,'blue':3,'white':3},
	{'black':3,'blue':3,'white':3}
]

OBJECTIVE_CARDS = [ColorCombination(**v) for v in OBJECTIVE_CARDS_]

#subtract 1 for each player less than 4 at start of game
COLOR_STOCKPILE_AMOUNT = 7
GOLD_STOCKPILE_AMOUNT = 5

# how much is available at the start
GEMS_PILE = ColorCombination(True, 
	gold=GOLD_STOCKPILE_AMOUNT, 
	**{color:COLOR_STOCKPILE_AMOUNT for color in COST_COLOR_ORDER}
)