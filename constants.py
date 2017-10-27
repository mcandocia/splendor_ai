from __future__ import print_function
import csv


CARD_CSV_FILENAME = '/ntfsl/workspace/splendor_ai/card_data.csv'

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
	with open(CARD_CSV_FILENAME, 'rb') as f:
		reader = csv.reader(f)
		next(f)
		for row in reader:
			data = {
			'tier':row[0],
			'color':row[1],
			'points':row[2],
			'cost':{
			color:int_or_zero(value) for color, value in 
			zip(['black','white','red','blue','green'],
				row[3:])
			}
			}
			tier = int(row[0])
			if tier==1:
				TIER_1_CARDS.append(data)
			elif tier==2:
				TIER_2_CARDS.append(data)
			else:
				TIER_3_CARDS.append(data)

#loads cards into respective lists
LOAD_CARDS()

OBJECTIVE_CARDS = [
{'green':4,'blue:4'},
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

#subtract 1 for each player less than 4 at start of game
COLOR_STOCKPILE_AMOUNT = 7
GOLD_STOCKPILE_AMOUNT = 5

#important when serializing color-dependent data
COLOR_ORDER = ['black','white','red','blue','green','gold']
#for card prices and requirements of objectives
COST_COLOR_ORDER = ['black','white','red','blue','green']