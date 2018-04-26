card_order = sort(c('bakery','ranch','mine','forest','stadium',...))
player_list = c('player_1', 'player_2', 'player_3', 'player_4')

serialize_player_input <- function(player){
 input = game_data[[player]][['coins']]
 for (card in card_order){
  input = c(input, game_data[[player]][[card]])
 }
 input
}

game_input_serialized = unlist(sapply(
  player_list, serialize_player_input
))

# game_input_serialized initial value: 
# c(2,1,1,0,0,0,0,0,0,0,0,0,...,2,1,1,0,0,0,...)






game_data = list(
	player_1=list(
		coins=2,
		cards=list(
			wheat_field=1,
			bakery=1,
			ranch=0,
			...
		)
	),
	player_2=list(
		…,
	),
	...
)
			
		
