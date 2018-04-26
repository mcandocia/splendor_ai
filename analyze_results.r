library(plyr)
library(dplyr)
library(ggplot2)
library(scales)
library(cetcolor)
library(reshape2)

simdata = read.csv('splendor_comparison_data.csv')

better_label_sizes = theme(
  plot.title = element_text(size=rel(1.8)),
  plot.subtitle=element_text(size=rel(1.3)),
  axis.title = element_text(size=rel(1.8)),
  axis.text = element_text(size=rel(1.8)),
  legend.title=element_text(size=rel(1.8)),
  legend.text=element_text(size=rel(1.1))
)

# let's look at win rates vs. matchups

# linear model

# glm(unique_player_win ~ factor(unique_player_level):factor(other_player_level):factor(unique_player_id), data=simdata)

win_summaries = simdata %>% 
  group_by(unique_player_level, other_player_level) %>%
  summarize(win_rate = mean(unique_player_win)) %>%
  ungroup() %>%
  mutate(target_player_level = factor(unique_player_level+1, levels=c(1,5,11)+1),
         other_player_level = factor(other_player_level+1, levels=c(1,5,11)+1)
  ) %>% 
  mutate(label = percent(win_rate))

ggplot(win_summaries) + geom_tile(aes(x=target_player_level, y=other_player_level, fill=win_rate)) + 
  ggtitle('Win Rate of AI vs. 3 Other AI by Number of Training Rounds') + 
  xlab('Training Rounds for Reference Player') + ylab('Training Rounds for Other Players') + 
  scale_fill_gradient2('Win Rate', midpoint=0.25, label=percent) + 
  geom_text(aes(x=target_player_level, y = other_player_level, label=label), size=12) +
  better_label_sizes

# visualize turns
# NOT USEFUL
turn_summaries = simdata %>%
  filter(unique_player_level==other_player_level) %>%
  group_by(unique_player_level) %>%
  summarise(round_average = mean(n_rounds))

ggplot(simdata %>% filter(unique_player_level==other_player_level)) + 
  stat_count(aes(x=n_rounds, y=..prop.., group=unique_player_level)) + 
  facet_grid(unique_player_level~.) + 
  


# let's look at cards purchased by tier and training level

card_summaries = simdata %>% 
  filter(unique_player_level==other_player_level) %>%
  group_by(unique_player_level) %>%
  summarize(tier_1_average = mean(
    c(player_1_ncards_tier_1,player_2_ncards_tier_1, player_0_ncards_tier_1, player_3_ncards_tier_1)
    ),
    tier_2_average = mean(
      c(player_1_ncards_tier_2,player_2_ncards_tier_2, player_0_ncards_tier_2, player_3_ncards_tier_2)
    ),
    tier_3_average = mean(
      c(player_1_ncards_tier_3,player_2_ncards_tier_3, player_0_ncards_tier_3, player_3_ncards_tier_3)
    )
    
)

card_summaries_winner = simdata %>% 
  filter(unique_player_level==other_player_level) %>%
  group_by(unique_player_level) %>%
  summarize(tier_1_average = mean(
    player_winner_ncards_tier_1
  ),
  tier_2_average = mean(
    player_winner_ncards_tier_2
  ),
  tier_3_average = mean(
    player_winner_ncards_tier_3
  )
  
  )

objective_summaries = simdata %>% 
  filter(unique_player_level==other_player_level) %>%
  group_by(unique_player_level) %>%
  summarize(objective_average = mean(
    c(player_1_n_objectives, player_0_n_objectives, player_2_n_objectives, player_3_n_objectives))
  
  )

objective_summaries_winner = simdata %>% 
  filter(unique_player_level==other_player_level) %>%
  group_by(unique_player_level) %>%
  summarize(objective_average = mean(
    player_winner_n_objectives)
    
  )

point_summaries = simdata %>%
  filter(unique_player_level==other_player_level) %>%
  group_by(unique_player_level) %>%
  summarize(point_average = mean(
    c(player_0_points, player_1_points, player_2_points, player_3_points)
  ))


reserve_summaries = simdata %>%
  filter(unique_player_level==other_player_level) %>%
  group_by(unique_player_level) %>%
  summarize(reserved_cards_average = mean(
    c(player_0_n_reserved_cards, player_1_n_reserved_cards, player_2_n_reserved_cards, player_3_n_reserved_cards)
  ))


# card_summaries_winner
# objective_summaries_winner
winner_cards = melt(card_summaries_winner, id.var = 'unique_player_level') %>% 
  mutate(variable = gsub('_',' ', variable))


ggplot(winner_cards %>% mutate(unique_player_level=factor(unique_player_level +1, levels=c(2,6,12))), 
                                                          aes(x=variable, y=value)) + 
  geom_bar(stat='identity', mapping=aes(fill=unique_player_level), position='dodge') +
  scale_fill_discrete('Number of \nTraining Rounds') + 
  xlab('Tier Average') + ylab('Average Number of Cards Owned by Winner') + 
  scale_y_continuous(breaks=0:11) + better_label_sizes +
  ggtitle('Average Number of Cards Per Tier Owned by Winner of Splendor',
          subtitle='sample size of 200 for each round') + 
  geom_text(aes(label=round(value, digits=1),  group=unique_player_level), 
            vjust='inward', size=8.5,
            position=position_dodge(width=0.9))

ggplot(objective_summaries_winner %>% mutate(unique_player_level=factor(unique_player_level +1, levels=c(2,6,12)))) + 
  geom_bar(aes(x=unique_player_level, y=objective_average, fill='orange'), stat='identity') + 
  xlab('Number of AI Training Rounds') + ylab('Average Number of Objectives Completed by Winner') + 
  ggtitle("Average Number of Objectives Completed by Winner at Different Training Rounds",
          subtitle='sample size of 200 for each round') + 
  better_label_sizes + 
  scale_fill_identity() + 
  geom_text(aes(label=round(objective_average, 2), y=objective_average, x=unique_player_level),
            vjust='inward', size=13)

ggplot(win_summaries) + geom_tile(aes(x=target_player_level, y=other_player_level, fill=win_rate)) + 
  ggtitle('Win Rate of AI vs. 3 Other AI by Number of Training Rounds') + 
  xlab('Training Rounds for Reference Player') + ylab('Training Rounds for Other Players') + 
  scale_fill_gradient2('Win Rate', midpoint=0.25, label=percent) + 
  geom_text(aes(x=target_player_level, y = other_player_level, label=label), size=12) +
  better_label_sizes


ggplot(turn_summaries %>% mutate(unique_player_level=factor(unique_player_level +1, levels=c(2,6,12)))) + 
  geom_bar(aes(x=unique_player_level, y=round_average, fill='orange'), stat='identity') + 
  xlab('Number of AI Training Rounds') + ylab('Average Number of Turns Per Player') + 
  ggtitle("Average Number of Turns per Player at Different Training Rounds",
          subtitle='sample size of 200 for each round') + 
  better_label_sizes + 
  scale_fill_identity() + 
  geom_text(aes(label=round(round_average, 1), y=round_average, x=unique_player_level),
            vjust='inward', size=13)