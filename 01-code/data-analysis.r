library(tidyverse)
library(tidyboot)
library(brms)

d <- read_csv('results.csv')

d |> select(starts_with("prob")) |> 
  pivot_longer(everything()) |> 
  group_by(name) |> 
  tidyboot::tidyboot_mean(value)


# sanity check if randomization worked

table(d$trigger_feature)
table(d$nuisance_feature)
table(d$trigger_word)

# effects of nuisance feature or target feature

# there do seem to be differences based on the nuisance feature
d |> group_by(nuisance_feature) |> 
  tidyboot::tidyboot_mean(prob_interpretation_target)

fit_nuisance_feature <- brms::brm(
  formula = prob_interpretation_target ~ nuisance_feature, 
  family = brms::brmsfamily("beta"), 
  data = d)

d |> ggplot(aes(x = prob_interpretation_target)) +
  facet_grid(nuisance_feature ~.) +
  geom_density(alpha = 0.3) +
  geom_rug()

# there might be trigger-feature effects!
# interpretation seems to be better for 'texture' trigger words, than for 'shape' than for 'color'
d |> group_by(trigger_feature) |> 
  tidyboot::tidyboot_mean(prob_interpretation_target)

fit_trigger_feature <- brms::brm(
  formula = prob_interpretation_target ~ trigger_feature, 
  family = brms::brmsfamily("beta"), 
  data = d)

faintr::compare_groups(
  fit_trigger_feature, 
  higher = trigger_feature == "texture",
  lower = trigger_feature == "shape",
)

d |> ggplot(aes(x = prob_interpretation_target)) +
  facet_grid(trigger_feature ~.) +
  geom_density(alpha = 0.3) +
  geom_rug()

# positional effects

d <- d |> 
  mutate(interpretation_indices = str_c(
    "[",
    interpretation_index_target,
    ",",
    interpretation_index_distractor,
    "]"
  ))

table(d$interpretation_indices)

d |> group_by(interpretation_index_target, interpretation_index_competitor) |> 
  tidyboot::tidyboot_mean(prob_interpretation_target)
  
d |> ggplot(aes(x = interpretation_index_target, y = prob_interpretation_target)) +
  geom_jitter()
  

d |> group_by(interpretation_index_competitor) |> 
  tidyboot::tidyboot_mean(prob_interpretation_competitor)
  
d |> ggplot(aes(x = interpretation_index_competitor, y = prob_interpretation_competitor)) +
  geom_jitter()

d |> ggplot(aes(x = interpretation_indices, y = prob_interpretation_target)) +
  geom_jitter(alpha = 0.5)



