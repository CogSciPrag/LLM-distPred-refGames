source('00-premable.r')

# Qing & Franke data
QF_prod  <- prop.table(c(135 + 119, 25 + 9 , 0)) |> round(3)
QF_inter <- prop.table(c(115+62, 117+65, 1)) |> round(3)

d <- read_csv('02-data-prepped.csv') |> 
  mutate(
    response   = factor(response  , levels = c("target", "competitor", "distractor")),
    prediction = factor(prediction, levels = c("target", "competitor", "distractor")),
    condition  = factor(condition,  levels = c("production", "interpretation"))
  ) 

d_counts <- d |> 
  count(condition, prediction, response) |> 
  group_by(condition, prediction) |> 
  mutate(total = sum(n)) |> 
  ungroup() |> 
  mutate(proportion = n / total)
  
  
d_counts |> 
  ggplot(aes(x = prediction, y = n, fill = response)) +
  geom_col(position = "dodge") +
  ylab("") +
  facet_grid(~ condition)

########################################################################################
# RQ1: Do we see more target choices when model arg-max-predicts target choices? => yes!
########################################################################################

# it seems that in those cases where the model arg-max predicts the 'competitor' option,
# the competitor is also selected more often

# run stats for this:

fit_argmaxpred <- brm(
  formula(response_target ~ condition * prediction),
  data = d |> mutate(response_target = response == "target"),
  family = bernoulli()
)

# main effect of 'prediction = target"? => yes
faintr::compare_groups(
  fit_argmaxpred,
  higher = prediction == "target",
  lower  = prediction != "target"
)

# main effect of 'prediction = target" for production? => yes
faintr::compare_groups(
  fit_argmaxpred,
  higher = prediction == "target" & condition == "production",
  lower  = prediction != "target" & condition == "production"
)

# main effect of 'prediction = target" for interpretation? => yes
faintr::compare_groups(
  fit_argmaxpred,
  higher = prediction == "target" & condition == "interpretation",
  lower  = prediction != "target" & condition == "interpretation"
)

########################################################################################
# RQ3: Does the position of options influence the choice for machine and humans?
########################################################################################

make_position_string <- function(target_index, competitor_index, condition = "interpretation") {
  position <- c('d', 'd', 'd', 'd')
  position[target_index] <- 't'
  position[competitor_index] <- 'c'
  if (condition == "interpretation") {
    position <- position[1:3]
  }
  str_c("[", str_c(position, sep = ",", collapse = ","), "]")
} 


position_production <- map_chr(
  1:nrow(d),
  function(i) {return(make_position_string(d$production_index_target[i] + 1 ,
                                           d$production_index_competitor[i] + 1, 
                                           "production"))}
)

position_interpretation <- map_chr(
  1:nrow(d),
  function(i) {return(make_position_string(d$interpretation_index_target[i] + 1,
                                           d$interpretation_index_competitor[i] + 1 ))}
) |> 
  factor(levels = c(
    "[t,c,d]", 
    "[t,d,c]",
    "[d,t,c]", 
    "[d,c,t]",
    "[c,t,d]", 
    "[c,d,t]" 
  ))


# PRODUCTION
# human response proportions per position

d$position_production <- position_production

d_response_props_per_position <- d |> 
  filter(condition == "production") |> 
  count(position_production, response) |> 
  group_by(position_production) |> 
  mutate(total = sum(n),
         proportion = n/total) |> 
  ungroup()

d_response_props_per_position |> 
  ggplot(aes(x = position_production, y = proportion, fill = response)) +
  geom_col(position = "dodge")

# model predictions per position

d_prediction_per_position <- d |> 
  filter(condition == "production") |> 
  select(position_production, starts_with("prob_prod")) |> 
  rename(
    target = prob_production_target,
    competitor = prob_production_competitor,
    distractor1 = prob_production_distractor1,
    distractor2 = prob_production_distractor2
  ) |> 
  pivot_longer(-1, names_to = "response") |> 
  group_by(position_production, response) |> 
  summarize(probability = mean(value)) |> 
  ungroup() |> 
  mutate(response = factor(response, levels = c('target', 'competitor', 'distractor')))

d_prediction_per_position |> 
  ggplot(aes(x = position_production, y = probability, fill = response)) +
  geom_col(position = "dodge")

# plot human and model together

d_response_props_per_position |> 
  mutate(source = "human") |> 
  select(-n, -total) |> 
  rename(probability = proportion) |> 
  rbind(d_prediction_per_position |> mutate(source = "model")) |> 
  ggplot(aes(x = position_production, y = probability, fill = response)) +
  geom_col(position = "dodge") +
  facet_grid(source ~ .) +
  xlab("order of choice options")

# the picture looks not too dissimilar; 
# the remaining question is whether the model (with a different alpha) COULD possibly be 
# a good predictor of the human data;
# we need model criticism after all!?


#############################################################################
# RQ3a: Are human responses dependent on the order of presented alternatives?
#############################################################################

fit_position <- brm(
  formula = response ~ position_production,
  data = d |> filter(condition == "production"),
  family = categorical()
)

fit_intercept <- brm(
  formula = response ~ 1,
  data = d |> filter(condition == "production"),
  family = categorical()
) 

loo_compare(fit_position |> loo(), 
            fit_intercept |> loo())
# RESULT: intercept-only model is numerically worse, but not significantly so;
# leaving out the position argument does NOT seem to strongly weaken the predictions




# human response proportions per position

d$position_interpretation <- position_interpretation

d_response_props_per_position <- d |> 
  filter(condition == "interpretation") |> 
  count(position_interpretation, response) |> 
  group_by(position_interpretation) |> 
  mutate(total = sum(n),
         proportion = n/total) |> 
  ungroup()

d_response_props_per_position |> 
  ggplot(aes(x = position_interpretation, y = proportion, fill = response)) +
  geom_col(position = "dodge")

# model predictions per position

d_prediction_per_position <- d |> 
  filter(condition == "interpretation") |> 
  select(position_interpretation, starts_with("prob_interpre")) |> 
  rename(
    target = prob_interpretation_target,
    competitor = prob_interpretation_competitor,
    distractor = prob_interpretation_distractor
  ) |> 
  pivot_longer(-1, names_to = "response") |> 
  group_by(position_interpretation, response) |> 
  summarize(probability = mean(value)) |> 
  ungroup() |> 
  mutate(response = factor(response, levels = c('target', 'competitor', 'distractor')))

d_prediction_per_position |> 
  ggplot(aes(x = position_interpretation, y = probability, fill = response)) +
  geom_col(position = "dodge")

# plot human and model together

d_response_props_per_position |> 
  mutate(source = "human") |> 
  select(-n, -total) |> 
  rename(probability = proportion) |> 
  rbind(d_prediction_per_position |> mutate(source = "model")) |> 
  ggplot(aes(x = position_interpretation, y = probability, fill = response)) +
  geom_col(position = "dodge") +
  facet_grid(source ~ .) +
  xlab("order of choice options")

# the picture looks not too dissimilar; 
# the remaining question is whether the model (with a different alpha) COULD possibly be 
# a good predictor of the human data;
# we need model criticism after all!?


#############################################################################
# RQ3a: Are human responses dependent on the order of presented alternatives?
#############################################################################

fit_position <- brm(
  formula = response ~ position_interpretation,
  data = d |> filter(condition == "interpretation"),
  family = categorical()
)

fit_intercept <- brm(
  formula = response ~ 1,
  data = d |> filter(condition == "interpretation"),
  family = categorical()
) 

loo_compare(fit_position |> loo(), 
            fit_intercept |> loo())
# RESULT: intercept-only model is numerically worse, but not significantly so;
# leaving out the position argument does NOT seem to strongly weaken the predictions


# d |> filter(condition == "interpretation") |> 
#   count(position_interpretation, response) |> 
#   rename(dv = response) |> 
#   mutate(source = "human") |> 
#   rbind(d |> filter(condition == "interpretation") |> 
#           count(position_interpretation, prediction) |> 
#           rename(dv = prediction) |> 
#           mutate(source = "model")) |> 
#   ggplot(aes(x = position_interpretation, y = n, fill = dv)) +
#   facet_grid(source ~ .) +
#   geom_col(position="dodge")


















# ################################
# # interpretation
# ################################
# 
# # positional effects
# 
# d <- d |> 
#   mutate(interpretation_indices = str_c(
#     "[",
#     interpretation_index_target,
#     ",",
#     interpretation_index_distractor,
#     "]"
#   ))
# 
# table(d$interpretation_indices)
# 
# d |> filter(condition == "interpretation") |> 
#   group_by(interpretation_index_target, interpretation_index_competitor) |> 
#   tidyboot::tidyboot_mean(prob_interpretation_target)
# 
# d |> filter(condition == "interpretation") |> 
#   ggplot(aes(x = interpretation_index_target, y = prob_interpretation_target)) +
#   geom_jitter()
# 
# d |> filter(condition == "interpretation") |> 
#   group_by(interpretation_index_competitor) |> 
#   tidyboot::tidyboot_mean(prob_interpretation_competitor)
# 
# d |> filter(condition == "interpretation") |> 
#   ggplot(aes(x = interpretation_index_competitor, y = prob_interpretation_competitor)) +
#   geom_jitter()
# 
# d |> filter(condition == "interpretation") |> 
#   ggplot(aes(x = interpretation_indices, y = prob_interpretation_target)) +
#   geom_jitter(alpha = 0.5)
# 
# d |> filter(condition == "interpretation") |> 
#   count(interpretation_indices, prediction) |> 
#   ggplot(aes(x = interpretation_indices, y = n, fill = prediction)) +
#   geom_col(position="dodge")
#   
# d |> filter(condition == "interpretation") |>   
#   ggplot(aes(x = interpretation_indices, y = prob_interpretation_target)) +
#   geom_jitter(alpha = 0.5)
# 
# d |> filter(condition == "interpretation") |>
#   count(interpretation_indices, response)
# 
# d |> filter(condition == "interpretation") |> 
#   count(interpretation_indices, response) |> 
#   rename(dv = response) |> 
#   mutate(source = "human") |> 
#   rbind(d |> filter(condition == "interpretation") |> 
#           count(interpretation_indices, prediction) |> 
#           rename(dv = prediction) |> 
#           mutate(source = "model")) |> 
#   ggplot(aes(x = interpretation_indices, y = n, fill = dv)) +
#   facet_grid(source ~ .) +
#   geom_col(position="dodge")
# 
# 
# 
# 
