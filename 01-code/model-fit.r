library(tidyverse)
library(tidyboot)
library(brms)
library(cmdstanr)

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


#################################################
# CmdStanR example
#################################################

check_cmdstan_toolchain()
cmdstan_path()

file <- file.path(cmdstan_path(), "examples", "bernoulli", "bernoulli.stan")
mod <- cmdstan_model(file)
mod$print()

mod$exe_file()

# names correspond to the data block in the Stan program
data_list <- list(N = 10, y = c(0,1,0,0,0,0,0,0,0,1))

fit <- mod$sample(
  data = data_list,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  refresh = 500 # print update every 500 iters
)

posterior_samples <- fit$draws(format = "df")


#################################################
# Estimating alpha for simulation data
#################################################

# prepare data: interpretation

k_inter <- c(115+117, 65+62, 1)
N_inter <- sum(k_inter)
scores_inter <- d |> 
  select(starts_with("scores_interpretation_")) |> 
  as.matrix.data.frame()
props_inter <- k_inter / N_inter
n_options_inter = dim(scores_inter)[2]
n_scores_inter = dim(scores_inter)[1]

data_list_inter <- list(
  N = N_inter,
  n_options = n_options_inter,
  n_scores = n_scores_inter,
  k = k_inter,
  scores = scores_inter
)

# prepare data: production

k_prod <- c(135 + 119, 9 + 25, 0, 0)
N_prod <- sum(k_prod)
scores_prod <- d |> 
  select(starts_with("scores_production_")) |> 
  as.matrix.data.frame()
props_prod <- k_prod / N_prod
n_options_prod = dim(scores_prod)[2]
n_scores_prod = dim(scores_prod)[1]

data_list_prod <- list(
  N = N_prod,
  n_options = n_options_prod,
  n_scores = n_scores_prod,
  k = k_prod,
  scores = scores_prod
)


# fitting

data_list <- data_list_inter
data_list <- data_list_prod


mod <- cmdstan_model('multinomial-interpretation.stan')
mod$print()
 
fit_mle <- mod$optimize(data = data_list)
print(fit_mle)

fit <- mod$sample(
  data = data_list,
  seed = 123,
  adapt_delta = 0.99,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 2000
  # refresh = 500 # print update every 500 iters
)

fit$summary()
posterior_samples <- fit$draws(format = "df") |> 
  select(alpha)

posterior_samples |> 
  ggplot(aes(x = alpha)) +
  geom_density()

# for interpretation: there is actually no need to transform log-probabilities!
# alpha = 1 is credible, so we could use the raw probabilities (after normalization) as well!
aida::summarize_sample_vector(posterior_samples$alpha)
  
# TODO: why does the posterior for alpha have such a long tail? why is the posterior mean so high?



