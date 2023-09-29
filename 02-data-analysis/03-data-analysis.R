# dependencies etc.
source("00-premable.r")
rerun = TRUE
source('00-stan-fit-helpers.R')

#######################################################
## read and prep data, get counts
#######################################################

tcd <- c("target", "competitor", "distractor")

d <- read_csv('02-data-prepped.csv') |> 
  mutate(
    response   = factor(response  , levels = tcd),
    prediction = factor(prediction, levels = tcd),
    condition  = factor(condition,  levels = c("production", "interpretation"))
  ) 

d_counts <- d |> 
  count(condition, response) |> 
  group_by(condition) |> 
  mutate(total = sum(n)) |> 
  ungroup() |> 
  mutate(proportion = n / total)

#######################################################
## plot empirical counts
#######################################################

refgame_counts <- d_counts |> 
  ggplot(aes(x = response, y = n, fill = response)) +
  geom_col(position = "dodge") +
  facet_grid(condition ~ .) +
  ylab("") + xlab("") +
  theme(legend.position="none") +
  ylim(c(0,604)) +
  ggtitle("human data") +
  theme(axis.text.x = element_text(angle = 25, vjust = 1, hjust=1)) +
  theme(strip.text = element_text(size = 12)) + 
  theme(plot.title = element_text(size = 12))

refgame_counts

ggsave(plot = refgame_counts, 
       filename = "../03-paper/00-pics/refgame-counts.pdf", 
       height = 4, width = 3, scale = 1.1)

#######################################################
## prepare model predictions (-> move to prep script?)
#######################################################

softmax_row <- function(matrix, alpha = 1) {
  exp_matrix <- exp(alpha * matrix)
  row_sums <- rowSums(exp_matrix)
  softmax_matrix <- exp_matrix / row_sums
  return(softmax_matrix)
}

#################
# production
#################

# wrangle data
x <- d |> 
  filter(condition == "production") |> 
  select(submission_id, trial_nr, item, condition, starts_with("scores_produ")) |> 
  mutate(scores_production_distractor = 
           log(exp(scores_production_distractor1) + exp(scores_production_distractor2))) |> 
  select(-scores_production_distractor1, -scores_production_distractor2)

# narrow- and intermediate-scope predictions for production
matrix_global <- x |> 
  select(starts_with("scores_produ")) |> as.matrix()
LLM_pred_prod_narrow       <- softmax_row(matrix(apply(matrix_global, 2, mean), nrow = 1))
LLM_pred_prod_intermediate <- matrix(apply(softmax_row(matrix_global), 2, mean), nrow = 1)

# wide-scope predictions
matrix_itemLevel <- x |> 
  select(item, starts_with("scores_produ")) |> 
  unique() |> 
  arrange(item) |> 
  select(starts_with("scores_produ")) |> as.matrix()
LLM_pred_prod_wide  <- softmax_row(matrix_itemLevel)

#################
# interpretation
#################

# wrangle data
x <- d |> 
  filter(condition == "interpretation") |> 
  select(submission_id, trial_nr, item, condition, starts_with("scores_inter"))

# narrow- and intermediate-scope predictions for production
matrix_global <- x |> 
  select(starts_with("scores_inter")) |> as.matrix()
LLM_pred_inter_narrow       <- softmax_row(matrix(apply(matrix_global, 2, mean), nrow = 1))
LLM_pred_inter_intermediate <- matrix(apply(softmax_row(matrix_global), 2, mean), nrow = 1)

# wide-scope predictions
matrix_itemLevel <- x |> 
  select(item, starts_with("scores_inter")) |> 
  unique() |> 
  arrange(item) |> 
  select(starts_with("scores_inter")) |> as.matrix()
LLM_pred_inter_wide  <- softmax_row(matrix_itemLevel)

######################################################
## plot model predictions for (alpha=1)
## NB: w/ alpha=1, wide- and intermediate-scope
##     predictions are identical
######################################################

d_llm_prob_averages <- 
  tibble(condition = factor(c(rep("production",6), rep("interpretation",6)),
                          levels = c("production", "interpretation")),
       model = factor(rep(c(rep(c("narrow"),3), rep(c("intermediate/wide"),3)),2), 
                      levels = c("narrow", "intermediate/wide")),
       response = factor(rep(tcd,4), levels = tcd),
       prob = c(LLM_pred_prod_narrow[1,],
                LLM_pred_prod_intermediate[1,],
                LLM_pred_inter_narrow[1,],
                LLM_pred_inter_intermediate[1,]))

model_predictions_vanilla <- d_llm_prob_averages |> 
  ggplot(aes(x = response, y = prob, fill = response)) +
  geom_col(position = "dodge") +
  facet_grid(condition ~ model) +
  ylab("") + xlab("") +
  theme(legend.position="none") +
  ylim(c(0,1)) +
  ggtitle("model predictions") +
  theme(axis.text.x = element_text(angle = 25, vjust = 1, hjust=1)) +
  theme(strip.text = element_text(size = 12)) + 
  theme(plot.title = element_text(size = 12))

model_predictions_vanilla

ggsave(plot = model_predictions_vanilla, 
       filename = "../03-paper/00-pics/model-predictions-vanilla.pdf", 
       height = 4, width = 6, scale = 1.1)

#######################################################
## prepare data
#######################################################

# production - global 
d_production <- d |> 
  filter(condition == "production") |> 
  mutate(response = case_when(response == "distractor1" ~ "distractor",
                              response == "distractor2" ~ "distractor",
                              TRUE ~ response)) |> 
  pull(response) |> 
  match(c("target", "competitor", "distractor"))
d_prod_global <- matrix(table(d_production) |> as.integer(), nrow = 1)

# production - item-level
d_prod_item <- d |> 
  filter(condition == "production") |> 
  mutate(response = case_when(response == "distractor1" ~ "distractor",
                              response == "distractor2" ~ "distractor",
                              TRUE ~ response)) |> 
  group_by(item) |> 
  count(response) |> 
  pivot_wider(id_cols = item, names_from = response, values_from = n) |> 
  ungroup() |> 
  arrange(item) |> 
  select(-item) |> 
  as.matrix()
d_prod_item[is.na(d_prod_item)] = 0
d_prod_item <- d_prod_item[,c(3,1,2)]

# production - global 
d_interpretation <- d |> 
  filter(condition == "interpretation") |> 
  pull(response) |> 
  match(c("target", "competitor", "distractor"))
d_inter_global <- matrix(table(d_interpretation) |> as.integer(), nrow = 1)

# interpretation - item-level
d_inter_item <- d |> 
  filter(condition == "interpretation") |> 
  group_by(item) |> 
  count(response) |> 
  pivot_wider(id_cols = item, names_from = response, values_from = n) |> 
  ungroup() |> 
  arrange(item) |> 
  select(-item) |> 
  as.matrix()
d_inter_item[is.na(d_inter_item)] = 0

#######################################################
## RSA model predictions (Vanilla, alpha = 0)
#######################################################

RSA_pred_prod  <- array(c(2/3, 1/3, 0), dim=c(1,1,3))
RSA_pred_inter <- array(c(0.6, 0.4, 0), dim=c(1,1,3))

#######################################################
## fit data
#######################################################

fit_prod_narrow <- fit_data(d_prod_global, array(LLM_pred_prod_narrow, dim=c(1,1,3)))
fit_prod_intermediate <- fit_data(d_prod_global, array(LLM_pred_prod_intermediate, dim=c(1,1,3)))
fit_prod_wide <- fit_data(d_prod_global, array(LLM_pred_prod_wide, dim=c(1,nrow(LLM_pred_prod_wide),3)))
fit_prod_RSA  <- fit_data(d_prod_global, RSA_pred_prod)

fit_inter_narrow <- fit_data(d_inter_global, array(LLM_pred_inter_narrow, dim=c(1,1,3)))
fit_inter_intermediate <- fit_data(d_inter_global, array(LLM_pred_inter_intermediate, dim=c(1,1,3)))
fit_inter_wide <- fit_data(d_inter_global, array(LLM_pred_inter_wide, dim=c(1,nrow(LLM_pred_inter_wide),3)))
fit_inter_RSA  <- fit_data(d_inter_global, RSA_pred_inter)

#######################################################
## Bayesian stats
#######################################################

posterior_stats <- rbind(
  produce_summary_prodInt_epsilonAlpha(fit_prod_narrow, fit_inter_narrow) |> 
    mutate(model = "narrow"),
  produce_summary_prodInt_epsilonAlpha(fit_prod_intermediate, fit_inter_intermediate) |> 
    mutate(model = "intermediate"),
  produce_summary_prodInt_epsilonAlpha(fit_prod_wide, fit_inter_wide)  |> 
    mutate(model = "wide"),
  produce_summary_prodInt_epsilonAlpha(fit_prod_RSA, fit_inter_RSA)  |> 
    mutate(model = "RSA")
) |>  mutate(
  condition = factor(condition, levels = c("production", "interpretation")),
  model     = factor(model, levels = rev(c("narrow", "wide", "intermediate", "RSA")))
)

# plot
posterior_stats |> 
  filter(condition != "diff. prod-inter") |> 
  ggplot() +
  geom_linerange(aes(x = model, y = mean, ymin = `|95%`, ymax = `95%|`), color = "gray") +
  geom_point(aes(x = model, y = mean, ymin = `|95%`, ymax = `95%|`)) +
  facet_grid(condition ~ Parameter, scales = "free") +
  coord_flip()

ggsave(filename = "../03-paper/00-pics/posterior-stats.pdf", width = 8, height = 4, scale = 1.0)

# summary stats as table

posterior_stats |> 
  filter(!is.na(condition)) |> 
  select(c(6,5,1,2,3,4)) |> 
  xtable::xtable()
  

#######################################################
## get posterior predictives
#######################################################

pp_prod_narrow <- get_posterior_predictives(fit_prod_narrow, d_prod_global, filename = "post_pred_prod_narrow") |> 
  mutate(condition = "production", model = "narrow")
pp_prod_intermediate <- get_posterior_predictives(fit_prod_intermediate, d_prod_global, filename = "post_pred_prod_intermediate") |> 
  mutate(condition = "production", model = "intermediate")
pp_prod_wide <- get_posterior_predictives(fit_prod_wide, d_prod_global, filename = "post_pred_prod_wide") |> 
  mutate(condition = "production", model = "wide")
pp_prod_RSA <- get_posterior_predictives(fit_prod_RSA, d_prod_global, filename = "post_pred_prod_RSA") |> 
  mutate(condition = "production", model = "RSA")


pp_inter_narrow <- get_posterior_predictives(fit_inter_narrow, d_inter_global, filename = "post_pred_inter_narrow") |> 
  mutate(condition = "interpretation", model = "narrow")
pp_inter_intermediate <- get_posterior_predictives(fit_inter_intermediate, d_inter_global, filename = "post_pred_inter_intermediate") |> 
  mutate(condition = "interpretation", model = "intermediate")
pp_inter_wide <- get_posterior_predictives(fit_inter_wide, d_inter_global, filename = "post_pred_inter_wide") |> 
  mutate(condition = "interpretation", model = "wide")
pp_inter_RSA <- get_posterior_predictives(fit_inter_RSA, d_inter_global, filename = "post_pred_inter_RSA") |> 
  mutate(condition = "interpretation", model = "RSA")


PPC_data = rbind(
  pp_prod_narrow,
  pp_prod_intermediate,
  pp_prod_wide,
  pp_prod_RSA,
  pp_inter_narrow,
  pp_inter_intermediate,
  pp_inter_wide,
  pp_inter_RSA
) |> select(-row) |> 
  mutate(
    condition = factor(condition, levels = c("production", "interpretation")),
    model     = factor(model, levels = c("narrow", "wide", "intermediate", "RSA"))
    )

#######################################################
## make posterior predictives plots
#######################################################

plot_PPC <- function(PP_prod, PP_inter, name = "bla"){
  PP_prod <- PP_prod |> 
    mutate(condition = "production")
  PP_inter <- PP_inter |> 
    mutate(condition = "interpretation")
  
  rbind(PP_prod, PP_inter) |> 
    mutate(condition = factor(condition, levels = c("production", "interpretation"))) |> 
    ggplot(aes(x = response, y = observed, fill = response)) +
    geom_col() +
    facet_grid(.~condition) +
    geom_pointrange(aes(x = response, y = mean, ymin = `|95%`, ymax = `95%|`), size = 0.7, linewidth = 1) +
    ylab("") + xlab("") +
    theme(legend.position="none") +
    theme(axis.text.x = element_text(angle = 25, vjust = 1, hjust=1)) +
    theme(strip.text = element_text(size = 12))
  
  # ggsave(filename = "../04-paper/00-pics/PPC-alpha-eps-model.pdf", width = 8, height = 3.5, scale = 0.9)

}

# individual plots
plot_PPC(pp_prod_narrow, pp_inter_narrow)
plot_PPC(pp_prod_intermediate, pp_inter_intermediate)
plot_PPC(pp_prod_wide, pp_inter_wide)
plot_PPC(pp_prod_RSA, pp_inter_RSA)

# all models in one

PPC_data |> 
  ggplot() +
  geom_col(data = PPC_data |> filter(model == "narrow"),
           aes(x = response, y = observed, fill = response)) +
  facet_grid(.~condition) +
  geom_pointrange(aes(x = response, y = mean, ymin = `|95%`, ymax = `95%|`, shape = model, group = model), 
                  position = position_dodge(width = 0.5), size = 0.7, linewidth = 1, color = "black") +
  ylab("") + xlab("") +
  # theme(legend.position="none") +
  theme(axis.text.x = element_text(angle = 25, vjust = 1, hjust=1)) +
  theme(strip.text = element_text(size = 12)) +
  guides(
    shape = guide_legend(
      direction = "vertical",
      title = "model",
      override.aes = list(
        # shape = c("narrow" = 1, "wide" = 2, "intermediate" = 3),
        fill = "black"
      )
    ),
    fill = "none"
  ) +
  theme(legend.position = "right")

ggsave(filename = "../03-paper/00-pics/PPC-alpha-eps-model.pdf", width = 8, height = 3.5, scale = 1.0)

#######################################################
## Bayesian p-values
#######################################################

tibble(
  condition = rep(c("production", "interpretation"), each = 4),
  model = rep(c("narrow", "wide", "intermediate", "RSA"), 2),
  Bppp_value = c(extract_bayesian_p(fit_prod_narrow),
                 extract_bayesian_p(fit_prod_wide),
                 extract_bayesian_p(fit_prod_intermediate),
                 extract_bayesian_p(fit_prod_RSA),
                 extract_bayesian_p(fit_inter_narrow),
                 extract_bayesian_p(fit_inter_wide),
                 extract_bayesian_p(fit_inter_intermediate),
                 extract_bayesian_p(fit_inter_RSA))  
) |> 
  pivot_wider(id_cols = model, names_from = condition, values_from = Bppp_value) |> 
  xtable::xtable()

#######################################################
## item-level analysis (LLM)
#######################################################

# prepare data (item-level)

d_llm_prob_prod <- d |> 
  filter(condition == "production") |> 
  select(item, starts_with("scores_produ")) |> 
  mutate(scores_production_distractor = 
           log(exp(scores_production_distractor1) + exp(scores_production_distractor2))) |> 
  select(-scores_production_distractor1, -scores_production_distractor2) |> 
  group_by(item) |> 
  mutate(n = n()) |>  
  pivot_longer(-c("item", "n")) |> 
  separate(name, into = c("Variable", "condition", "response"), sep = "_") |> 
  select(-Variable) |> 
  mutate(prob = exp(value)) |>
  group_by(item, condition) |> 
  mutate(prob = prob / sum(prob) * n) |> 
  ungroup()

d_llm_prob_inter <- d |> 
  filter(condition == "interpretation") |> 
  select(item, starts_with("scores_inter")) |> 
  group_by(item) |> 
  mutate(n = n()) |>  
  pivot_longer(-c("item", "n")) |> 
  separate(name, into = c("Variable", "condition", "response"), sep = "_") |> 
  select(-Variable) |> 
  mutate(prob = exp(value)) |>
  group_by(item, condition) |> 
  mutate(prob = prob / sum(prob) * n) |> 
  ungroup() 

d_item_analysis <- d_llm_prob_prod |>
  rbind(d_llm_prob_inter) |>
  mutate(condition = factor(condition, levels = c('production', 'interpretation'))) |>
  full_join(d |>
              select(item, feature_set, position_production, position_interpretation) |>
              unique(),
            by = 'item') |>
  mutate(position = case_when(condition == "production" ~ position_production,
                              TRUE ~ position_interpretation)) |>
  # filter(response != "distractor") |>
  unique() |>
  pivot_wider(id_cols = c(item, condition, feature_set, position_production, position_interpretation, position),
              names_from = response, values_from = prob) |> 
  arrange(condition, item)

# prepare data (item-level, production)

d_counts_items <- d |> 
  count(condition, item, response) |> 
  group_by(condition, item) |> 
  mutate(total = sum(n)) |> 
  ungroup() |> 
  mutate(proportion = n / total) |> 
  arrange(condition, item, response)

data_items_prod <- d_counts_items |> 
  filter(condition == "production") |> 
  pivot_wider(id_cols = item, names_from = response, values_from = n, values_fill = 0) |> 
  select(-item) |> 
  as.matrix()

pred_items_prod <- d_item_analysis |>
  filter(condition == "production") |> 
  group_by(item) |> 
  summarize(
    target = mean(target),
    competitor = mean(competitor),
    distractor = mean(distractor)
  ) |> 
  ungroup() |> 
  select(-item) |> 
  as.matrix()

# prepare data (item-level, interpretation)

data_items_inter <- d_counts_items |> 
  filter(condition == "interpretation") |> 
  pivot_wider(id_cols = item, names_from = response, values_from = n, values_fill = 0) |> 
  select(-item) |> 
  as.matrix()

pred_items_inter <- d_item_analysis |>
  filter(condition == "interpretation") |> 
  group_by(item) |> 
  summarize(
    target = mean(target),
    competitor = mean(competitor),
    distractor = mean(distractor)
  ) |> 
  ungroup() |> 
  select(-item) |> 
  as.matrix()

# fit models

fit_items_prod <- fit_data(
  data_items_prod, 
  array(pred_items_prod, dim = c(nrow(pred_items_prod),1, 3)), 
  model_name = '00-stan-files/llm-average-matrix-epsilon-arrayed.stan')

fit_items_inter <- fit_data(
  data_items_inter, 
  array(pred_items_inter, dim = c(nrow(pred_items_inter),1, 3)),
  model_name = '00-stan-files/llm-average-matrix-epsilon-arrayed.stan')

# summary stats

produce_summary_prodInt_epsilonAlpha(fit_items_prod, fit_items_inter)

# PPC visualization

# production
post_pred_items_prod <- 
  get_posterior_predictives(
    fit_items_prod,
    data_items_prod,
    "post-pred-prod-item"
  ) |> 
  group_by(row) |> 
  mutate(total = sum(observed)) |> 
  filter(response == "target") |> 
  mutate(row = factor(row))

post_pred_items_prod |> 
  ggplot(aes(x = fct_reorder(row, mean/total), y = observed/total)) +
  geom_pointrange(aes(y = mean/total, ymin = `|95%`/total, ymax = `95%|`/total), 
                  size = 0.6, linewidth = 1, color = "lightgray") +
  geom_point(aes(y = mean/total), color = "darkgray") +
  geom_point(color = project_colors[2]) +
  coord_flip() +
  xlab("item") +
  ylab("proportion of target") +
  theme(axis.text.y = element_blank(),
        axis.ticks.y = element_blank()) +
  ggtitle("production")

ggsave(filename = "../03-paper/00-pics/item-prod-postPred.pdf", width = 5, height = 4, scale = 1.0)

post_pred_items_prod |> 
  ggplot(aes(x = mean/total, observed / total)) +
  geom_segment((aes(x = 0, y = 0, xend = 1, yend=1)), color = "gray") +
  geom_point(alpha = 0.8) +
  xlim(c(0,1)) +
  ylim(c(0,1)) + 
  ylab("observed") +
  xlab("predicted") +
  ggtitle("production")

ggsave(filename = "../03-paper/00-pics/item-prod-obs-pred.pdf", width = 5, height = 4, scale = 1.0)

# interpretation
post_pred_items_inter <- 
  get_posterior_predictives(
    fit_items_inter,
    data_items_inter,
    "post-pred-inter-item"
  ) |> 
  group_by(row) |> 
  mutate(total = sum(observed)) |> 
  filter(response == "target") |> 
  mutate(row = factor(row))

post_pred_items_inter |> 
  ggplot(aes(x = fct_reorder(row, mean/total), y = observed/total)) +
  geom_pointrange(aes(y = mean/total, ymin = `|95%`/total, ymax = `95%|`/total), 
                  size = 0.6, linewidth = 1, color = "lightgray") +
  geom_point(aes(y = mean/total), color = "darkgray") +
  geom_point(color = project_colors[2]) +
  coord_flip() +
  xlab("item") +
  ylab("proportion of target") +
  theme(axis.text.y = element_blank(),
        axis.ticks.y = element_blank()) +
  ggtitle("interpretation")

ggsave(filename = "../03-paper/00-pics/item-inter-postPred.pdf", width = 5, height = 4, scale = 1.0)

post_pred_items_inter |> 
  ggplot(aes(x = mean/total, y =  observed/ total)) +
  # geom_segment(aes(x = `|95%`/total, xend = `95%|`/total, yend = observed/total), color = "gray", alpha = 0.7) +
  geom_segment((aes(x = 0, y = 0, xend = 1, yend=1)), color = "gray") +
  geom_point(alpha = 0.8) +
  xlim(c(0,1)) +
  ylim(c(0,1)) + 
  ylab("observed") +
  xlab("predicted") +
  ggtitle("interpretation")

ggsave(filename = "../03-paper/00-pics/item-inter-obs-pred.pdf", width = 5, height = 4, scale = 1.0)

rbind(
  post_pred_items_prod |> mutate(condition = "production"),
  post_pred_items_inter |> mutate(condition = "interpretation")
  ) |>
  mutate(condition = factor(condition, levels = c("production", "interpretation"))) |> 
  ggplot(aes(x = mean/total, y =  observed/ total)) +
  geom_segment((aes(x = 0, y = 0, xend = 1, yend=1)), color = "gray") +
  geom_point(alpha = 0.8) +
  facet_grid(. ~ condition) +
  xlim(c(0,1)) +
  ylim(c(0,1)) + 
  ylab("observed proportion of target") +
  xlab("posterior mean of predicted probability for target") 

ggsave(filename = "../03-paper/00-pics/item-combined-obs-pred.pdf", width = 9, height = 4, scale = 1.0)


# Bayesian posterior predictive p-values

message("Bayesian p value for production (LLM, by-item):", extract_bayesian_p(fit_items_prod))
message("Bayesian p value for interpretation (LLM, by-item):", extract_bayesian_p(fit_items_inter))


#######################################################
## item-level analysis (RSA)
#######################################################

fit_items_prod_RSA <- fit_data(
  data_items_prod, 
  array(rep(c(2/3,1/3,0), each = nrow(data_items_prod)), dim = c(nrow(data_items_prod),1, 3)), 
  model_name = '00-stan-files/llm-average-matrix-epsilon-arrayed.stan')

fit_items_inter_RSA <- fit_data(
  data_items_inter, 
  array(rep(c(0.6,0.4,0), each = nrow(data_items_inter)), dim = c(nrow(data_items_inter),1, 3)),
  model_name = '00-stan-files/llm-average-matrix-epsilon-arrayed.stan')

# Bayesian posterior predictive p-values

message("Bayesian p value for production (RSA, by-item):", extract_bayesian_p(fit_items_prod_RSA))
message("Bayesian p value for interpretation (RSA, by-item):", extract_bayesian_p(fit_items_prod_RSA))
