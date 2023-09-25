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
       filename = "../04-paper/00-pics/refgame-counts.pdf", 
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
       filename = "../04-paper/00-pics/model-predictions-vanilla.pdf", 
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
## fit data
#######################################################

fit_prod_narrow <- fit_data(d_prod_global, array(LLM_pred_prod_narrow, dim=c(1,1,3)))
fit_prod_intermediate <- fit_data(d_prod_global, array(LLM_pred_prod_intermediate, dim=c(1,1,3)))
fit_prod_wide <- fit_data(d_prod_global, array(LLM_pred_prod_wide, dim=c(1,nrow(LLM_pred_prod_wide),3)))

fit_inter_narrow <- fit_data(d_inter_global, array(LLM_pred_inter_narrow, dim=c(1,1,3)))
fit_inter_intermediate <- fit_data(d_inter_global, array(LLM_pred_inter_intermediate, dim=c(1,1,3)))
fit_inter_wide <- fit_data(d_inter_global, array(LLM_pred_inter_wide, dim=c(1,nrow(LLM_pred_inter_wide),3)))

#######################################################
## Bayesian stats
#######################################################

produce_summary_prodInt_epsilonAlpha(fit_prod_narrow, fit_inter_narrow)
produce_summary_prodInt_epsilonAlpha(fit_prod_intermediate, fit_inter_intermediate)
produce_summary_prodInt_epsilonAlpha(fit_prod_wide, fit_inter_wide)

#######################################################
## get posterior predictives
#######################################################

pp_prod_narrow <- get_posterior_predictives(fit_prod_narrow, d_prod_global, filename = "post_pred_prod_narrow") |> 
  mutate(condition = "production", model = "narrow")
pp_prod_intermediate <- get_posterior_predictives(fit_prod_intermediate, d_prod_global, filename = "post_pred_prod_intermediate") |> 
  mutate(condition = "production", model = "intermediate")
pp_prod_wide <- get_posterior_predictives(fit_prod_wide, d_prod_global, filename = "post_pred_prod_wide") |> 
  mutate(condition = "production", model = "wide")

pp_inter_narrow <- get_posterior_predictives(fit_inter_narrow, d_inter_global, filename = "post_pred_inter_narrow") |> 
  mutate(condition = "interpretation", model = "narrow")
pp_inter_intermediate <- get_posterior_predictives(fit_inter_intermediate, d_inter_global, filename = "post_pred_inter_intermediate") |> 
  mutate(condition = "interpretation", model = "intermediate")
pp_inter_wide <- get_posterior_predictives(fit_inter_wide, d_inter_global, filename = "post_pred_inter_wide") |> 
  mutate(condition = "interpretation", model = "wide")

PPC_data = rbind(
  pp_prod_narrow,
  pp_prod_intermediate,
  pp_prod_wide,
  pp_inter_narrow,
  pp_inter_intermediate,
  pp_inter_wide
) |> select(-row) |> 
  mutate(
    condition = factor(condition, levels = c("production", "interpretation")),
    model     = factor(model, levels = c("narrow", "wide", "intermediate"))
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

ggsave(filename = "../03-paper/00-pics/PPC-alpha-eps-model.pdf", width = 8, height = 3.5, scale = 0.9)

#######################################################
## Bayesian p-values
#######################################################

tibble(
  condition = rep(c("production", "interpretation"), each = 3),
  model = rep(c("narrow", "intermediate", "wide"), 2),
  Bppp_value = c(extract_bayesian_p(fit_prod_narrow),
                 extract_bayesian_p(fit_prod_intermediate),
                 extract_bayesian_p(fit_prod_wide),
                 extract_bayesian_p(fit_inter_narrow),
                 extract_bayesian_p(fit_inter_intermediate),
                 extract_bayesian_p(fit_inter_wide))  
)





#####################################################################################















