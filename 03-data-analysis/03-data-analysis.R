# dependencies etc.
source("00-premable.r")
rerun = T
source('00-stan-fit-helpers.R')

#######################################################
## read and prep data, get counts, massage human data
#######################################################

tcd <- c("target", "competitor", "distractor")

d <- read_csv('02-prepped-data/data-prepped-GPT.csv') |> 
    mutate(
      response   = factor(response  , levels = tcd),
      prediction = factor(prediction, levels = tcd),
      condition  = factor(condition,  levels = c("production", "interpretation"))
    )  

# raw counts
d_counts <- d |> 
  count(condition, response) |> 
  group_by(condition) |> 
  mutate(total = sum(n)) |> 
  ungroup() |> 
  mutate(proportion = n / total)


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

####### item-level data

# item-level counts
d_counts_items <- d |> 
  count(condition, item, response) |> 
  group_by(condition, item) |> 
  mutate(total = sum(n)) |> 
  ungroup() |> 
  mutate(proportion = n / total) |> 
  arrange(condition, item, response)

# prepare data (item-level, production)
data_items_prod <- d_counts_items |> 
  filter(condition == "production") |> 
  pivot_wider(id_cols = item, names_from = response, values_from = n, values_fill = 0) |> 
  select(-item) |> 
  as.matrix()

# prepare data (item-level, interpretation)
data_items_inter <- d_counts_items |> 
  filter(condition == "interpretation") |> 
  pivot_wider(id_cols = item, names_from = response, values_from = n, values_fill = 0) |> 
  select(-item) |> 
  as.matrix()

#######################################################
## fit models
#######################################################

get_model_fits_RSA <- function(rerun = FALSE) {
  
  model_path = str_c("03-model-fits/model_fits-RSA.rds")
  
  if (rerun == FALSE) {
    return(read_rds(model_path))
  }
  ## RSA model predictions (Vanilla, alpha = 0)
  RSA_pred_prod_raw  <- c(2/3,1/3,0) + 1e-7 / sum(c(2/3,1/3,0) + 1e-7)
  RSA_pred_inter_raw <- c(0.6,0.4,0) + 1e-7 / sum(c(0.6,0.4,0) + 1e-7)
  RSA_pred_prod  <- array(RSA_pred_prod_raw,  dim=c(1,1,3))
  RSA_pred_inter <- array(RSA_pred_inter_raw, dim=c(1,1,3))
  
  # condition-level fits
  fit_prod_RSA  <- fit_data(d_prod_global, RSA_pred_prod)
  fit_inter_RSA  <- fit_data(d_inter_global, RSA_pred_inter)
  
  # posterior-predictives (condition-level)
  pp_prod_RSA <- get_posterior_predictives(fit_prod_RSA, d_prod_global, filename = "post_pred_prod_RSA") |> 
    mutate(condition = "production", model = "RSA")
  pp_inter_RSA <- get_posterior_predictives(fit_inter_RSA, d_inter_global, filename = "post_pred_inter_RSA") |> 
    mutate(condition = "interpretation", model = "RSA")
 
  # item-level fits  
  fit_items_prod_RSA <- fit_data(
    data_items_prod, 
    array(rep(RSA_pred_prod_raw, each = nrow(data_items_prod)), dim = c(nrow(data_items_prod),1, 3)), 
    model_name = '00-stan-files/llm-average-matrix-epsilon-arrayed.stan')
  fit_items_inter_RSA <- fit_data(
    data_items_inter, 
    array(rep(RSA_pred_inter_raw, each = nrow(data_items_inter)), dim = c(nrow(data_items_inter),1, 3)),
    model_name = '00-stan-files/llm-average-matrix-epsilon-arrayed.stan')
  
  model_fits <- list(
    fit_prod_RSA          = fit_prod_RSA,
    fit_inter_RSA         = fit_inter_RSA,
    pp_prod_RSA = pp_prod_RSA,
    pp_inter_RSA = pp_inter_RSA,
    fit_items_prod_RSA = fit_items_prod_RSA,
    fit_items_inter_RSA = fit_items_inter_RSA
  )
  
  write_rds(model_fits, model_path)
  
  return(model_fits)
}

get_model_fits <- function(model_name, rerun = FALSE) {
  
  model_path = str_c("03-model-fits/model_fits-", model_name, ".rds")
  
  if (rerun == FALSE) {
    return(read_rds(model_path))
  }
    
  ######################################################
  ## read LLM predictions
  ######################################################
  
  model_predictions <- readRDS(str_c("02-prepped-data/predictions-", model_name, ".rds"))
  
  LLM_pred_prod_avg_scores  <- model_predictions$LLM_pred_prod_avg_scores
  LLM_pred_prod_avg_probs   <- model_predictions$LLM_pred_prod_avg_probs
  LLM_pred_prod_WTA         <- model_predictions$LLM_pred_prod_WTA
  LLM_pred_inter_avg_scores <- model_predictions$LLM_pred_inter_avg_scores
  LLM_pred_inter_avg_probs  <- model_predictions$LLM_pred_inter_avg_probs
  LLM_pred_inter_WTA        <- model_predictions$LLM_pred_inter_WTA
  pred_items_prod           <- model_predictions$pred_items_prod
  pred_items_inter          <- model_predictions$pred_items_inter
  
  #######################################################
  ## fit data
  #######################################################
  
  # condition-level
  fit_prod_avg_scores  <- fit_data(d_prod_global, array(LLM_pred_prod_avg_scores, dim=c(1,1,3)))
  fit_prod_WTA         <- fit_data(d_prod_global, array(LLM_pred_prod_WTA, dim=c(1,1,3)))
  fit_prod_avg_probs   <- fit_data(d_prod_global, array(LLM_pred_prod_avg_probs, dim=c(1,nrow(LLM_pred_prod_avg_probs),3)))
  fit_inter_avg_scores <- fit_data(d_inter_global, array(LLM_pred_inter_avg_scores, dim=c(1,1,3)))
  fit_inter_WTA        <- fit_data(d_inter_global, array(LLM_pred_inter_WTA, dim=c(1,1,3)))
  fit_inter_avg_probs  <- fit_data(d_inter_global, array(LLM_pred_inter_avg_probs, dim=c(1,nrow(LLM_pred_inter_avg_probs),3)))
  
  # item-level data
  fit_items_prod <- fit_data(
    data_items_prod, 
    array(pred_items_prod, dim = c(nrow(pred_items_prod),1, 3)), 
    model_name = '00-stan-files/llm-average-matrix-epsilon-arrayed.stan')
  fit_items_inter <- fit_data(
    data_items_inter, 
    array(pred_items_inter, dim = c(nrow(pred_items_inter),1, 3)),
    model_name = '00-stan-files/llm-average-matrix-epsilon-arrayed.stan')
  
  #####################################################
  ## get posterior predictives
  #####################################################
  
  # condition-level
  pp_prod_avg_scores  <- get_posterior_predictives(fit_prod_avg_scores, d_prod_global, filename = "post_pred_prod_avg_scores") |>
    mutate(condition = "production", model = "avg. scores")
  pp_prod_avg_probs   <- get_posterior_predictives(fit_prod_avg_probs, d_prod_global, filename = "post_pred_prod_avg_probs") |>
    mutate(condition = "production", model = "avg. probabilities")
  pp_prod_WTA         <- get_posterior_predictives(fit_prod_WTA, d_prod_global, filename = "post_pred_prod_WTA") |>
    mutate(condition = "production", model = "avg. WTA")
  pp_inter_avg_scores <- get_posterior_predictives(fit_inter_avg_scores, d_inter_global, filename = "post_pred_inter_avg_scores") |>
    mutate(condition = "interpretation", model = "avg. scores")
  pp_inter_avg_probs  <- get_posterior_predictives(fit_inter_avg_probs, d_inter_global, filename = "post_pred_inter_avg_probs") |>
    mutate(condition = "interpretation", model = "avg. probabilities")
  pp_inter_WTA        <- get_posterior_predictives(fit_inter_WTA, d_inter_global, filename = "post_pred_inter_WTA") |>
    mutate(condition = "interpretation", model = "avg. WTA")
  
  # item-level
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
  
  #####################################################
  ## prepare and save fit object
  #####################################################
  
  model_fits <- list(
    fit_prod_avg_scores   = fit_prod_avg_scores,
    fit_prod_WTA          = fit_prod_WTA,
    fit_prod_avg_probs    = fit_prod_avg_probs,
    fit_inter_avg_scores  = fit_inter_avg_scores,
    fit_inter_WTA         = fit_inter_WTA,
    fit_inter_avg_probs   = fit_inter_avg_probs,
    fit_items_prod        = fit_items_prod,
    fit_items_inter       = fit_items_inter,
    pp_prod_avg_scores    = pp_prod_avg_scores,
    pp_prod_WTA           = pp_prod_WTA,
    pp_prod_avg_probs     = pp_prod_avg_probs,
    pp_inter_avg_scores   = pp_inter_avg_scores,
    pp_inter_WTA          = pp_inter_WTA,
    pp_inter_avg_probs    = pp_inter_avg_probs,
    post_pred_items_prod  = post_pred_items_prod,
    post_pred_items_inter = post_pred_items_inter
  )
  
  # save model fits
  write_rds(model_fits, model_path)
  
  return(model_fits)
}

model_names = c(
  "LLaMA2-chat-hf-70b", 
  "LLaMA2-chat-hf-13b", 
  "LLaMA2-chat-hf-7b", 
  "LLaMA2-hf-70b",
  "LLaMA2-hf-13b",
  "LLaMA2-hf-7b",
  "GPT"
)

model_fits_RSA <- get_model_fits_RSA(rerun = TRUE)
attach(model_fits_RSA)

model_fits = get_model_fits(model_names[4], rerun = TRUE)
attach(model_fits)



#######################################################
## fit data
#######################################################

# fit_prod_avg_scores <- fit_data(d_prod_global, array(LLM_pred_prod_avg_scores, dim=c(1,1,3)))
# fit_prod_WTA <- fit_data(d_prod_global, array(LLM_pred_prod_WTA, dim=c(1,1,3)))
# fit_prod_avg_probs <- fit_data(d_prod_global, array(LLM_pred_prod_avg_probs, dim=c(1,nrow(LLM_pred_prod_avg_probs),3)))
# fit_prod_RSA  <- fit_data(d_prod_global, RSA_pred_prod)
# # fit_prod_intermediate <- fit_data(d_prod_global, array(LLM_pred_prod_intermediate, dim=c(1,1,3)))
# 
# fit_inter_avg_scores <- fit_data(d_inter_global, array(LLM_pred_inter_avg_scores, dim=c(1,1,3)))
# fit_inter_WTA <- fit_data(d_inter_global, array(LLM_pred_inter_WTA, dim=c(1,1,3)))
# fit_inter_avg_probs <- fit_data(d_inter_global, array(LLM_pred_inter_avg_probs, dim=c(1,nrow(LLM_pred_inter_avg_probs),3)))
# fit_inter_RSA  <- fit_data(d_inter_global, RSA_pred_inter)
# # fit_inter_intermediate <- fit_data(d_inter_global, array(LLM_pred_inter_intermediate, dim=c(1,1,3)))

#######################################################
## Bayesian stats
#######################################################

posterior_stats <- rbind(
  produce_summary_prodInt_epsilonAlpha(fit_prod_avg_scores, fit_inter_avg_scores) |> 
    mutate(model = "avg. scores"),
  produce_summary_prodInt_epsilonAlpha(fit_prod_avg_probs, fit_inter_avg_probs) |> 
    mutate(model = "avg. probabilities"),
  produce_summary_prodInt_epsilonAlpha(fit_prod_WTA, fit_inter_WTA)  |> 
    mutate(model = "avg. WTA"),
  # produce_summary_prodInt_epsilonAlpha(fit_prod_avg_probs, fit_inter_avg_probs)  |> 
  #   mutate(model = "avg_probs"),
  produce_summary_prodInt_epsilonAlpha(fit_prod_RSA, fit_inter_RSA)  |> 
    mutate(model = "RSA")
) |>  mutate(
  condition = factor(condition, levels = c("production", "interpretation", "diff. prod-inter")),
  model     = factor(model, levels = rev(c("avg. scores", "avg. probabilities","avg. WTA", "RSA")))
  # model     = factor(model, levels = rev(c("avg_scores", "avg_probs", "intermediate", "WTA", "RSA")))
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

# pp_prod_avg_scores <- get_posterior_predictives(fit_prod_avg_scores, d_prod_global, filename = "post_pred_prod_avg_scores") |> 
#   mutate(condition = "production", model = "avg. scores")
# pp_prod_avg_probs <- get_posterior_predictives(fit_prod_avg_probs, d_prod_global, filename = "post_pred_prod_avg_probs") |> 
#   mutate(condition = "production", model = "avg. probabilities")
# pp_prod_WTA <- get_posterior_predictives(fit_prod_WTA, d_prod_global, filename = "post_pred_prod_WTA") |> 
#   mutate(condition = "production", model = "avg. WTA")
# pp_prod_RSA <- get_posterior_predictives(fit_prod_RSA, d_prod_global, filename = "post_pred_prod_RSA") |> 
#   mutate(condition = "production", model = "RSA")
# 
# 
# pp_inter_avg_scores <- get_posterior_predictives(fit_inter_avg_scores, d_inter_global, filename = "post_pred_inter_avg_scores") |> 
#   mutate(condition = "interpretation", model = "avg. scores")
# pp_inter_avg_probs <- get_posterior_predictives(fit_inter_avg_probs, d_inter_global, filename = "post_pred_inter_avg_probs") |> 
#   mutate(condition = "interpretation", model = "avg. probabilities")
# pp_inter_WTA <- get_posterior_predictives(fit_inter_WTA, d_inter_global, filename = "post_pred_inter_WTA") |> 
#   mutate(condition = "interpretation", model = "avg. WTA")
# pp_inter_RSA <- get_posterior_predictives(fit_inter_RSA, d_inter_global, filename = "post_pred_inter_RSA") |> 
#   mutate(condition = "interpretation", model = "RSA")


PPC_data = rbind(
  pp_prod_avg_scores,
  pp_prod_avg_probs,
  pp_prod_WTA,
  pp_prod_RSA,
  pp_inter_avg_scores,
  pp_inter_avg_probs,
  pp_inter_WTA,
  pp_inter_RSA
) |> select(-row) |> 
  mutate(
    condition = factor(condition, levels = c("production", "interpretation")),
    model     = factor(model, levels = rev(c("avg. scores", "avg. probabilities","avg. WTA", "RSA")))
    # model     = factor(model, levels = c("avg_scores", "avg_probs", "intermediate", "WTA", "RSA"))
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
# plot_PPC(pp_prod_WTA, pp_inter_WTA)
# plot_PPC(pp_prod_avg_scores, pp_inter_avg_scores)
# plot_PPC(pp_prod_avg_probs, pp_inter_avg_probs)
# plot_PPC(pp_prod_RSA, pp_inter_RSA)

# all models in one
PPC_data |> 
  ggplot() +
  # geom_col(data = PPC_data |> filter(model == "narrow"),
  #          aes(x = response, y = observed, fill = response)) +
  geom_col(data = d_counts, aes(x= response, y = n, fill = response)) +
  facet_grid(.~condition) +
  geom_pointrange(aes(x = response, y = mean, ymin = `|95%`, ymax = `95%|`, shape = model, group = model), 
                  position = position_dodge(width = 0.75), size = 0.6, linewidth = 0.8, color = project_colors[6]) +
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
  model = rep(c("avg_scores", "avg_probs", "avg_WTA", "RSA"), 2),
  Bppp_value = c(extract_bayesian_p(fit_prod_avg_scores),
                 extract_bayesian_p(fit_prod_avg_probs),
                 extract_bayesian_p(fit_prod_WTA),
                 extract_bayesian_p(fit_prod_RSA),
                 extract_bayesian_p(fit_inter_avg_scores),
                 extract_bayesian_p(fit_inter_avg_probs),
                 extract_bayesian_p(fit_inter_WTA),
                 extract_bayesian_p(fit_inter_RSA))  
) |> 
  pivot_wider(id_cols = model, names_from = condition, values_from = Bppp_value) |> 
  xtable::xtable()

#######################################################
## item-level analysis (LLM)
#######################################################

# prepare data (item-level)
# d_llm_prob_prod <- d |> 
#   filter(condition == "production") |> 
#   select(item, starts_with("scores_produ")) |> 
#   mutate(scores_production_distractor = 
#            log(exp(scores_production_distractor1) + exp(scores_production_distractor2))) |> 
#   select(-scores_production_distractor1, -scores_production_distractor2) |> 
#   group_by(item) |> 
#   mutate(n = n()) |>  
#   pivot_longer(-c("item", "n")) |> 
#   separate(name, into = c("Variable", "condition", "response"), sep = "_") |> 
#   select(-Variable) |> 
#   mutate(prob = exp(value)) |>
#   group_by(item, condition) |> 
#   mutate(prob = prob / sum(prob) * n) |> 
#   ungroup()
# 
# d_llm_prob_inter <- d |> 
#   filter(condition == "interpretation") |> 
#   select(item, starts_with("scores_inter")) |> 
#   group_by(item) |> 
#   mutate(n = n()) |>  
#   pivot_longer(-c("item", "n")) |> 
#   separate(name, into = c("Variable", "condition", "response"), sep = "_") |> 
#   select(-Variable) |> 
#   mutate(prob = exp(value)) |>
#   group_by(item, condition) |> 
#   mutate(prob = prob / sum(prob) * n) |> 
#   ungroup() 
# 
# d_item_analysis <- d_llm_prob_prod |>
#   rbind(d_llm_prob_inter) |>
#   mutate(condition = factor(condition, levels = c('production', 'interpretation'))) |>
#   full_join(d |>
#               select(item, feature_set, position_production, position_interpretation) |>
#               unique(),
#             by = 'item') |>
#   mutate(position = case_when(condition == "production" ~ position_production,
#                               TRUE ~ position_interpretation)) |>
#   # filter(response != "distractor") |>
#   unique() |>
#   pivot_wider(id_cols = c(item, condition, feature_set, position_production, position_interpretation, position),
#               names_from = response, values_from = prob) |> 
#   arrange(condition, item)
# 
# d_counts_items <- d |> 
#   count(condition, item, response) |> 
#   group_by(condition, item) |> 
#   mutate(total = sum(n)) |> 
#   ungroup() |> 
#   mutate(proportion = n / total) |> 
#   arrange(condition, item, response)
# 
# # prepare data (item-level, production)
# data_items_prod <- d_counts_items |> 
#   filter(condition == "production") |> 
#   pivot_wider(id_cols = item, names_from = response, values_from = n, values_fill = 0) |> 
#   select(-item) |> 
#   as.matrix()
# 
# # prepare predictions (item-level, production)
# pred_items_prod <- d_item_analysis |>
#   filter(condition == "production") |> 
#   group_by(item) |> 
#   summarize(
#     target = mean(target),
#     competitor = mean(competitor),
#     distractor = mean(distractor)
#   ) |> 
#   ungroup() |> 
#   select(-item) |> 
#   as.matrix()
# 
# # prepare data (item-level, interpretation)
# data_items_inter <- d_counts_items |> 
#   filter(condition == "interpretation") |> 
#   pivot_wider(id_cols = item, names_from = response, values_from = n, values_fill = 0) |> 
#   select(-item) |> 
#   as.matrix()
# 
# # prepare predictions (item-level, production)
# pred_items_inter <- d_item_analysis |>
#   filter(condition == "interpretation") |> 
#   group_by(item) |> 
#   summarize(
#     target = mean(target),
#     competitor = mean(competitor),
#     distractor = mean(distractor)
#   ) |> 
#   ungroup() |> 
#   select(-item) |> 
#   as.matrix()
# 
# # fit models
# fit_items_prod <- fit_data(
#   data_items_prod, 
#   array(pred_items_prod, dim = c(nrow(pred_items_prod),1, 3)), 
#   model_name = '00-stan-files/llm-average-matrix-epsilon-arrayed.stan')
# fit_items_inter <- fit_data(
#   data_items_inter, 
#   array(pred_items_inter, dim = c(nrow(pred_items_inter),1, 3)),
#   model_name = '00-stan-files/llm-average-matrix-epsilon-arrayed.stan')

# summary stats
produce_summary_prodInt_epsilonAlpha(fit_items_prod, fit_items_inter)


# PPC visualization

# production
# post_pred_items_prod <- 
#   get_posterior_predictives(
#     fit_items_prod,
#     data_items_prod,
#     "post-pred-prod-item"
#   ) |> 
#   group_by(row) |> 
#   mutate(total = sum(observed)) |> 
#   filter(response == "target") |> 
#   mutate(row = factor(row))

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
# post_pred_items_inter <- 
#   get_posterior_predictives(
#     fit_items_inter,
#     data_items_inter,
#     "post-pred-inter-item"
#   ) |> 
#   group_by(row) |> 
#   mutate(total = sum(observed)) |> 
#   filter(response == "target") |> 
#   mutate(row = factor(row))

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
## item-level analysis (w/ cond-level predict. by RSA)
#######################################################

# get_Bpppv_itemLevel_glbPredictor <- function(pred_prod, pred_inter) {
#   
#   fit_items_prod_glblPredictor <- fit_data(
#     data_items_prod, 
#     array(rep(pred_prod, each = nrow(data_items_prod)), dim = c(nrow(data_items_prod),1, 3)), 
#     model_name = '00-stan-files/llm-average-matrix-epsilon-arrayed.stan')
#   
#   fit_items_inter_glblPredictor <- fit_data(
#     data_items_inter, 
#     array(rep(pred_inter, each = nrow(data_items_inter)), dim = c(nrow(data_items_inter),1, 3)),
#     model_name = '00-stan-files/llm-average-matrix-epsilon-arrayed.stan')
#   
#   return(list(
#     production_Bpppv     = extract_bayesian_p(fit_items_prod_glblPredictor), 
#     interpretation_Bpppv = extract_bayesian_p(fit_items_inter_glblPredictor),
#     production_fit       = fit_items_prod_glblPredictor,
#     interpretation_fit   = fit_items_inter_glblPredictor
#     ))
# }
# 
# fit_item_RSA  <- get_Bpppv_itemLevel_glbPredictor(RSA_pred_prod_raw, RSA_pred_inter_raw)
# 
# # # Bayesian posterior predictive p-values
# 
# message("Bayesian p value for production (RSA, by-item):", fit_item_RSA$production_Bpppv)
# message("Bayesian p value for interpretation (RSA, by-item):", fit_item_RSA$interpretation_Bpppv)

#######################################################
## table with summary statistics for all fits 
#######################################################


sumStats_cond <- posterior_stats |> 
  filter(!is.na(condition)) |> 
  filter(condition != "diff. prod-inter") |> 
  mutate(data = "cond.-level") |> 
  select(c(7,6,5,1,2,3,4))

sumStats_item <- produce_summary_prodInt_epsilonAlpha(fit_items_prod, fit_items_inter) |> 
  mutate(model = "LLM") |>
  rbind(
    produce_summary_prodInt_epsilonAlpha(fit_items_prod_RSA, fit_items_inter_RSA) |> 
      mutate(model = "RSA")) |> 
  filter(condition != "diff. prod-inter") |> 
  mutate(data = "item-level") |> 
  select(c(7,6,5,1,2,3,4))

sumStats_item_xtable <- 
  rbind(sumStats_cond, sumStats_item) |> 
  pivot_wider(id_cols = c("data", "model", "condition"), names_from = Parameter, values_from = 5:7) |> 
  select(c(1,2,3,4,6,8,5,7,9)) |>
  mutate(condition = factor(condition, levels = c("production", "interpretation"))) |> 
  mutate(data = factor(data, levels = c("item-level", "cond.-level"))) |> 
  mutate(model = factor(model, levels = c(
    "RSA",
    "LLM", 
    "avg. scores", 
    "avg. probabilities",
    "avg. WTA"
    ))) |> 
  arrange(data, model, condition) |> 
  mutate(
    Bpppv = c(
      extract_bayesian_p(fit_items_prod_RSA),
      extract_bayesian_p(fit_items_inter_RSA),
      extract_bayesian_p(fit_items_prod),
      extract_bayesian_p(fit_items_inter),
      extract_bayesian_p(fit_prod_RSA),
      extract_bayesian_p(fit_inter_RSA),
      extract_bayesian_p(fit_prod_avg_scores),
      extract_bayesian_p(fit_inter_avg_scores),
      extract_bayesian_p(fit_prod_avg_probs),
      extract_bayesian_p(fit_inter_avg_probs),
      extract_bayesian_p(fit_prod_WTA),
      extract_bayesian_p(fit_inter_WTA)
      )
  ) |> 
  mutate(significant = ifelse(Bpppv <= 0.05, "*", "")) |>
  xtable::xtable() 

# sumStats_item_xtable
  
capture.output(print(sumStats_item_xtable))|> 
  str_replace_all(pattern = "_alpha", "") |> 
  str_replace_all(pattern = "_epsilon", "") |> 
  paste(collapse = "\n") |> cat()


#######################################################
## interpreting alpha-fits for item-level analysis
#######################################################

mean_target_prob <- function(alpha) {
  mean(softmax_row(matrix_itemLevel_prod, alpha)[as.logical(wta_row(matrix_itemLevel_prod))])
}

make_alpha_plot <- function(current_condition = "production") {
  sumStats <- sumStats_item |> 
    filter(model == "LLM") |> 
    filter(Parameter == "alpha") |> 
    filter(condition == current_condition) |> 
    select(`|95%`,  mean, `95%|`) |> 
    as.numeric()
  
  plot_data <- tibble(
    alpha = c(sumStats,seq(0,0.5, length.out = 1000)),
    mean_target_prob = map_dbl(alpha, mean_target_prob)
  ) 
  
  plot_data |> 
    ggplot(aes(x = alpha, y = mean_target_prob)) +
    geom_area(
      # aes(xmin = 1/3), 
      data = filter(plot_data, alpha >= sumStats[1] & alpha <= sumStats[3]), 
      fill = "gray", alpha = 0.5) +
    geom_segment(aes(y = 1/3, yend = 1/3, x=0, xend = 0.5), color = project_colors[1], linetype = "dotted", size = 1.1) +
    geom_segment(aes(y = 1, yend = 1, x=0, xend = 0.5), color = project_colors[4], linetype = "dotted", size = 1.1) +
    geom_line(size = 1.5) +
    geom_linerange(aes(x = sumStats[2], xmin = sumStats[1], xmax = sumStats[3], y = 0)) +
    geom_point(aes(x = sumStats[2], y = 0), size = 2) +
    geom_label(aes(x = 0.1, y = 1, label = "WTA strategy"), color = project_colors[4]) +
    geom_label(aes(x = 0.4, y = 1/3, label = "random choice"), color = project_colors[1]) +
    geom_segment(aes(x = sumStats[2], xend = sumStats[2], y = 0, yend = mean_target_prob(sumStats[2]))) +
    ylab("mean target probability") +
    ggtitle(current_condition)
  
}

outplot <- make_alpha_plot("production") + make_alpha_plot("interpretation")

ggsave(
  filename = "../03-paper/00-pics/closeness-target-by-alpha-item-level.pdf",
  plot = outplot,
  width = 9, height = 4, scale =1)
