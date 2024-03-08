# dependencies etc.
rerun = F
source("00-premable.r")
source('00-stan-fit-helpers.R')

#######################################################-
## read and prep data, get counts, massage human data
#######################################################-

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

#######################################################-
## helper functions: save/load cmdstanr fit object
#######################################################-

save_fit_object <- function(fit_object, path, filename) {
  fit_object$save_object(file = str_c(path, filename))
}

#######################################################-
## fit models
#######################################################-

get_model_fits_RSA <- function(rerun = FALSE) {
  
  model_path = str_c("03-model-fits/RSA/")
  
  if (rerun == FALSE) {
    model_fits <- list(
      model_name            = "RSA",
      fit_prod_RSA          = readRDS(file = str_c(model_path, "fit_prod_RSA.RDS")),
      fit_inter_RSA         = readRDS(file = str_c(model_path, "fit_inter_RSA.RDS")),
      pp_prod_RSA           = readRDS(file = str_c(model_path, "pp_prod_RSA.RDS")),
      pp_inter_RSA          = readRDS(file = str_c(model_path, "pp_inter_RSA.RDS")),
      fit_items_prod_RSA    = readRDS(file = str_c(model_path, "fit_items_prod_RSA.RDS")),
      fit_items_inter_RSA   = readRDS(file = str_c(model_path, "fit_items_inter_RSA.RDS"))
    )
    return(model_fits)
  }
  
  ## RSA model predictions (Vanilla, alpha = 0)
  RSA_pred_prod_raw  <- c(2/3,1/3,0) + 1e-7 / sum(c(2/3,1/3,0) + 1e-7)
  RSA_pred_inter_raw <- c(0.6,0.4,0) + 1e-7 / sum(c(0.6,0.4,0) + 1e-7)
  RSA_pred_prod      <- array(RSA_pred_prod_raw,  dim=c(1,1,3))
  RSA_pred_inter     <- array(RSA_pred_inter_raw, dim=c(1,1,3))
  
  # condition-level fits
  fit_prod_RSA   <- fit_data(d_prod_global, RSA_pred_prod)
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
  
  # save everything
  save_fit_object(fit_prod_RSA, model_path, "fit_prod_RSA.RDS")
  save_fit_object(fit_inter_RSA, model_path, "fit_iter_RSA.RDS")
  save_fit_object(fit_items_prod_RSA, model_path, "fit_items_prod_RSA.RDS")
  save_fit_object(fit_items_inter_RSA, model_path, "fit_items_inter_RSA.RDS")
  saveRDS(pp_prod_RSA, file = str_c(model_path, "pp_prod_RSA.RDS"))
  saveRDS(pp_inter_RSA, file = str_c(model_path, "pp_inter_RSA.RDS"))
  
  # return output
  model_fits <- list(
    model_name            = "RSA",
    fit_prod_RSA          = fit_prod_RSA,
    fit_inter_RSA         = fit_inter_RSA,
    pp_prod_RSA           = pp_prod_RSA,
    pp_inter_RSA          = pp_inter_RSA,
    fit_items_prod_RSA    = fit_items_prod_RSA,
    fit_items_inter_RSA   = fit_items_inter_RSA
  )
  
  return(model_fits)
}

load_fitted_model <- function(model_name) {
  
  model_path = str_c("03-model-fits/",model_name,"/")
  
  model_fits <- list(
    model_name             = model_name,
    fit_prod_avg_scores    = readRDS(file = str_c(model_path, "fit_prod_avg_scores.RDS")),
    fit_prod_WTA           = readRDS(file = str_c(model_path, "fit_prod_WTA.RDS")),
    fit_prod_avg_probs     = readRDS(file = str_c(model_path, "fit_prod_avg_probs.RDS")),
    fit_inter_avg_scores   = readRDS(file = str_c(model_path, "fit_inter_avg_scores.RDS")),
    fit_inter_WTA          = readRDS(file = str_c(model_path, "fit_inter_WTA.RDS")),
    fit_inter_avg_probs    = readRDS(file = str_c(model_path, "fit_inter_avg_probs.RDS")),
    fit_items_prod         = readRDS(file = str_c(model_path, "fit_items_prod.RDS")),
    fit_items_inter        = readRDS(file = str_c(model_path, "fit_items_inter.RDS")),
    pp_prod_avg_scores     = readRDS(file = str_c(model_path, "pp_prod_avg_scores.RDS")),
    pp_prod_WTA            = readRDS(file = str_c(model_path, "pp_prod_WTA.RDS")),
    pp_prod_avg_probs      = readRDS(file = str_c(model_path, "pp_prod_avg_probs.RDS")),
    pp_inter_avg_scores    = readRDS(file = str_c(model_path, "pp_inter_avg_scores.RDS")),
    pp_inter_WTA           = readRDS(file = str_c(model_path, "pp_inter_WTA.RDS")),
    pp_inter_avg_probs     = readRDS(file = str_c(model_path, "pp_inter_avg_probs.RDS")),
    post_pred_items_prod   = readRDS(file = str_c(model_path, "post_pred_items_prod.RDS")),
    post_pred_items_inter  = readRDS(file = str_c(model_path, "post_pred_items_inter.RDS")),
    matrix_itemLevel_prod  = readRDS(file = str_c(model_path, "model_predictions$matrix_itemLevel_prod.RDS")),
    matrix_itemLevel_inter = readRDS(file = str_c(model_path, "model_predictions$matrix_itemLevel_inter.RDS"))
  )
  
  return(model_fits)
}

get_model_fits <- function(model_name, rerun = FALSE) {
  
  model_path = str_c("03-model-fits/",model_name,"/")
  
  if (rerun == FALSE) {
    return(load_fitted_model(model_name))
  }

   
  ######################################################-
  ## read LLM predictions
  ######################################################-
  
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
  ## save fit objects
  #####################################################

  # save everything
  save_fit_object(fit_prod_avg_scores, model_path, "fit_prod_avg_scores.RDS")
  save_fit_object(fit_prod_WTA, model_path, "fit_prod_WTA.RDS")
  save_fit_object(fit_prod_avg_probs, model_path, "fit_prod_avg_probs.RDS")
  save_fit_object(fit_inter_avg_scores, model_path, "fit_inter_avg_scores.RDS")
  save_fit_object(fit_inter_WTA, model_path, "fit_inter_WTA.RDS")
  save_fit_object(fit_inter_avg_probs, model_path, "fit_inter_avg_probs.RDS")
  save_fit_object(fit_items_prod, model_path, "fit_items_prod.RDS")
  save_fit_object(fit_items_inter, model_path, "fit_items_inter.RDS")
  saveRDS(pp_prod_avg_scores, file = str_c(model_path, "pp_prod_avg_scores.RDS"))
  saveRDS(pp_prod_WTA, file = str_c(model_path, "pp_prod_WTA.RDS"))
  saveRDS(pp_prod_avg_probs, file = str_c(model_path, "pp_prod_avg_probs.RDS"))
  saveRDS(pp_inter_avg_scores, file = str_c(model_path, "pp_inter_avg_scores.RDS"))
  saveRDS(pp_inter_WTA, file = str_c(model_path, "pp_inter_WTA.RDS"))
  saveRDS(pp_inter_avg_probs, file = str_c(model_path, "pp_inter_avg_probs.RDS"))
  saveRDS(post_pred_items_prod, file = str_c(model_path, "post_pred_items_prod.RDS"))
  saveRDS(post_pred_items_inter, file = str_c(model_path, "post_pred_items_inter.RDS"))
  saveRDS(model_predictions$matrix_itemLevel_prod,
          file = str_c(model_path, "model_predictions$matrix_itemLevel_prod.RDS"))
  saveRDS(model_predictions$matrix_itemLevel_inter,
          file = str_c(model_path, "model_predictions$matrix_itemLevel_inter.RDS"))

  # return output
  model_fits <- list(
    model_name             = model_name,
    fit_prod_avg_scores    = fit_prod_avg_scores,
    fit_prod_WTA           = fit_prod_WTA,
    fit_prod_avg_probs     = fit_prod_avg_probs,
    fit_inter_avg_scores   = fit_inter_avg_scores,
    fit_inter_WTA          = fit_inter_WTA,
    fit_inter_avg_probs    = fit_inter_avg_probs,
    fit_items_prod         = fit_items_prod,
    fit_items_inter        = fit_items_inter,
    pp_prod_avg_scores     = pp_prod_avg_scores,
    pp_prod_WTA            = pp_prod_WTA,
    pp_prod_avg_probs      = pp_prod_avg_probs,
    pp_inter_avg_scores    = pp_inter_avg_scores,
    pp_inter_WTA           = pp_inter_WTA,
    pp_inter_avg_probs     = pp_inter_avg_probs,
    post_pred_items_prod   = post_pred_items_prod,
    post_pred_items_inter  = post_pred_items_inter,
    matrix_itemLevel_prod  = model_predictions$matrix_itemLevel_prod,
    matrix_itemLevel_inter = model_predictions$matrix_itemLevel_inter
  )

  return(model_fits)
}

# all model names (in order)
model_names = c(
  "LLaMA2-chat-hf-70b", 
  "LLaMA2-chat-hf-13b", 
  "LLaMA2-chat-hf-7b", 
  "LLaMA2-hf-70b",
  "LLaMA2-hf-13b",
  "LLaMA2-hf-7b",
  "GPT"
)

model_fits_RSA <- get_model_fits_RSA(rerun = rerun)

for (model in model_names) {
  print(model)
  model_fits      <- get_model_fits(model, rerun = rerun)
}

# uncomment selected lines for individual reruns
# model_fits_70b_chat <- get_model_fits("LLaMA2-chat-hf-70b", rerun = T) 
# model_fits_13b_chat <- get_model_fits("LLaMA2-chat-hf-13b", rerun = T) 
# model_fits_07b_chat <- get_model_fits("LLaMA2-chat-hf-7b",  rerun = T)
# model_fits_70b      <- get_model_fits("LLaMA2-hf-70b", rerun = T)
# model_fits_13b      <- get_model_fits("LLaMA2-hf-13b", rerun = T)
# model_fits_07b      <- get_model_fits("LLaMA2-hf-7b", rerun = T)
# model_fits_GPT      <- get_model_fits("GPT", rerun = T)







