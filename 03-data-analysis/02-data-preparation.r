source('00-premable.r')

##################################################
## helper functions 
##################################################

softmax_row <- function(matrix_of_scores, alpha = 1) {
  exp_matrix <- exp(alpha * matrix_of_scores)
  row_sums <- rowSums(exp_matrix)
  softmax_matrix <- exp_matrix / row_sums
  return(softmax_matrix)
}

wta_row <- function(matrix_of_scores) {
  result_matrix <- matrix(0, nrow = nrow(matrix_of_scores), ncol = ncol(matrix_of_scores))
  for (i in 1:nrow(matrix_of_scores)) {
    max_val <- max(matrix_of_scores[i, ])
    max_indices <- which(matrix_of_scores[i, ] == max_val)
    result_matrix[i, max_indices] <- 1/length(max_indices)
  }
  return(result_matrix)
}

##################################################
## main function to prepare data
##################################################

get_prepped_data_for_model <- function(model_name, results_file) {
  
  d_raw <- read_csv('../02-data/data-raw-human.csv') |> 
    select(- c(
      "experiment_duration", "experiment_end_time", "experiment_start_time",
      "prolific_pid", "prolific_session_id", "prolific_study_id"  )) |> 
    rename(item = trial) |> 
    mutate(item = factor(item, levels = 0:99))
  
  d_sim <- read_csv(results_file) |> 
    rename(item = trial) |> 
    mutate(item = factor(item, levels = 0:99))
  
  d_join <- left_join(d_raw, d_sim, by = "item") 
  
  d_join$response_type = map_chr(1:nrow(d_join), function(i) {
    condition <-  d_join[i,] |> pull(condition)
    response  <-  d_join[i,] |> pull(response)
    if (condition == "production") {
      if (response == d_join[i,] |> pull(production_target)) {
        return("target")
      } else {
        if (response == d_join[i,] |> pull(production_competitor)) {
          return("competitor")
        } 
        else {
          return("distractor")
        }  
      }
    }
    if (condition == "interpretation") {
      if (response |> stringr::str_sub(3) == d_join[i,] |> pull(interpretation_target)) {
        return("target")
      } else {
        if (response |> stringr::str_sub(3) == d_join[i,] |> pull(interpretation_competitor)) {
          return("competitor")
        } else {
          return("distractor")
        }
      }
    }
  })
  
  d_join$prediction_type = map_chr(1:nrow(d_join), function(i) {
    condition <-  d_join[i,] |> pull(condition)
    string_match = ifelse (condition == "production", 'scores_production', 'scores_interpretation')
    scores <-  d_join[i,] |> select(starts_with(string_match)) |> as.double()
    return(c("target", "competitor", "distractor", "distractor")[which(scores == max(scores))][1])
  })
  
  d <- d_join |> 
    rename(
      prediction = prediction_type,
      response_raw = response,
      response = response_type
    ) 
  
  ## add position information string
  
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
  ) |> 
    factor(levels = c(
      "[d,d,c,t]",
      "[d,d,t,c]",
      "[d,c,t,d]",
      "[d,t,c,d]",
      "[c,d,t,d]",
      "[t,d,d,c]",
      "[t,c,d,d]",
      "[d,c,d,t]",
      "[c,t,d,d]",
      "[t,d,c,d]",
      "[c,d,d,t]",
      "[d,t,d,c]"
    ))
  
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
  
  d$position_production <- position_production
  d$position_interpretation <- position_interpretation
  
  d <- d |> 
    mutate(position = case_when(condition == "production" ~ position_production,
                                TRUE ~ position_interpretation) |> 
             factor(levels = c(
               "[t,c,d]", 
               "[t,d,c]",
               "[d,t,c]", 
               "[d,c,t]",
               "[c,t,d]", 
               "[c,d,t]",
               "[d,d,c,t]",
               "[d,d,t,c]",
               "[d,c,t,d]",
               "[d,t,c,d]",
               "[c,d,t,d]",
               "[t,d,d,c]",
               "[t,c,d,d]",
               "[d,c,d,t]",
               "[c,t,d,d]",
               "[t,d,c,d]",
               "[c,d,d,t]",
               "[d,t,d,c]"
             ))
    )
  
  # add 'feature_set' information & make leveled factors 
  tcd <- c("target", "competitor", "distractor")
  d <-  d |>  
    mutate(feature_set = str_c(trigger_feature, "-", nuisance_feature)) |> 
    mutate(
      response   = factor(response  , levels = tcd),
      prediction = factor(prediction, levels = tcd),
      condition  = factor(condition,  levels = c("production", "interpretation"))
    )  
  
  ## write to file
  d |> write_csv(file = str_c("02-prepped-data/data-prepped-", model_name ,".csv"))
  
  #######################################################
  ## prepare model predictions 
  #######################################################
  
  ####### production
  
  # wrangle data
  x_prod <- d |> 
    filter(condition == "production") |> 
    select(submission_id, trial_nr, item, condition, starts_with("scores_produ")) |> 
    mutate(scores_production_distractor = 
             log(exp(scores_production_distractor1) + exp(scores_production_distractor2))) |> 
    select(-scores_production_distractor1, -scores_production_distractor2)
  
  # narrow- and intermediate-scope predictions for production
  matrix_global_prod <- x_prod |> 
    select(starts_with("scores_produ")) |> as.matrix()
  LLM_pred_prod_narrow       <- softmax_row(matrix(apply(matrix_global_prod, 2, mean), nrow = 1))
  LLM_pred_prod_intermediate <- matrix(apply(softmax_row(matrix_global_prod), 2, mean), nrow = 1)
  LLM_pred_prod_WTA          <- matrix(apply(wta_row(matrix_global_prod), 2, mean), nrow = 1)
  
  # wide-scope predictions
  matrix_itemLevel_prod <- x_prod |> 
    select(item, starts_with("scores_produ")) |> 
    unique() |> 
    arrange(item) |> 
    select(starts_with("scores_produ")) |> as.matrix()
  LLM_pred_prod_wide  <- softmax_row(matrix_itemLevel_prod)
  
  LLM_pred_prod_avg_scores   <- LLM_pred_prod_narrow
  LLM_pred_prod_avg_probs    <- LLM_pred_prod_wide
  LLM_pred_prod_WTA          <- LLM_pred_prod_WTA + 1e-7 / sum(LLM_pred_prod_WTA + 1e-7)
  
  ####### interpretation
  
  # wrangle data
  x_inter <- d |> 
    filter(condition == "interpretation") |> 
    select(submission_id, trial_nr, item, condition, starts_with("scores_inter"))
  
  # narrow- and intermediate-scope predictions for production
  matrix_global_inter <- x_inter |> 
    select(starts_with("scores_inter")) |> as.matrix()
  LLM_pred_inter_narrow       <- softmax_row(matrix(apply(matrix_global_inter, 2, mean), nrow = 1))
  LLM_pred_inter_intermediate <- matrix(apply(softmax_row(matrix_global_inter), 2, mean), nrow = 1)
  LLM_pred_inter_WTA          <- matrix(apply(wta_row(matrix_global_inter), 2, mean), nrow = 1)
  
  # wide-scope predictions
  matrix_itemLevel_inter <- x_inter |> 
    select(item, starts_with("scores_inter")) |> 
    unique() |> 
    arrange(item) |> 
    select(starts_with("scores_inter")) |> as.matrix()
  LLM_pred_inter_wide  <- softmax_row(matrix_itemLevel_inter)
  
  # predictions (to compare)
  LLM_pred_inter_avg_scores   <- LLM_pred_inter_narrow
  LLM_pred_inter_avg_probs    <- LLM_pred_inter_wide
  LLM_pred_inter_WTA          <- LLM_pred_inter_WTA + 1e-7 / sum(LLM_pred_inter_WTA + 1e-7)
  
  
  ######################################################
  # prepare data (item-level)
  ######################################################
  
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
  
  # prepare predictions (item-level, production)
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
  
  # prepare predictions (item-level, interpretation)
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
  
  # save model predictions
  model_predictions <- list(
    LLM_pred_prod_avg_scores  = LLM_pred_prod_avg_scores,
    LLM_pred_prod_avg_probs   = LLM_pred_prod_avg_probs,
    LLM_pred_prod_WTA         = LLM_pred_prod_WTA,
    LLM_pred_inter_avg_scores = LLM_pred_inter_avg_scores,
    LLM_pred_inter_avg_probs  = LLM_pred_inter_avg_probs,
    LLM_pred_inter_WTA        = LLM_pred_inter_WTA,
    pred_items_prod           = pred_items_prod,
    pred_items_inter          = pred_items_inter,
    matrix_itemLevel_prod     = matrix_itemLevel_prod,
    matrix_itemLevel_inter    = matrix_itemLevel_inter
  )
  
  saveRDS(model_predictions, file = str_c("02-prepped-data/predictions-", model_name, ".rds"))
  
}

get_prepped_data_for_model("LLaMA2-chat-hf-70b", "../02-data/results_Llama-2-70b-chat-hf.csv")
get_prepped_data_for_model("LLaMA2-chat-hf-13b", "../02-data/results_Llama-2-13b-chat-hf.csv")
get_prepped_data_for_model("LLaMA2-chat-hf-7b" , "../02-data/results_Llama-2-7b-chat-hf.csv")
get_prepped_data_for_model("LLaMA2-hf-70b"     , "../02-data/results_Llama-2-70b-hf.csv")
get_prepped_data_for_model("LLaMA2-hf-13b"     , "../02-data/results_Llama-2-13b-hf.csv")
get_prepped_data_for_model("LLaMA2-hf-7b"      , "../02-data/results_Llama-2-7b-hf.csv")
get_prepped_data_for_model("GPT"               , "../02-data/results_GPT.csv")
