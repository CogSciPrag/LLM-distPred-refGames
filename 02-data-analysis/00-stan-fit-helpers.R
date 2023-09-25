#' Fit data (multiple uses)
#'
#' Helper function for fitting multinomial model.
#'
#' @param d Matrix of counts; rows are cases (conditions of interest); columns are target-competitor-distractor options.
#' @param prob Row-stochastic matrix of predictions for each condition; same size as `d`.
#' @param model_name Filepath and name of Stan model file.
#' @return Fit object.
#'
#' @export
fit_data <- function(d, prob, model_name = '00-stan-files/llm-average-matrix-epsilon-arrayed.stan') {
  n_chunk <- dim(prob)[1]
  n_item  <- dim(prob)[2]
  data_list <- list(d=d, n_chunk=n_chunk, n_item=n_item, prob=prob, uniform = c(1/3, 1/3, 1/3))
  mod <- cmdstan_model(model_name)
  fit <- mod$sample(
    data = data_list,
    seed = 123,
    adapt_delta = 0.99,
    chains = 4,
    parallel_chains = 4,
    iter_warmup = 1000,
    iter_sampling = 2000
  ) 
  return(fit)
}

extract_bayesian_p <- function(fit) {
  fit$draws(format = "df") |> 
    pull(at_least_as_extreme) |> 
    mean()
}

#' Get the posterior predictives for a fitted model
#'
#' A brief description of what your function does.
#'
#' @param fit CmdStan fit object.
#' @param d_obs Matrix of counts per condition: rows are conditions, columns are target-comp-dist.
#' @return Summary stats for the posterior predictive.
#' 
#' 
get_posterior_predictives <- function(fit, d_obs, filename) {
  
  filename = str_c('cached_results/', filename, '.Rds')
  
  if (rerun == FALSE) {
    return(readRDS(filename))
  }
  
  colnames(d_obs) <- tcd
    
  make_reps <- function(obs, pred) {
    out <-  rmultinom(n=1, 
              size = sum(obs), 
              prob = pred)
    tibble(target_pred = out[1,1],
         competitor_pred = out[2,1],
         distractor_pred = out[3,1])
  }
  
  d_obs_tibble <- as_tibble(d_obs) |> 
    mutate(row = 1:nrow(d_obs)) |> 
    rename(target_obs = target,
           competitor_obs = competitor,
           distractor_obs = distractor)
  
  thetas <- fit$draws(variables = "theta", format = "df") |> 
    pivot_longer(starts_with("theta")) |> 
    separate_wider_regex(
      name,
      c(parameter = "theta", "\\[", row = "\\d+", ",", col = "\\d+", "\\]")
      ) |> 
    select(-parameter) |> 
    mutate(option = case_when(col == 1 ~ "target",
                              col == 2 ~ "competitor",
                              col == 3 ~ "distractor"
                              )) |> 
    pivot_wider(id_cols = c(".draw", "row"), 
                names_from = option) |> 
    mutate(row = as.integer(row)) |> 
    full_join(d_obs_tibble, by = 'row')
    
  
  
  d_reps <- map_df(1:nrow(thetas), function(i) {
        make_reps(c(thetas$target_obs[i], thetas$competitor_obs[i], thetas$distractor_obs[i]),
                  c(thetas$target[i], thetas$competitor[i], thetas$distractor[i]))
      })
    
  d_reps <- cbind(thetas, d_reps) |> as_tibble()
  
  d_reps_summary <- d_reps |> 
    pivot_longer(ends_with("_pred")) |> 
    separate_wider_delim(cols = name, names = c("response", "z"), delim = "_") |> 
    select(-z) |> 
    group_by(row, response) |> 
    reframe(aida::summarize_sample_vector(value)[-1]) |> 
    mutate(response = factor(response, levels = tcd)) |> 
    arrange(row, response) |> 
    full_join(
      d_obs_tibble |> 
        pivot_longer(-row) |> 
        separate_wider_delim(cols = name, names = c("response", "z"), delim = "_") |> 
        select(-z) |> 
        mutate(response = factor(response, levels = tcd)) |> 
        arrange(row, response) |> 
        rename(observed = value),
      by = c("row", "response")
    )
  
  if (! is.null(dimnames(d_obs)[[1]])) {
    d_reps_summary <- d_reps_summary |> 
      mutate(rownames = rownames(d_obs)[row])
  }
  
  write_rds(d_reps_summary, filename)
  return(d_reps_summary)
  
}

#' helper function for summaries of relevant Bayesian stats
produce_summary_prodInt_epsilonAlpha <- function(fit_prod, fit_inter) {
  rbind(
    aida::summarize_sample_vector(fit_prod$draws(format = "df") |> pull(alpha), name = "alpha"),
    aida::summarize_sample_vector(fit_prod$draws(format = "df") |> pull(epsilon), name = "epsilon"),
    aida::summarize_sample_vector(fit_inter$draws(format = "df") |> pull(alpha), name = "alpha"),
    aida::summarize_sample_vector(fit_inter$draws(format = "df") |> pull(epsilon), name = "epsilon"),
    aida::summarize_sample_vector(fit_prod$draws(format = "df") |> pull(alpha) - 
                                    fit_inter$draws(format = "df") |> pull(alpha), name = "alpha"),
    aida::summarize_sample_vector(fit_prod$draws(format = "df") |> pull(epsilon) - 
                                    fit_inter$draws(format = "df") |> pull(epsilon), name = "epsilon")
  ) |> 
    mutate(condition = rep(c("production", "interpretation", "diff. prod-inter"), each = 2))
}
