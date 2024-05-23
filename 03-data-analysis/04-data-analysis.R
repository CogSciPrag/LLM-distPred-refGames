# dependencies etc.
source("00-premable.r")
source('00-stan-fit-helpers.R')
source('02-data-preparation.r')
source('03-model-fitting.R')

model_names = c(
  "LLaMA2-chat-hf-70b", 
  "LLaMA2-chat-hf-13b", 
  "LLaMA2-chat-hf-7b", 
  "LLaMA2-hf-70b",
  "LLaMA2-hf-13b",
  "LLaMA2-hf-7b",
  "GPT"
)

model_name = model_names[7]

retrieve_results_model_fits <- function(model_name) {

  #######################################################
  ## retrieve information from fitted models (RSA & GPT)
  #######################################################
  
  model_fits_RSA <- get_model_fits_RSA(rerun = FALSE)
  attach(model_fits_RSA)
  
  model_fits = get_model_fits(model_name, rerun = FALSE)
  attach(model_fits)
  
  #######################################################
  ## Bayesian stats for posterior of alpha & epsilon
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
  
  ggsave(filename = str_c("../04-paper/00-pics/", model_name, "-posterior-stats.pdf"), 
         width = 8, height = 4, scale = 1.0)
  
  #######################################################
  ## plot posterior predictives
  #######################################################
  
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
    mutate(model = case_when(model == "avg. probabilities" ~ "avg. prob.", T ~ model)) |> 
    mutate(
      condition = factor(condition, levels = c("production", "interpretation")),
      model     = factor(model, levels = rev(c("avg. scores", "avg. prob.","avg. WTA", "RSA")))
    )
  
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
    # include legend on the right
    theme(legend.position = "right") +
    # suppress legend title
    theme(legend.title = element_blank())
  
  ggsave(filename = str_c("../04-paper/00-pics/", model_name, "-PPC-alpha-eps-model.pdf"), 
         width = 8, height = 3.5, scale = 1.0)
  
  #######################################################
  ## Bayesian p-values
  #######################################################
  
  # tibble(
  #   condition = rep(c("production", "interpretation"), each = 4),
  #   model = rep(c("avg_scores", "avg_probs", "avg_WTA", "RSA"), 2),
  #   Bppp_value = c(extract_bayesian_p(fit_prod_avg_scores),
  #                  extract_bayesian_p(fit_prod_avg_probs),
  #                  extract_bayesian_p(fit_prod_WTA),
  #                  extract_bayesian_p(fit_prod_RSA),
  #                  extract_bayesian_p(fit_inter_avg_scores),
  #                  extract_bayesian_p(fit_inter_avg_probs),
  #                  extract_bayesian_p(fit_inter_WTA),
  #                  extract_bayesian_p(fit_inter_RSA))  
  # ) |> 
  #   pivot_wider(id_cols = model, names_from = condition, values_from = Bppp_value) |> 
  #   xtable::xtable()
  
  #######################################################
  ## item-level analysis (LLM)
  #######################################################
  
  # summary stats
  # produce_summary_prodInt_epsilonAlpha(fit_items_prod, fit_items_inter)
  
  # PPC visualization
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
  
  ggsave(filename = str_c("../04-paper/00-pics/", model_name, "-item-prod-postPred.pdf")
         , width = 5, height = 4, scale = 1.0)
  
  post_pred_items_prod |> 
    ggplot(aes(x = mean/total, observed / total)) +
    geom_segment((aes(x = 0, y = 0, xend = 1, yend=1)), color = "gray") +
    geom_point(alpha = 0.8) +
    xlim(c(0,1)) +
    ylim(c(0,1)) + 
    ylab("observed") +
    xlab("predicted") +
    ggtitle("production")
  
  ggsave(filename = str_c("../04-paper/00-pics/", model_name, "-item-prod-obs-pred.pdf"), 
         width = 5, height = 4, scale = 1.0)
  
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
  
  ggsave(filename = str_c("../04-paper/00-pics/", model_name, "-item-inter-postPred.pdf"), 
         width = 5, height = 4, scale = 1.0)
  
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
  
  ggsave(filename = str_c("../04-paper/00-pics/", model_name, "-item-inter-obs-pred.pdf"), 
         width = 5, height = 4, scale = 1.0)
  
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
  
  ggsave(filename = str_c("../04-paper/00-pics/", model_name, "-item-combined-obs-pred.pdf"), 
         width = 9, height = 4, scale = 1.0)
  
  #######################################################
  ## table with summary statistics for all fits 
  #######################################################
  
  sumStats_cond <- posterior_stats |> 
    filter(!is.na(condition)) |> 
    filter(condition != "diff. prod-inter") |> 
    mutate(data = "cond.") |> 
    rename(method = model) |> 
    mutate(model  = case_when(method == "RSA" ~ "RSA", T ~ model_name)) |> 
    select(c(7,8,6,5,1,2,3,4)) |> 
    mutate(method = case_when(method == "RSA" ~ "---", T ~ method))
  
  sumStats_item <- produce_summary_prodInt_epsilonAlpha(fit_items_prod, fit_items_inter) |> 
    mutate(model = model_name) |>
    rbind(
      produce_summary_prodInt_epsilonAlpha(fit_items_prod_RSA, fit_items_inter_RSA) |> 
        mutate(model = "RSA")) |> 
    mutate(method = "---") |> 
    filter(condition != "diff. prod-inter") |> 
    mutate(data = "item") |> 
    select(c(8,6,7,5,1,2,3,4))
  
  sumStats_combined <- 
    rbind(sumStats_cond, sumStats_item) |> 
    pivot_wider(id_cols = c("data", "model", "method", "condition"), names_from = Parameter, values_from = 6:8) |> 
    select(c(2,1,3,4,5,7,9,6,8,10)) |>
    mutate(
      condition = case_when(condition == "production" ~ "prd.", T ~ "int."),
      condition = factor(condition, levels = c("prd.", "int."))
    ) |> 
    mutate(data = factor(data, levels = c("item", "cond."))) |> 
    mutate(model = factor(model, levels = c("RSA", model_name))) |> 
    mutate(
      method = case_when(method == "avg. probabilities" ~ "avg. prob.", T ~ method),
      method = factor(method, levels = c(
        "---", 
        "avg. scores", 
        "avg. prob.",
        "avg. WTA"
      ))) |> 
    arrange(model, data, method, condition) |> 
    mutate(
      Bpppv = c(
        extract_bayesian_p(fit_items_prod_RSA),
        extract_bayesian_p(fit_items_inter_RSA),
        extract_bayesian_p(fit_prod_RSA),
        extract_bayesian_p(fit_inter_RSA),
        extract_bayesian_p(fit_items_prod),
        extract_bayesian_p(fit_items_inter),
        extract_bayesian_p(fit_prod_avg_scores),
        extract_bayesian_p(fit_inter_avg_scores),
        extract_bayesian_p(fit_prod_avg_probs),
        extract_bayesian_p(fit_inter_avg_probs),
        extract_bayesian_p(fit_prod_WTA),
        extract_bayesian_p(fit_inter_WTA)
      )
    ) |> 
    mutate(significant = ifelse(Bpppv <= 0.05, "*", ""))
  
  write_csv(sumStats_combined, file = str_c("04-analysis-results/", model_name, "sumStats.csv"))
  
  
  #######################################################
  ## interpreting alpha-fits for item-level analysis
  #######################################################
  
  mean_target_prob <- function(alpha, current_condition = "production") {
    if (current_condition == "production") {
      return(mean(softmax_row(matrix_itemLevel_prod, alpha)[as.logical(wta_row(matrix_itemLevel_prod))]))
    } else {
      return(mean(softmax_row(matrix_itemLevel_inter, alpha)[as.logical(wta_row(matrix_itemLevel_inter))]))
    }
  }
  
  make_alpha_plot <- function(current_condition = "production") {
    sumStats <- sumStats_item |> 
      filter(model == model_name) |> 
      filter(Parameter == "alpha") |> 
      filter(condition == current_condition) |> 
      select(`|95%`,  mean, `95%|`) |> 
      as.numeric()
    
    xMax <- sumStats[3]+ 0.1*sumStats[2]
    
    plot_data <- tibble(
      alpha = c(sumStats,seq(0, xMax, length.out = 1000)),
      mean_target_prob = map_dbl(alpha, function(x) mean_target_prob(x, current_condition))
    ) 
    
    polygon_df <- rbind(
      data.frame(alpha = c(0, sumStats[1]), 
                 mean_target_prob = c(as.numeric(plot_data[1,2]), as.numeric(plot_data[1,2]))),
      filter(plot_data, alpha >= sumStats[1] & alpha <= sumStats[3]),
      data.frame(alpha = c(sumStats[2], 0), 
                 mean_target_prob = c(as.numeric(plot_data[3,2]), as.numeric(plot_data[3,2])))
    )
    
    plot_data |> 
      ggplot(aes(x = alpha, y = mean_target_prob)) +
      geom_area(
        # aes(xmin = 1/3), 
        data = filter(plot_data, alpha >= sumStats[1] & alpha <= sumStats[3]), 
        fill = "gray", alpha = 0.5) +
      geom_polygon(data = polygon_df, fill = "gray", alpha = 0.5) +
      geom_segment(aes(y = 1/3, yend = 1/3, x=0, xend = xMax), 
                   color = project_colors[1], 
                   linetype = "dotted", size = 1.1) +
      geom_segment(aes(y = 1, yend = 1, x=0, xend = xMax), 
                   color = project_colors[4], 
                   linetype = "dotted", size = 1.1) +
      geom_line(size = 1.5) +
      # 95% CI for alpha
      geom_linerange(aes(x = sumStats[2], xmin = sumStats[1], xmax = sumStats[3], y = 0)) +
      geom_point(aes(x = sumStats[2], y = 0), size = 2) +
      geom_segment(aes(x = sumStats[2], xend = sumStats[2], y = 0, 
                       yend = mean_target_prob(sumStats[2], current_condition))) +
      # 95% CI for prediction
      geom_linerange(aes(y = as.numeric(plot_data[2,2]), ymin = as.numeric(plot_data[1,2]),
                         ymax = as.numeric(plot_data[3,2]), x = 0)) +
      geom_point(aes(y = as.numeric(plot_data[2,2]), x = 0), size = 2) +
      geom_segment(aes(y = as.numeric(plot_data[2,2]), yend = as.numeric(plot_data[2,2]), 
                       x = 0, xend = sumStats[2])) +
      # reference lines
      geom_label(aes(x = 0.1, y = 1, label = "WTA strategy"), color = project_colors[4]) +
      geom_label(aes(x = 0.4, y = 1/3, label = "random choice"), color = project_colors[1]) +
      ylab("mean prob. of best option") +
      ggtitle(current_condition)
    
  }
  
  alpha_plot_production <- make_alpha_plot("production")
  alpha_plot_interpretation <- make_alpha_plot("interpretation")
  outplot <- gridExtra::grid.arrange(alpha_plot_production, alpha_plot_interpretation, ncol = 2)
  
  ggsave(
    filename = stringr::str_c("../04-paper/00-pics/", model_name, "-closeness-target-by-alpha-item-level.pdf"),
    plot = outplot, width = 9, height = 4, scale =1)
    
}

for (model_name in model_names) {
  retrieve_results_model_fits(model_name)
}

##################################################
## inspecting all model fits and summary stats
##################################################

# get model fits
RSA                <- get_model_fits_RSA(rerun = FALSE)
LLaMA2_chat_hf_70b <- get_model_fits("LLaMA2-chat-hf-70b", rerun = FALSE)
LLaMA2_chat_hf_13b <- get_model_fits("LLaMA2-chat-hf-13b", rerun = FALSE)
LLaMA2_chat_hf_7b  <- get_model_fits("LLaMA2-chat-hf-7b", rerun = FALSE)
LLaMA2_hf_70b      <- get_model_fits("LLaMA2-hf-70b", rerun = FALSE)
LLaMA2_hf_13b      <- get_model_fits("LLaMA2-hf-13b", rerun = FALSE)
LLaMA2_hf_7b       <- get_model_fits("LLaMA2-hf-7b", rerun = FALSE)
GPT                <- get_model_fits("GPT", rerun = FALSE)

# sumStats_RSA                <- read_csv("04-analysis-results/RSAsumStats.csv"))
sumStats_LLaMA2_chat_hf_70b <- read_csv("04-analysis-results/LLaMA2-chat-hf-70bsumStats.csv")
sumStats_LLaMA2_chat_hf_13b <- read_csv("04-analysis-results/LLaMA2-chat-hf-13bsumStats.csv")
sumStats_LLaMA2_chat_hf_7b  <- read_csv("04-analysis-results/LLaMA2-chat-hf-7bsumStats.csv")
sumStats_LLaMA2_hf_70b      <- read_csv("04-analysis-results/LLaMA2-hf-70bsumStats.csv")
sumStats_LLaMA2_hf_13b      <- read_csv("04-analysis-results/LLaMA2-hf-13bsumStats.csv")
sumStats_LLaMA2_hf_7b       <- read_csv("04-analysis-results/LLaMA2-hf-7bsumStats.csv")
sumStats_GPT                <- read_csv("04-analysis-results/GPTsumStats.csv")

# table of all model results (LaTeX)

# bind all sumStats together
sumStats_all <- 
  rbind(
    sumStats_LLaMA2_chat_hf_70b,
    sumStats_LLaMA2_chat_hf_13b, 
    sumStats_LLaMA2_chat_hf_7b,
    sumStats_LLaMA2_hf_70b,
    sumStats_LLaMA2_hf_13b,
    sumStats_LLaMA2_hf_7b,
    sumStats_GPT              
  ) |> 
  unique() |> 
  mutate(
    model = factor(model, levels = c("RSA", "GPT", 
                                     "LLaMA2-hf-7b", "LLaMA2-hf-13b", "LLaMA2-hf-70b", 
                                     "LLaMA2-chat-hf-7b", "LLaMA2-chat-hf-13b", "LLaMA2-chat-hf-70b"))
  ) |> 
    
  # specify levels for 'backend models'
  mutate(backend = case_when(
    model == "LLaMA2-hf-7b" ~ "L2-hf-7b",
    model == "LLaMA2-hf-13b" ~ "L2-hf-13b",
    model == "LLaMA2-hf-70b" ~ "L2-hf-70b",
    model == "LLaMA2-chat-hf-7b" ~ "L2-chat-7b",
    model == "LLaMA2-chat-hf-13b" ~ "L2-chat-13b",
    model == "LLaMA2-chat-hf-70b" ~ "L2-chat-70b",
    T ~ model
  )) |> 
  mutate(model = factor(backend, 
                          levels = c(
                            "RSA", "GPT", 
                            "L2-hf-7b", "L2-hf-13b", "L2-hf-70b", 
                            "L2-chat-7b", "L2-chat-13b", "L2-chat-70b"
                            ))) |> 
  mutate(
    condition = factor(condition, levels = c("prd.", "int.")),
    method = factor(method, levels = c("avg. scores", "avg. prob.", "avg. WTA", "---")),
    data = factor(data, levels = c("item", "cond."))
  ) |> 
  arrange(model, data, method, condition) |> 
  select(-backend) |> 
  mutate(
    ` ` = ifelse(is.na(significant), "!", "")
  ) |> 
  select(' ', everything()) |> 
  select(-significant)

# print in LaTeX format
capture.output(print(xtable::xtable(data.frame(sumStats_all), include.rownames = FALSE)))|>
  str_replace_all(pattern = "_alpha", "") |>
  str_replace_all(pattern = "_epsilon", "") |>
  paste(collapse = "\n") |> cat()

##################################################
# visual posterior predictive checks 
# for all LLaMA models (condensed version)
##################################################

# extract data for a given model
extract_PPC_data <- function(backend, backend_name) {
  PPC_data = rbind(
    backend$pp_prod_avg_scores,
    backend$pp_prod_avg_probs,
    backend$pp_prod_WTA,
    backend$pp_prod_RSA,
    backend$pp_inter_avg_scores,
    backend$pp_inter_avg_probs,
    backend$pp_inter_WTA,
    backend$pp_inter_RSA
  ) |> select(-row) |> 
    mutate(model = case_when(model == "avg. probabilities" ~ "avg. prob.", T ~ model)) |> 
    mutate(
      condition = factor(condition, levels = c("production", "interpretation")),
      model     = factor(model, levels = rev(c("avg. scores", "avg. prob.","avg. WTA", "RSA"))),
      backend   = backend_name
    )  
  return(PPC_data)
}

# assemble data for plotting
vPPC_data_all_models <- rbind(
  extract_PPC_data(GPT, "GPT"),
  extract_PPC_data(RSA, "RSA"),
  extract_PPC_data(LLaMA2_chat_hf_70b, "LLaMA2-chat-hf-70b"),
  extract_PPC_data(LLaMA2_chat_hf_13b, "LLaMA2-chat-hf-13b"),
  extract_PPC_data(LLaMA2_chat_hf_7b, "LLaMA2-chat-hf-7b"),
  extract_PPC_data(LLaMA2_hf_70b, "LLaMA2-hf-70b"),
  extract_PPC_data(LLaMA2_hf_13b, "LLaMA2-hf-13b"),
  extract_PPC_data(LLaMA2_hf_7b, "LLaMA2-hf-7b")
) |> 
  # specify levels for 'backend models'
  mutate(backend = case_when(
    backend == "LLaMA2-hf-7b" ~ "L2-7b",
    backend == "LLaMA2-hf-13b" ~ "L2-13b",
    backend == "LLaMA2-hf-70b" ~ "L2-70b",
    backend == "LLaMA2-chat-hf-7b" ~ "L2-chat-7b",
    backend == "LLaMA2-chat-hf-13b" ~ "L2-chat-13b",
    backend == "LLaMA2-chat-hf-70b" ~ "L2-chat-70b",
    T ~ backend
  )) |> 
  mutate(backend = factor(backend, 
                         levels = c(
                           "L2-7b", "L2-13b", "L2-70b", 
                           "L2-chat-7b", "L2-chat-13b", "L2-chat-70b",
                           "GPT", "RSA"))) |> 
  # norm to human data
  mutate(
    `|95%` = `|95%` - observed,
    mean   =  mean  - observed,
    `95%|` = `95%|` - observed
  ) |> 
  mutate(
    response_short = case_when(
      response == "target"     ~ "trgt",
      response == "competitor" ~ "cmpt",
      response == "distractor" ~ "dstr"
    ),
    condition_short = case_when(
      condition == "production"     ~ "prd_",
      condition == "interpretation" ~ "int_"
    ),
    cond_resp_shrt = str_c(condition_short, response_short),
    cond_resp_short = factor(cond_resp_shrt, levels = c("prd_trgt", "prd_cmpt", "prd_dstr", "int_trgt", "int_cmpt", "int_dstr"))
  )
  
# make the plot
vPPC_data_all_models |> 
  filter(model != "RSA") |>
  filter(backend != "GPT") |>
  ggplot() +
  geom_hline(aes(yintercept = 0), color = project_colors[3]) +
  geom_pointrange(aes(x = cond_resp_short, y = mean, ymin = `|95%`, ymax = `95%|`, shape = response, 
                      group = response, color = condition), 
                  position = position_dodge(width = 0.75), size = 0.6, linewidth = 0.8) +
  facet_grid(model ~ backend, scales = "free") +
  ylab("") + xlab("") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.box = "horizontal")

# save plot to file
ggsave("../04-paper/00-pics/vPPC_all_LLaMA_models.pdf", width = 12, height = 6, scale = 1.0)


##################################################
##  inspecting model fit diagnostics
##################################################
model = GPT
model = LLaMA2_hf_13b

extract_main_diagnostics <- function(model) {
  message("Extracting diagnostics for model: ", model$model_name)
  
  get_rhat_ess <- function(fit) {
    summary <- fit$summary()
    tibble(max_rhat = summary |> filter(variable != "at_least_as_extreme") |>  pull(rhat) |> max(),
           min_ess  = summary |> filter(variable != "at_least_as_extreme") |> pull(ess_bulk) |> min())
  }
  
  fits_ordered <- c(
    model$fit_items_prod,
    model$fit_items_inter,
    model$fit_prod_avg_scores,
    model$fit_inter_avg_scores,
    model$fit_prod_avg_probs,
    model$fit_inter_avg_probs,
    model$fit_prod_WTA,
    model$fit_inter_WTA  
  )
  
  results <- map_df(fits_ordered, function(fit) get_rhat_ess(fit))
  
  diagnostics <- tibble(
    model = model$model_name,
    data  = c("item", "item", "condition", "condition", "condition", "condition", "condition", "condition"),
    method = c("---", "---", "avg. scores", "avg. scores", "avg. prob.", "avg. prob.", "avg. WTA", "avg. WTA"),
    condition = c("production", "interpretation", "production", "interpretation", 
                  "production", "interpretation", "production", "interpretation")) |> 
    bind_cols(results)
    
  return(diagnostics)
  
}

main_diagnostics <- 
  rbind(
    extract_main_diagnostics(GPT),
    extract_main_diagnostics(LLaMA2_chat_hf_70b),
    extract_main_diagnostics(LLaMA2_chat_hf_13b),
    extract_main_diagnostics(LLaMA2_chat_hf_7b),
    extract_main_diagnostics(LLaMA2_hf_70b),
    extract_main_diagnostics(LLaMA2_hf_13b),
    extract_main_diagnostics(LLaMA2_hf_7b)
  )

# fits with "LLaMA2-hf-13b" for condition-level data and average-probs aggregation
# has some NAs due to predictions at ceiling for some items, but this is not indicative
# of a systematic failure of the HMC inference (inferences for some parameters are 
# "uncertainty-less" clamped to 1)





