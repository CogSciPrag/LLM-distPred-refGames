source('00-premable.r')

d_raw <- read_csv('01-data-raw.csv') |> 
  select(- c(
    "experiment_duration", "experiment_end_time", "experiment_start_time",
    "prolific_pid", "prolific_session_id", "prolific_study_id"  )) |> 
  rename(item = trial) |> 
  mutate(item = factor(item, levels = 0:99))

d_sim <- read_csv('01-simulation-results.csv') |> 
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
  return(c("target", "competitor", "distractor", "distractor")[which(scores == max(scores))])
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

# add 'feature_set' information

d <-  d |>  mutate(feature_set = str_c(trigger_feature, "-", nuisance_feature)) 

## write to file

d |> 
  write_csv(file = "02-data-prepped.csv")


