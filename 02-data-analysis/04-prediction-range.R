source('00-premable.r')

plot_predictive_range <- function(p, title = "production", gridsize = 100) {
  
  vertices <- data.frame(
    x = c(0, 1/3, 0.5),
    y = c(1, 1/3, 0.5)
  )
  
  vertices_simplex <- data.frame(
    x = c(0, 0  , 1),
    y = c(0, 1  , 0)
  )
  
  # get predictions for a grid of alpha and epsilon
  grid_ae <- expand.grid(
    alpha = seq(0,7, length.out = gridsize),
    epsilon = seq(0,1, length.out = gridsize)
  ) |> as_tibble() 
  pred_matrix = matrix(0, nrow = nrow(grid_ae), ncol = 3)
  for (i in 1:nrow(pred_matrix)) {
    q <- p^grid_ae$alpha[i] / sum(p^grid_ae$alpha[i])
    pred_matrix[i,] <- (1-grid_ae$epsilon[i]) * q + grid_ae$epsilon[i] * c(1/3,1/3,1/3)
  }
  dimnames(pred_matrix)[[2]] <- tcd
  
  cbind(grid_ae, as_tibble(pred_matrix)) |> 
    ggplot(aes(y = target, x = competitor)) + 
    # simplex
    geom_polygon(data = vertices_simplex, aes(x = x, y = y), fill = "lightgray", alpha = 0.25) +
    # simplex subset that satisfies constraints 
    geom_polygon(data = vertices, aes(x = x, y = y), fill = "gray", alpha = 0.5) +
    # geom_segment(aes(x = 1/3, y = 1/3, xend = p[2], yend = p[1]), color = project_colors[3], size = 1.5) +
    xlim(c(0,1)) +   ylim(c(0,1)) +
    # simplex boundaries
    geom_segment(aes(x= 0, y=1, xend=1, yend=0), color = "firebrick") +
    geom_segment(aes(x= 0, y=0, xend=1, yend=0), color = "firebrick") +
    geom_segment(aes(x= 0, y=0, xend=0, yend=1), color = "firebrick") +
    # boundaries of region satisfying the constraint
    # constraint: competitor is bigger than distractor
    geom_segment(
      aes(x = 0, xend=1/3, y=1, yend=1/3),
      color = "gray"
    ) +
    # constraint: target is bigger than competitor
    geom_segment(
      aes(x = 1/3, xend=0.5, y=1/3, yend=0.5),
      color = "gray"
    ) +
    # predictions for different alpha and epsilon
    geom_line(aes(group = epsilon), color = project_colors[1], alpha = 1) +
    # Laplace point
    geom_point(aes(x = 1/3, y = 1/3), color = "darkorange", size = 3) +
    # vanilla LLM prediction
    geom_point(aes(x = p[2], y = p[1]), color = project_colors[3], size = 3) +
    ggtitle(title) +
    coord_flip()  
}

# p_prod <- prob_prod[1,] / sum(prob_prod[1,])
p_prod <- c(0.9, 0.09, 0.01) # for more stable results with lower gridsize (still)
prediction_range_prod  <- plot_predictive_range(p_prod, "production", gridsize = 100)
prediction_range_prod

# p_inter <- prob_inter[1,] / sum(prob_inter[1,])
p_inter <- c(0.681758393, 0.316123548, 0.002118059)
prediction_range_inter <- plot_predictive_range(p_inter, "interpretation", gridsize = 400)
prediction_range_inter

ggsave(plot = prediction_range_prod,  
       filename = "../04-paper/00-pics/prediction-range-prod.pdf",
       width = 5, height = 5)
ggsave(plot = prediction_range_inter, 
       filename = "../04-paper/00-pics/prediction-range-inter.pdf",
       width = 5, height = 5)

ggsave(plot = prediction_range_prod,  
       filename = "../04-paper/00-pics/prediction-range-prod.png", 
       width = 5, height = 5)
ggsave(plot = prediction_range_inter, 
       filename = "../04-paper/00-pics/prediction-range-inter.png",
       width = 5, height = 5)