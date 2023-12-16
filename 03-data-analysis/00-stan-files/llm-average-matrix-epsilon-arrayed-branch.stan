data {
  int<lower=0> n_chunk;
  int<lower=0> n_item;
  matrix[n_item, 3] prob[n_chunk];
  int<lower=0> d[n_chunk, 3];
  vector[3] uniform;
}

parameters {
  real<lower=0> alpha;
  real<lower=0,upper=1> epsilon;
}

transformed parameters {
  matrix[n_item, 3] prob_power[n_chunk];
  matrix[n_item, 3] prob_power_normalized[n_chunk];
  simplex[3] theta_pure[n_chunk];
  simplex[3] theta[n_chunk];
  // vector[n_chunk] theta_sum;
  for (k in 1:n_chunk) {
    for (i in 1:n_item) {
      for (j in 1:3) {
        prob_power[k,i,j] = pow(prob[k,i,j] + epsilon, alpha);
      }
      prob_power_normalized[k,i,] = (prob_power[k,i,] / sum(prob_power[k,i,]));
    }  
  }
  for (k in 1:n_chunk) {
    for (j in 1:3) {
      theta_pure[k,j] = mean(prob_power_normalized[k][,j]);
    }
    // theta[k] = (1-epsilon) * theta_pure[k] + epsilon * uniform;
    theta[k] = theta_pure[k];
  }
  // print(prob);
  // print(alpha);
  // print(prob_power);
  // print(prob_power_normalized);
  // print(theta_pure);
  // print(epsilon);
  // print(theta);
  // print(d);
}

model {
  alpha ~ lognormal(0.5,1);
  epsilon ~ exponential(0.5);
  for (k in 1:n_chunk) {
      d[k,] ~ multinomial(theta[k,]);
  }
}

generated quantities {
  int<lower=0> d_rep[n_chunk, 3];
  real<upper=1> llh_rep[n_chunk];
  real<upper=1> llh_dat[n_chunk];
  int<lower=0,upper=1> at_least_as_extreme;
  for (k in 1:n_chunk) {
    d_rep[k,]  = multinomial_rng(theta[k,], sum(d[k,]));
    llh_rep[k] = multinomial_lpmf(d_rep[k,] | theta[k,]);
    llh_dat[k] = multinomial_lpmf(d[k,] | theta[k,]);
  }
  at_least_as_extreme = sum(llh_rep) <= sum(llh_dat) ? 1 : 0;
}

