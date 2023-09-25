data {
  int<lower=0> n_row;
  matrix[n_row, 3] prob;
  int<lower=0> d[n_row, 3];
  vector[3] uniform;
}

parameters {
  real<lower=0> alpha;
  real<lower=0, upper=1> epsilon;
}

// transformed parameters {
//   matrix[n_row, 3] prob_power;
//   simplex[3] theta[n_row];
//   for (i in 1:n_row) {
//     for (j in 1:3) {
//       prob_power[i,j] = pow(prob[i,j], alpha) + epsilon;
//     }
//     theta[i,] = (prob_power[i,] / sum(prob_power[i,]))';
//     // theta[i,] = (exp(alpha*prob[i,]) / sum(exp(alpha*prob[i,])))';
//   }
// }

transformed parameters {
  matrix[n_row, 3] prob_power;
  simplex[3] theta_pure[n_row];
  simplex[3] theta[n_row];
  for (i in 1:n_row) {
    for (j in 1:3) {
      prob_power[i,j] = pow(prob[i,j], alpha);
    }
    theta_pure[i,] = (prob_power[i,] / sum(prob_power[i,]))';
    theta[i,] = (1-epsilon) * theta_pure[i,] + epsilon * uniform;
  }
}


model {
  alpha ~ lognormal(0.5,1);
  epsilon ~ exponential(5);
  for (i in 1:n_row) {
    d[i,] ~ multinomial(theta[i,]);
  }
}

generated quantities {
  int<lower=0> d_rep[n_row, 3];
  real<upper=1> llh_rep[n_row];
  real<upper=1> llh_dat[n_row];
  int<lower=0,upper=1> at_least_as_extreme;
  for (i in 1:n_row) {
    d_rep[i,]  = multinomial_rng(theta[i,], sum(d[i,]));
    llh_rep[i] = multinomial_lpmf(d_rep[i,] | theta[i,]);
    llh_dat[i] = multinomial_lpmf(d[i,] | theta[i,]);
  }
  at_least_as_extreme = sum(llh_rep) <= sum(llh_dat) ? 1 : 0;
}

