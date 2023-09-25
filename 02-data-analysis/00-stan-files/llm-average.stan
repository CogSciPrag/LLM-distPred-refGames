data {
  int<lower=0> N;
  vector[3] prob;
  int d[N];
}

parameters {
  real<lower=0> alpha;
}

transformed parameters {
  vector[3] prob_power;
  simplex[3] theta;
  for (i in 1:3) {
    prob_power[i] = pow(prob[i], alpha);
  }
  theta = prob_power / sum(prob_power);
  // theta = exp(alpha*prob) / sum(exp(alpha*prob));
}

model {
  alpha ~ lognormal(0.5,1);
  for (i in 1:N) {
    d[i] ~ categorical(theta);
  }
}

generated quantities {
}
