## load required libraries ####
if(!require(data.table)) install.packages("data.table")
if(!require(purrr)) install.packages("purrr")
if(!require(magrittr)) install.packages("magrittr")
if(!require(caret)) install.packages("caret")
if(!require(glmnet)) install.packages("glmnet")
if(!require(doParallel)) install.packages("doParallel")
if(!require(foreach)) install.packages("foreach")
if(!require(stabs)) install.packages("stabs")

source("./data_simulator.R")
source("./auxiliary_functions.R")

## stability selection function with adalasso ####
run_cpss <- function(data, q, PFER){
    # stratified subsampling
    stabs_rsmp <- stabs::subsample(rep(1, nrow(data$x)), B = 50, strata = as.factor(data$y))
    
    simdata_stabsel <- stabs::stabsel(x = data$x, y = data$y,
                                      fitfun = glmnet.adalasso,
                                      args.fitfun = list(type = "conservative", family = "binomial", standardize = FALSE, 
                                                         l2_lambda = 10, weighted = TRUE, gamma = 2),
                                      sampling.type = "SS",
                                      assumption = "unimod",
                                      B = 50,   
                                      folds  = stabs_rsmp,
                                      q = q, PFER = PFER)
    
    return(list(data = data, result = simdata_stabsel))
}

## senarios with various N ####

# generate data with various N 
N <- rep(c(30, 50, 100), each = 100)

set.seed(39374)
simdata_N <- map(N, binary_data_generator, P = 500, p_ref = 10, tau = -9, 
                                 sigma = 10, rho = runif(10, .05, .95))

# perform stability selection with parallisation
cl <- parallel::makeCluster(detectCores())
doParallel::registerDoParallel(cl)

simulation_N <- foreach(i = 1:300) %dopar% {
  run_cpss(simdata_N[[i]], 10, 2)
}

parallel::stopCluster(cl)

## senarios with various P ####

# generate data with various P
P <- rep(c(100, 500, 5000), each = 100)

set.seed(44234374)
simdata_P <- map(P, binary_data_generator, N = 50, p_ref = 10, tau = -9, 
                 sigma = 10, rho = runif(10, .05, .95))

# perform stability selection with parallisation
cl <- parallel::makeCluster(detectCores())
doParallel::registerDoParallel(cl)

simulation_P <- foreach(i = 1:300) %dopar% {
  run_cpss(simdata_P[[i]], 10, 2)
}

parallel::stopCluster(cl)

## senarios with various pref ####

# generate data with various pref
pref <- rep(c(2, 5, 10, 20), each = 100)

set.seed(123474)
simdata_pref <- map(pref, binary_data_generator, N = 50, P = 500, tau = -9, 
                 sigma = 10, rho = runif(10, .05, .95))

# perform stability selection with parallisation
cl <- parallel::makeCluster(detectCores())
doParallel::registerDoParallel(cl)

simulation_pref <- foreach(i = 1:400) %dopar% {
  run_cpss(simdata_pref[[i]], 10, 2)
}

parallel::stopCluster(cl)

## senarios with various tau ####

# generate data with various tau
tau <- rep(c(0, -2, -4, -6), each = 100)

# generate binary data
set.seed(964896)
simdata_tau <- map(tau, binary_data_generator, N = 50, P = 500, p_ref = 10, 
                    sigma = 10, rho = runif(10, .05, .95))

# perform stability selection with parallisation
cl <- parallel::makeCluster(detectCores())
doParallel::registerDoParallel(cl)

simulation_tau <- foreach(i = seq_along(tau)) %dopar% {
  run_cpss(simdata_tau[[i]], 10, 2)
}

parallel::stopCluster(cl)

## senarios with various rho ####

# generate data with various rho
rho <- rep(c(.1, .3, .5, .7), each = 100)

# generate binary data
set.seed(6418241)
simdata_rho <- map(rho, binary_data_generator, N = 50, P = 500, p_ref = 10, 
                   tau = -9, sigma = 10)

# perform stability selection with parallisation
cl <- parallel::makeCluster(detectCores())
doParallel::registerDoParallel(cl)

simulation_rho <- foreach(i = seq_along(rho)) %dopar% {
  run_cpss(simdata_rho[[i]], 10, 2)
}

parallel::stopCluster(cl)
