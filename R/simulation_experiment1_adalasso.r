## load required libraries ####
if(!require(data.table)) install.packages("data.table")
if(!require(purrr)) install.packages("purrr")
if(!require(magrittr)) install.packages("magrittr")
if(!require(caret)) install.packages("caret")
if(!require(glmnet)) install.packages("glmnet")
if(!require(doParallel)) install.packages("doParallel")
if(!require(foreach)) install.packages("foreach")

source("./data_simulator.R")

set.seed(119752361)

## generate biniary data ####
simulated_data  <- lapply(1:100, function(x){
  simdat <- binary_data_generator(N = 50, P = 500, p_ref = 10, 
                                  tau = -9, sigma = 10, rho = runif(10, .05, .95))
  simdat <- simdat %$% data.table(as.factor(y), x)
  setnames(simdat, c("y", paste0("X", 1:500)))
  
  return(simdat)
})

## create repated stratified 4-fold CV ####
set.seed(873)
cv_ind <- map(simulated_data, function(x) createDataPartition(x$y, p = .7, list = FALSE))

## standardise the training and test data ####
vars <- paste0("X", 1:500)

simulated_data_rescaled <- map2(cv_ind, simulated_data, 
                               function(x, y){ 
                                 dat <- copy(y)[, c(vars) := lapply(.SD, function(a) (a - mean(a[x]))/sd(a[x])), .SDcols = c(vars)]
                                 return(list(train = dat[x], test = dat[-x]))
                               })

## seperate the training and test sets ####
simulated_data_train <- map(simulated_data_rescaled,
                            function(x) return(x$train))
simulated_data_test <- map(simulated_data_rescaled,
                           function(x) return(x$test))

## make cluster ####
cl <- makeCluster(detectCores())
registerDoParallel(cl)

## perform adapative lasso ####
adalasso_fit <- foreach(i = 1:100, .packages = c("glmnet", "caret")) %dopar% {
  
  # convert data to matrix
  xmat <- as.matrix(simulated_data_train[[i]][, -1])
  y <- simulated_data_train[[i]]$y
  
  # assign weights
  w0 <- .5*length(y)/sum(y == 0)
  w1 <- .5*length(y)/sum(y == 1)
  w01 <- ifelse(y == 0, w0, w1)
  
  foldid <- createFolds(as.factor(y), k = 5, list = FALSE)
  
  # estimate penalty factor (adpative weights) with ridge regression
  l2fit <- cv.glmnet(x = xmat, y = y,
                             alpha = 0, 
                             nlambda = 500, 
                             weights = w01, 
                             family = "binomial",
                             type.measure = "deviance",
                              foldid = foldid)
  
  # obtain penalty factor
  weight <- as.vector(1/abs(coef(l2fit, s = "lambda.min"))^2)[-1]
  
  # fit adalasso
  cv.glmnet(x = xmat, y = y,
                     alpha = 1, 
                     penalty.factor = weight,
                     weights = w01,
                     nlambda = 500, 
                     family = "binomial",
                    foldid = foldid)
  
}

## stop cluster ####
stopCluster(cl)

## extracts selected features ####
sel_vars <- map2(adalasso_fit, 1:100,
                 function(x, y){
                   coef <- coef(x, s = "lambda.min")
                   return(data.table(Simulation = y, 
                                     variable = rownames(coef)[c(summary(coef)$i)[-1]]))
                 }) %>%
  rbindlist

## obtain test result ####
adalasso_testpred <- map2(adalasso_fit, simulated_data_test, 
                          function(x, y){
                            data.table(y = as.factor(y$y), 
                                       response = as.vector(predict(x, s = x$lambda.min, 
                                                                    newx = as.matrix(y[, -1]),
                                                                    type = "response")),
                                       predict = factor(predict(x, 
                                                                s = x$lambda.min, 
                                                                newx = as.matrix(y[, -1]),
                                                                type = "class"),
                                                        levels = c(0, 1)))
                          }) %>%
  rbindlist