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
simulated_data <- lapply(1:100, function(x){
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

## fit glmboost ####
simdata_glmboost_list <- foreach(i = 1:100, .packages = "mboost") %dopar% { 

                            # setting initial number of iteration
                            iter <- 50
                            
                            # fit the logistic boosting model
                            fit <- glmboost(y ~ ., 
                                                   data = simulated_data_train[[i]],
                                                   family = Binomial(type = "adaboost", link = "logit"), 
                                                   control = boost_control(mstop = iter,
                                                                           nu = .1))
                            
                            # adjust the number of iteration using AIC (logit link only)
                            aic <- AIC(fit, method = "classical")
                            
                            while(iter <= mstop(aic) * 1.2 && iter < 1000){
                              iter <- iter + 50
                              mstop(fit) <- iter
                              aic <- AIC(fit, method = "classical")
                            }
                            
                            # set a resampling scheme.
                            rsmp <- cv(model.weights(fit), 
                                        type = "bootstrap", 
                                        strata = fit$response)
                            
                            # using resampling to search for the optimal iteration.
                            mboost_cvrisk <- cvrisk(fit, 
                                                    folds = rsmp, 
                                                    mc.cores = detectCores())
                            
                            # obtain the optimal model according to mstop
                            mstop(fit) <- mstop(mboost_cvrisk)
                            
                            # return fitted model
                            fit
}

## stop cluster ####
stopCluster(cl)

## extract variable importance of each fold ####
simdata_glmboost_varimp <- map(simdata_glmboost_list, 
  function(x){
    varimp_logit <- as.data.table(varimp(x))[order(reduction, decreasing = TRUE)]
    setnames(varimp_logit, "variable", "Variable")
    varimp_logit[, `:=` (blearner  = as.character(blearner),
      Variable  = as.character(Variable))]
    return(varimp_logit)
    }) %>%
  map2(1:100,
      function(x, y){
        data.table(Simulation = y, x)
      }) %>% 
  rbindlist

## summarise the results of all folds ####
simdata_glmboost_varimp_summary <- simdata_glmboost_varimp[reduction > 0, 
                                            .(mrd = mean(reduction), msf = mean(selfreq), selfreq = .N), 
                                            by = .(Variable, blearner, Simulation)][order(-selfreq)]

## select base-learners for prediction ####
bl_sel <- simdata_glmboost_varimp_summary[, blearner]

## obtain test result ####
simdata_glmboost_testpred <- pmap(list(simdata_glmboost_list, simulated_data_test, 1:100),
                            function(x,y,z) data.table(Simultaion = z,
                                                        y = y$y,
                                                        response = as.vector(predict(x, newdata = y, type = "response"))
                                                        )) %>%
  rbindlist

## classify the test prediction ####
simdata_glmboost_testpred[, predict := factor(ifelse(response > .5, 1, 0))]
