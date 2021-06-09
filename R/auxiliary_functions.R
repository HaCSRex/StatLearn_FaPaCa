# generate mboost prediction
predict_result <- function(fit, data, newdata, bl, link = "logit", 
							cols = c("ID", "Status", "FamilyID", "Clinical_Findings")){
  # prediction of each base leaner
  prd <- predict(fit, type = "link", 
                 newdata = newdata, 
                 which = bl)
  
  rsp <- predict(fit, type = "response", 
                 newdata = newdata)

  # logit link function
  if(is.character(link)){
  	link_fun <- make.link(link)$linkinv	
  } else{
  	link_fun <- link
  }
  
  return(data.table(data[, ..cols],
                    offset = attr(prd, "offset"),
                    prd,
                    eta = attr(prd, "offset") + rowSums(prd),
                    response = rsp))
}


glmnet.adalasso <- function(x, y, q, l2_lambda, type = c("conservative", "anticonservative"), family, weighted = TRUE, gamma = 2, ...) {
    require(glmnet)

    # calculate the weights
    if(weighted){
        w0 <- .5*length(y)/sum(y == 0)
        w1 <- .5*length(y)/sum(y == 1)
	    w01 <- ifelse(y == 0, w0, w1)  
    } else{
        w01 <- rep(1, length(y))
    }

    # determine the penal factor
    ridgefit <- glmnet(x, y, alpha = 0, weight = w01, intercept = TRUE, family = family, ...)   
    weight <- as.vector(1/abs(coef(ridgefit, s = l2_lambda))**gamma)[-1]
    pf <- ncol(x) * weight/sum(weight)

    ## fit model
    type <- match.arg(type)
    if (type == "conservative")
        fit <- suppressWarnings(glmnet(x, y, pmax = q, alpha = 1,
                                        weight = w01, family = family, penalty.factor = pf, ...))
    if (type == "anticonservative")
        fit <- glmnet(x, y, dfmax = q - 1, alpha = 1, 
                        weight = w01, family = family, penalty.factor = pf, ...)
    
    ## which coefficients are non-zero?
    selected <- predict(fit, type = "nonzero")
    selected <- selected[[length(selected)]]
    ret <- logical(ncol(x))
    ret[selected] <- TRUE
    names(ret) <- colnames(x)
    ## compute selection paths
    cf <- fit$beta
    sequence <- as.matrix(cf != 0)
    ## return both
    return(list(selected = ret, path = sequence))
}

summarySE <- function(data=NULL, measurevar, groupvars=NULL, na.rm=FALSE,
                      conf.interval=.95, .drop=TRUE) {
    library(plyr)
    
    # New version of length which can handle NA's: if na.rm==T, don't count them
    length2 <- function (x, na.rm=FALSE) {
        if (na.rm) sum(!is.na(x))
        else       length(x)
    }
    
    # This does the summary. For each group's data frame, return a vector with
    # N, mean, and sd
    datac <- ddply(data, groupvars, .drop=.drop,
                   .fun = function(xx, col) {
                       c(N    = length2(xx[[col]], na.rm=na.rm),
                         mean = mean   (xx[[col]], na.rm=na.rm),
                         sd   = sd     (xx[[col]], na.rm=na.rm)
                       )
                   },
                   measurevar
    )
    
    # Rename the "mean" column    
    datac <- rename(datac, c("mean" = measurevar))
    
    datac$se <- datac$sd / sqrt(datac$N)  # Calculate standard error of the mean
    
    # Confidence interval multiplier for standard error
    # Calculate t-statistic for confidence interval: 
    # e.g., if conf.interval is .95, use .975 (above/below), and use df=N-1
    ciMult <- qt(conf.interval/2 + .5, datac$N-1)
    datac$ci <- datac$se * ciMult
    
    return(datac)
}
