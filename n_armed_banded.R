library(tidyverse)

EpsilonGreedy <- function(n.trial, n.rep, epsilon) {
  # Args:
  #   n.trial:
  #   n.rep:
  #   epsilon: double
  #
  # Returns:
  #   simulation result including percentages of the optimal action and return taken in every trial, and epsilon.
  return.history <- matrix(0, nrow = n.trial, ncol = n.rep)
  optimized.history <- return.history
  for (i.rep in 1:n.rep) {
    q.true <- rnorm(10)
    q.estimatedimated <- rep(0, 10)
    q.cumulatedulated <- rep(0, 10)
    action.count <-  rep(0, 10)
    optimized.id <- which.max(q.true)
    for (i.trial in 1:n.trial) {
      if (runif(1) < epsilon) {
        action <- sample(1:10, 1)
      } else {
        action <- which.max(q.estimatedimated)
      }
      return.history[i.trial, i.rep] <- rnorm(1) + q.true[action]
      optimized.history[i.trial, i.rep] <- action == optimized.id
      action.count[action] <- action.count[action] + 1
      q.cumulatedulated[action] <- q.cumulatedulated[action] + return.history[i.trial, i.rep]
      q.estimatedimated[action] <- q.cumulatedulated[action] / action.count[action]
    }
  }
  return(data.frame(optimized = rowMeans(optimized.history), return = rowMeans(return.history), epsilon = epsilon))
}

EpsilonGreedyWithMultipleConditions <- function(args.matrix) {
  n.row <- dim(args.matrix)[1]
  return(map(1:n.row, ~EpsilonGreedy(args.matrix[.x, 1], args.matrix[.x, 2], args.matrix[.x, 3])))
}

PlotEpsilonGreedyResult <- function(results) {
  result.size <- length(results)
  colors <- rainbow(result.size)
  epsilons <- map_dbl(results, function(x) {
    return(unique(x$epsilon))
  })
  legends <- paste0("epsilon=", epsilons)
  par(mfrow = c(2, 1))
  plot(results[[1]]$return, type = 'l', col = colors[1], xlab = "Play", ylab = "average reward")
  for (i.result in 2:result.size) {
    lines(results[[i.result]]$return, type = 'l', col = colors[i.result])
  }
  legend("bottomright", legends, col = colors, lty = rep(1, result.size))
  plot(results[[1]]$optimized, type = 'l', col = colors[1], xlab = "Play", ylab = "% optimal action")
  for (i.result in 2:result.size) {
    lines(results[[i.result]]$optimized, type = 'l', col = colors[i.result])
  }
  legend("bottomright", legends, col = colors, lty = rep(1, result.size))
}

#SoftMax#

Softmax <- function(n.trial, n.rep, temperature) {
  return.history <- matrix(0, nrow = n.trial, ncol = n.rep)
  optimized.history <- return.history
    for (i.rep in 1:n.rep) {
      q.true <- rnorm(10)
      q.estimated <- rep(0,10)
      q.cumulated <- rep(0,10)
      action.count <- rep(0,10)
      optimized.id <- which.max(q.true)
      t <- temperature
      for(i.trial in 1:n.trial) {
        action <- sample(1:10, 1, prob = exp(q.estimated / t) / sum(exp(q.estimated / t)))
        return.history[i.trial, i.rep] <- rnorm(1) + q.true[action]
        optimized.history[i.trial,i.rep] <- action == optimized.id
        action.count[action] <- action.count[action] + 1
        q.cumulated[action] <- q.cumulated[action] + return.history[i.trial, i.rep]
        q.estimated[action] <- q.cumulated[action] / action.count[action]
        t <- max(0.001, 0.995 * t)
      }
    }
  return(data.frame(optimized = rowMeans(optimized.history), return = rowMeans(return.history)))
}

SampleTrainForSoftMax <- function() {
  case1 <- Softmax(1000, 2000, 1)
  return(c(case1 = case1))
}

PlotSoftmaxResult <- function(results) {
  par(mfrow = c(2, 1))
  plot(results$case1.return, type = 'l', xlab = "play", ylab = "average reward")
  plot(results$case1.optimized, type = 'l', xlab = "play", ylab = "% optimal action")
}

#epsilon greedy optimistic#

epGreedyOpt <- function(n.trial, n.rep, epsilon, optimistic) {
  return.history <- array(0, c(n.trial, n.rep, 10));
  optimized.history <- matrix(0, nrow = n.trial, ncol = n.rep)
  alpha <- 0.1
  for (i.rep in 1:n.rep) {
    q.true <- rnorm(10);
    q.estimated <- rep(optimistic, 10)
    q.cumulated <- rep(0, 10)
    action.count <- rep(0, 10)
    optimized.id <- which.max(q.true)
    for (i.trial in 1:n.trial) {
      if (runif(1) < epsilon) {
        action <- sample(1:10,1)
      } else {
        action <- which.max(q.estimated)
      }
      return.history[i.trial, i.rep,action] <- rnorm(1) + q.true[action]
      optimized.history[i.trial, i.rep] <- action == optimized.id
      action.count[action] <- action.count[action] + 1;
      q.cumulated[action] <- q.cumulated[action] + return.history[i.trial, i.rep, action]
      ret.weight <- (alpha * (1 - alpha)^rev(0:(action.count[action] - 1))) * return.history[which(return.history[, i.rep, action] != 0), i.rep, action]
      q.estimated[action] <- ((1 - alpha)^action.count[action]) * optimistic + sum(ret.weight)
    }
  }
  return(data.frame(optimized = rowMeans(optimized.history), return = rowMeans(return.history)))
}

SampleTrainWithGreedyOptimistic <- function() {
  case1 <- epGreedyOpt(1000, 2000, 0, 5)
  case2 <- epGreedyOpt(1000, 2000, 0.1, 0)
  return(c(case1 = case1, case2 = case2))
}

PlotGreedyOptimisticResult <- function(results) {
  plot(results$case1.optimized, type = 'l', xlab = "play", ylab = "% optimal action", col = 'red', ylim = c(0, 1.05))
  lines(results$case2.optimized, type='l', xlab = "play", ylab = "% optimal action", col = 'blue', ylim = c(0, 1.05))
  legend("bottomright", c("Q=5,epsilon=0", "Q=0,epsilon=0.1"), col = c("red", "blue"), lty = c(1, 1))
}

##reinforce comparison##

#p_t+1(a_t)=p_t(a_t)+beta*(r_t-r.bar_t)
#r.bar_t+1=r.bar_t+alpha*(r_t-r.bar_t)

ReinComp <- function(n.trial, n.rep, optimistic) {
  return.history <- matrix(0, nrow = n.trial, ncol = n.rep)
  optimized.history <- return.history
  alpha <- 0.1
  beta <- 0.1
  for (i.rep in 1:n.rep) {
    q.true <- rnorm(10)
    ret.ref <- optimistic
    pt <- rep(0, 10);
    action.count <- rep(0, 10)
    optimized.id <- which.max(q.true)
    for (i.trial in 1:n.trial) {
      action <- sample(1:10, 1, prob = exp(pt) / sum(exp(pt)))
      return.history[i.trial, i.rep] <- rnorm(1) + q.true[action]
      optimized.history[i.trial, i.rep] <- action == optimized.id
      action.count[action] <- action.count[action] + 1
      pt[action] <- pt[action] + beta * (return.history[i.trial, i.rep] - ret.ref)
      ret.ref <- ret.ref + alpha * (return.history[i.trial, i.rep] - ret.ref)
    }
  }
  return(data.frame(optimized = rowMeans(optimized.history), return = rowMeans(return.history)))
}

SampleTrainWithRainComp <- function() {
  case1 <- ReinComp(1000, 2000, 5)
  return(c(case1 = case1))
}

PlotRainCompResult <- function(results) {
  plot(results$case1.optimized, type = 'l', xlab = "play", ylab = "% optimal action", col = "black", ylim = c(0, 1.05))
}

##pursuit methods##

PursuitMethod <- function(n.trial, n.rep, optimistic) {
  return.history <- array(0, c(n.trial, n.rep, 10))
  optimized.history <- matrix(0, nrow = n.trial, ncol = n.rep)
  beta <- 0.01
  for (i.rep in 1:n.rep) {
    q.true <- rnorm(10)
    q.estimated <- rep(optimistic, 10)
    pie <- rep(0.1, 10)
    q.cumulated <- rep(0, 10)
    action.count <- rep(0, 10)
    optimized.id <- which.max(q.true)
    for (i.trial in 1:n.trial) {
      action <- sample(1:10, 1, prob = pie)
      return.history[i.trial, i.rep, action] <- rnorm(1) + q.true[action]
      optimized.history[i.trial, i.rep] <- action == optimized.id
      action.count[action] <- action.count[action] + 1
      q.cumulated[action] <- q.cumulated[action] + return.history[i.trial, i.rep, action]
      alpha <- 1 / action.count[action]
      ret.weight <- (alpha * (1 - alpha)^rev(0:(action.count[action] - 1))) * return.history[which(return.history[, i.rep, action] != 0), i.rep, action]
      q.estimated[action] <- ((1 - alpha)^action.count[action]) * optimistic + sum(ret.weight)
      maxQ <- which.max(q.estimated)
      pie[maxQ] <- pie[maxQ] + beta * (1 - pie[maxQ])
      pie[-maxQ] <- pie[-maxQ] + beta * (0 - pie[-maxQ])
    }
  }
  return(data.frame(optimized = rowMeans(optimized.history), return = rowMeans(return.history)))
}

SampleTrainWithPursuitMethod <- function() {
  case1 <- PursuitMethod(1000, 2000, 0)
  return(c(case1 = case1))
}

PlotPursuitMethodResult(results) {
  plot(result$case1.optimized, type = 'l', xlab = "play", ylab = "% optimal action", col = "black", ylim = c(0, 1.05))
}

