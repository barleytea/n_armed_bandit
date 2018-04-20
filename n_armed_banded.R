library(tidyverse)

EpsilonGreedy <- function(n.trial, n.rep, epsilon) {
  # Args:
  #   n.trial: the number of trials
  #   n.rep: the number of banded tasks in every trials
  #   epsilon: (positive double) probability of taking random action
  #
  # Returns:
  #   simulation result including percentages of the optimal action and return taken in every trial, and epsilon.
  return.history <- matrix(0, nrow = n.trial, ncol = n.rep)
  optimized.history <- return.history
  for (i.rep in 1:n.rep) {
    q.true <- rnorm(10)
    q.estimated <- rep(0, 10)
    q.cumulated <- rep(0, 10)
    action.count <-  rep(0, 10)
    optimized.id <- which.max(q.true)
    for (i.trial in 1:n.trial) {
      if (runif(1) < epsilon) {
        action <- sample(1:10, 1)
      } else {
        action <- which.max(q.estimated)
      }
      return.history[i.trial, i.rep] <- rnorm(1) + q.true[action]
      optimized.history[i.trial, i.rep] <- action == optimized.id
      action.count[action] <- action.count[action] + 1
      q.cumulated[action] <- q.cumulated[action] + return.history[i.trial, i.rep]
      q.estimated[action] <- q.cumulated[action] / action.count[action]
    }
  }
  return(data.frame(optimized = rowMeans(optimized.history), return = rowMeans(return.history), epsilon = epsilon))
}

EpsilonGreedyWithMultipleConditions <- function(args.matrix) {
  # Args:
  #   args.matrix: (matrix)
  #     column1: (positive integer) n.trials
  #     column2: (positive integer) n.rep
  #     column3: (positive double) epsiron
  #
  # Returns:
  #   list of Epsilon Greedy results
  n.row <- dim(args.matrix)[1]
  return(map(1:n.row, ~EpsilonGreedy(args.matrix[.x, 1], args.matrix[.x, 2], args.matrix[.x, 3])))
}

PlotEpsilonGreedyResult <- function(results) {
  # Args:
  #   results: (list) return of EpsilonGreedyWithMultipleConditions
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

# SoftMax

Softmax <- function(n.trial, n.rep, temperature) {
  # Args:
  #   n.trial: the number of trials
  #   n.rep: the number of banded tasks in every trials
  #   epsilon: (positive double) temperature
  #            high temperature enables agents to take random actions.
  #
  # Returns:
  #   simulation result including percentages of the optimal action and return taken in every trial, and temperature.
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
  return(data.frame(optimized = rowMeans(optimized.history), return = rowMeans(return.history), temperature = temperature))
}

SoftMaxWithMultipleConditions <- function(args.matrix) {
  # Args:
  #   args.matrix: (matrix)
  #     column1: (integer) n.trials
  #     column2: (integer) n.rep
  #     column3: (positive double) temperature
  #
  # Returns:
  #   list of Softmax results
  n.row <- dim(args.matrix)[1]
  return(map(1:n.row, Softmax(args.matrix[.x, 1], args.matrix[.x, 2], args.matrix[.x, 3])))
}

PlotSoftmaxResult <- function(results) {
  # Args:
  #   results: (list) return of SoftMaxWithMultipleConditions
  result.size <- length(results)
  colors <- rainbow(result.size)
  temperatures <- map_dbl(results, function(x) {
    return(unique(x$temperature))
  })
  legends <- paste0("temperature=", temperatures)
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

# epsilon greedy optimistic#

EpsilonGreedyOptimistic <- function(n.trial, n.rep, epsilon, optimistic) {
  # Args:
  #   n.trial: the number of trials
  #   n.rep: the number of banded tasks in every trials
  #   epsilon: (positive double) temperature
  #            high temperature enables agents to take random actions.
  #   optimistic: (positive number) optimistic
  #
  # Returns:
  #   simulation result including percentages of the optimal action and return taken in every trial, epsilon and optimistic.
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
  return(data.frame(optimized = rowMeans(optimized.history), return = rowMeans(return.history), epsilon = epsilon, optimistic = optimistic))
}

GreedyOptimisticWithMultipleConditions <- function(args.matrix) {
  # Args:
  #   args.matrix: (matrix)
  #     column1: (integer) n.trials
  #     column2: (integer) n.rep
  #     column3: (positive double) epsilon
  #     column4: (positive number) optimistic
  #
  # Returns:
  #   list of Epsilon Greedy Optimistic results
  n.row <- dim(args.matrix)[1]
  return(map(1:n.row, EpsilonGreedyOptimistic(args.matrix[.x, 1], args.matrix[.x, 2], args.matrix[.x, 3], args.matrix[.x, 4])))
}

PlotGreedyOptimisticResult <- function(results) {
  # Args:
  #   results: (list) return of GreedyOptimisticWithMultipleConditions
  result.size <- length(results)
  colors <- rainbow(result.size)
  epsilons <- map_dbl(results, function(x) {
    return(unique(x$temperature))
  })
  optimistics <- map_dbl(results, function(x) {
    return(unique(x$optimistic))
  })
  legends <- paste0("epsilon=", epsilons, ", optimistic=", optimistics)
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

# reinforce comparison

ReinComp <- function(n.trial, n.rep, optimistic) {
  # Args:
  #   n.trial: the number of trials
  #   n.rep: the number of banded tasks in every trials
  #   optimistic: (positive number) optimistic
  #
  # Returns:
  #   simulation result including percentages of the optimal action and return taken in every trial, and optimistic.
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
  return(data.frame(optimized = rowMeans(optimized.history), return = rowMeans(return.history), optimistic = optimistic))
}

RainCompWithMultipleConditions <- function(args.matrix) {
  # Args:
  #   args.matrix: (matrix)
  #     column1: (integer) n.trials
  #     column2: (integer) n.rep
  #     column3: (positive number) optimistic
  #
  # Returns:
  #   list of ReinforcementComparison results
  n.row <- dim(args.matrix)[1]
  return(map(1:n.row, RainComp(args.matrix[.x, 1], args.matrix[.x, 2], args.matrix[.x, 3])))
}

PlotRainCompResult <- function(results) {
  # Args:
  #   results: (list) return of RainCompWithMultipleConditions
  result.size <- length(results)
  colors <- rainbow(result.size)
  optimistics <- map_dbl(results, function(x) {
    return(unique(x$optimistic))
  })
  legends <- paste0("optimistic=", optimistics)
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

# pursuit methods

PursuitMethod <- function(n.trial, n.rep, optimistic) {
  # Args:
  #   n.trial: the number of trials
  #   n.rep: the number of banded tasks in every trials
  #   optimistic: (positive number) optimistic
  #
  # Returns:
  #   simulation result including percentages of the optimal action and return taken in every trial, and optimistic.
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
  return(data.frame(optimized = rowMeans(optimized.history), return = rowMeans(return.history), optimistic = optimistic))
}

PursuitMethodWithMultipleConditions <- function(args.matrix) {
  # Args:
  #   args.matrix: (matrix)
  #     column1: (integer) n.trials
  #     column2: (integer) n.rep
  #     column3: (positive number) optimistic
  #
  # Returns:
  #   list of Pursuit Method results
  n.row <- dim(args.matrix)[1]
  return(map(1:n.row, PursuitMethod(args.matrix[.x, 1], args.matrix[.x, 2], args.matrix[.x, 3])))
}

PlotPursuitMethodResult(results) {
  # Args:
  #   results: (list) return of PursuitMethodWithMultipleConditions
  result.size <- length(results)
  colors <- rainbow(result.size)
  optimistics <- map_dbl(results, function(x) {
    return(unique(x$optimistic))
  })
  legends <- paste0("optimistic=", optimistics)
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

