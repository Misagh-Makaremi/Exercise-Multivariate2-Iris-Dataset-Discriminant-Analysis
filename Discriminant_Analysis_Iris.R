# Load necessary libraries
library(MASS)

# Load iris dataset
data(iris)

# Check the structure of the iris dataset
str(iris)

# Assign the feature columns to variables
X1 <- as.matrix(iris[iris$Species == "setosa", 1:4])
X2 <- as.matrix(iris[iris$Species == "versicolor", 1:4])
X3 <- as.matrix(iris[iris$Species == "virginica", 1:4])

# Create a data frame combining all the data
data <- data.frame(rbind(X1, X2, X3),
                   group = factor(c(rep(1, nrow(X1)), rep(2, nrow(X2)), rep(3, nrow(X3)))))
colnames(data) <- c("M1", "M2", "M3", "M4", "group")

# Plot data (using the first two features for visualization)
plot(data$M1, data$M2, pch = 10)
plot(data$M1, data$M2, col=data$group, pch = 19)

# Mean vectors
m1 <- colMeans(X1)
m2 <- colMeans(X2)
m3 <- colMeans(X3)

m1 <- matrix(m1, ncol=1)
m2 <- matrix(m2, ncol=1)
m3 <- matrix(m3, ncol=1)

# Covariance matrices
S1 <- cov(X1)
S2 <- cov(X2)
S3 <- cov(X3)

# Linear discriminant analysis (LDA)
Spool <- ((nrow(X1)-1)*S1 + (nrow(X2)-1)*S2 + (nrow(X3)-1)*S3) / (nrow(X1) + nrow(X2) + nrow(X3) - 3)

plot(data$M1, data$M2, col=data$group, pch = 19, xlab="M1", ylab="M2")
points(m1[1], m1[2], pch=8, col='black')
points(m2[1], m2[2], pch=8, col='red')
points(m3[1], m3[2], pch=8, col='green')

# Bayesian Discrimination
x <- matrix(c(5.0, 3.5, 1.5, 0.2), ncol=1)
d1 <- t(x-m1) %*% solve(Spool) %*% (x-m1)
d2 <- t(x-m2) %*% solve(Spool) %*% (x-m2)
d3 <- t(x-m3) %*% solve(Spool) %*% (x-m3)
points(x[1], x[2], pch=10, col='blue')

# Prior distribution
p0 <- c(1/3, 1/3, 1/3)

# Posterior distribution
d <- c(d1, d2, d3)
post <- p0 * exp(-d/2) / sum(p0 * exp(-d/2))
print(post)

# Similarity index
l1t <- t(m1) %*% solve(Spool)
c1 <- -1/2 * t(m1) %*% solve(Spool) %*% m1

l2t <- t(m2) %*% solve(Spool)
c2 <- -1/2 * t(m2) %*% solve(Spool) %*% m2

l3t <- t(m3) %*% solve(Spool)
c3 <- -1/2 * t(m3) %*% solve(Spool) %*% m3

print(l1t %*% x + c1)
print(l2t %*% x + c2)
print(l3t %*% x + c3)

# Separation Line
l13 <- l1t - l3t
c13 <- c1 - c3

abline(a=-c13/l13[2], b=-l13[1]/l13[2], col='green', lty=2)
abline(a=-c13/l13[2], b=-l13[1]/l13[2], col='black', lty=3)

l23 <- l2t - l3t
c23 <- c2 - c3

abline(a=-c23/l23[2], b=-l23[1]/l23[2], col='red', lty=2)
abline(a=-c23/l23[2], b=-l23[1]/l23[2], col='green', lty=3)

# Quadratic discriminant analysis (QDA)
D1 <- t(x-m1) %*% solve(S1) %*% (x-m1) + log(det(S1))
D2 <- t(x-m2) %*% solve(S2) %*% (x-m2) + log(det(S2))
D3 <- t(x-m3) %*% solve(S3) %*% (x-m3) + log(det(S3))

# Posterior distribution
D <- c(D1, D2, D3)
post <- p0 * exp(-D/2) / sum(p0 * exp(-D/2))
print(post)

# Plotting Quadratic lines (using first two features for visualization)
plot(data$M1, data$M2, col=data$group, pch = 19, xlab="M1", ylab="M2")
points(m1[1], m1[2], pch=8, col='black')
points(m2[1], m2[2], pch=8, col='red')
points(m3[1], m3[2], pch=8, col='green')

# Separating Line for discriminating Population 1 and 3
x1 <- seq(min(data$M1) - 1, max(data$M1) + 1, length.out = 100)
xf <- function(x1, x2) {
  d1 <- t(matrix(c(x1, x2, mean(as.numeric(data$M3)), mean(as.numeric(data$M4))), ncol = 1) - m1) %*% solve(S1) %*% (matrix(c(x1, x2, mean(as.numeric(data$M3)), mean(as.numeric(data$M4))), ncol = 1) - m1) + log(det(S1))
  d3 <- t(matrix(c(x1, x2, mean(as.numeric(data$M3)), mean(as.numeric(data$M4))), ncol = 1) - m3) %*% solve(S3) %*% (matrix(c(x1, x2, mean(as.numeric(data$M3)), mean(as.numeric(data$M4))), ncol = 1) - m3) + log(det(S3))
  return(d1 - d3)
}

xs <- function(x1) {
  tryCatch({
    uniroot(function(x2) xf(x1, x2), lower = min(data$M2) - 1, upper = max(data$M2) + 1)$root
  }, error = function(e) {
    NA
  })
}

x1_seq <- seq(min(data$M1) - 1, max(data$M1) + 1, length.out = 100)
x2_values <- sapply(x1_seq, xs)

valid_idx <- !is.na(x2_values)
x1_valid <- x1_seq[valid_idx]
x2_valid <- x2_values[valid_idx]

lines(x1_valid, x2_valid, type = 'l', lty = 'dashed', col = 'green')

abline(a=-c13/l13[2], b=-l13[1]/l13[2], col='green', lty=2)
abline(a=-c13/l13[2], b=-l13[1]/l13[2], col='black', lty=3)

l23 <- l2t - l3t
c23 <- c2 - c3

abline(a=-c23/l23[2], b=-l23[1]/l23[2], col='red', lty=2)
abline(a=-c23/l23[2], b=-l23[1]/l23[2], col='green', lty=3)

# Using packages for QDA
model <- qda(group ~ M1 + M2 + M3 + M4, data)
str(model)

# Decision plot function for 2D projection (using first two features for visualization)
decisionplot <- function(model, data, class = NULL, predict_type = "class",
                         resolution = 100, showgrid = TRUE, ...) {
  
  if(!is.null(class)) cl <- data[,class] else cl <- 1
  data <- data[,1:4]
  k <- length(unique(cl))
  
  # Plot using the first two features
  plot(data[,1:2], col = as.integer(cl)+1L, pch = as.integer(cl)+1L, ...)
  
  # Make grid
  r <- sapply(data[,1:2], range, na.rm = TRUE)
  xs <- seq(r[1,1], r[2,1], length.out = resolution)
  ys <- seq(r[1,2], r[2,2], length.out = resolution)
  g <- expand.grid(M1 = xs, M2 = ys)
  
  # Predict using all four features
  g_full <- cbind(g, M3 = mean(as.numeric(data$M3)), M4 = mean(as.numeric(data$M4)))
  p <- predict(model, g_full, type = predict_type)
  if(is.list(p)) p <- p$class
  p <- as.factor(p)
  
  if(showgrid) points(g, col = as.integer(p)+1L, pch = ".")
  
  z <- matrix(as.integer(p), nrow = resolution, byrow = TRUE)
  contour(xs, ys, z, add = TRUE, drawlabels = FALSE,
          lwd = 2, levels = (1:(k-1))+.5)
  
  invisible(z)
}

# Use the decisionplot function to visualize the QDA decision boundaries
decisionplot(model, data, class = "group")

####################################################################
# Using packages for QDA
model <- qda(group ~ M1 + M2, data)
str(model)

# Decision plot function
decisionplot <- function(model, data, class = NULL, predict_type = "class",
                         resolution = 100, showgrid = TRUE, ...) {
  
  if(!is.null(class)) cl <- data[,class] else cl <- 1
  data <- data[,1:2]
  k <- length(unique(cl))
  
  plot(data, col = as.integer(cl)+1L, pch = as.integer(cl)+1L, ...)
  
  # make grid
  r <- sapply(data, range, na.rm = TRUE)
  xs <- seq(r[1,1], r[2,1], length.out = resolution)
  ys <- seq(r[1,2], r[2,2], length.out = resolution)
  g <- cbind(rep(xs, each=resolution), rep(ys, time = resolution))
  colnames(g) <- colnames(r)
  g <- as.data.frame(g)
  
  p <- predict(model, g, type = predict_type)
  if(is.list(p)) p <- p$class
  p <- as.factor(p)
  
  if(showgrid) points(g, col = as.integer(p)+1L, pch = ".")
  
  z <- matrix(as.integer(p), nrow = resolution, byrow = TRUE)
  contour(xs, ys, z, add = TRUE, drawlabels = FALSE,
          lwd = 2, levels = (1:(k-1))+.5)
  
  invisible(z)
}

decisionplot(model, data, class = "group")
