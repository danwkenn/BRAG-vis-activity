#Simulated benchmarking data:

library(gtools)
library(MASS)
library(MCMCpack)
#Set seed
set.seed(1002)

#Set number of organisations:
n <- 1000
K <- 10
prop_vec <- rdirichlet(1,2 * 0.7^(1:K))
J <- 4
mu_vec <- 2 * mvrnorm(K,mu = rep(0,J),Sigma = diag(J))
Sigma_vec <- lapply(1:K, function(x) riwish(J+1, diag(J)))

X <- matrix(,n,J)
z <- rep(NA,n)
for(i in 1:n){
  z[i] <- sample(size = 1,x = 1:K, prob = prop_vec)
  X[i,] <- mvrnorm(1,mu_vec[z[i],],Sigma_vec[[z[i]]])
}

colnames(X) <- c("Age_scaled","lpFemale","SES_scaled","lpRegional")
plot(as.data.frame(X))
table(z)

#Create response:
noise_var <- 2
y <- 3 + 2 * inv.logit(X[,1]) +  0.1 * X[,2]^2 + (function(x) 2 * (x<0) - 1)(X[,3]) +
  rnorm(n,0,sqrt(noise_var))
#plot(hclust(dist(X)))
par(mfrow = c(2,2))
plot(X[,1],y)
plot(X[,2],y)
plot(X[,3],y)
plot(X[,4],y)

data <- as.data.frame(cbind(y,X))
head(data)
write.csv(file = "benchmarking-sim-data.csv",data,row.names = FALSE)

kfit <- kmeans(centers = 8,x = X)
data <- as.data.frame(data)
test_id <- sample(1:nrow(data),size = round(nrow(data) * 0.2))
test_data <- data[test_id,]
train_data <- data[-test_id,]

library(caret)
library(parallel)
library(doParallel)

#Fit models:
train_control <- trainControl(method = "repeatedcv",number  = 10, repeats = 3)
metric <- "RMSE"
tune_grid <- expand.grid(mtry = 1:4)
cl <- makePSOCKcluster(5)
registerDoParallel(cl)

rf_fit <- train(
  y ~ .,
  data = train_data,
  trControl = train_control,
  method = "rf",
  tuneGrid = tune_grid,
  ntrees = 150
)

saveRDS(
  object = 
    list(
      data = data,
      cluster_allocations = kfit$cluster,
      rf_fit = rf_fit
    ),
  file = "brag-meeting-data.RDS"
)
