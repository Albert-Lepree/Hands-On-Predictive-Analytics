# install_github("gedeck/mlba/mlba", force=TRUE)


# install.packages("caret")
library(caret)
# Load Libraries
library(devtools)
library(ggplot2)
library(gridExtra)
library(mlba)
library(reshape2)

library(neuralnet)
# install.packages("rpart")
library(rpart)

# Custom RMSE function
rmse <- function(pred, actual) {
  sqrt(mean((pred - actual)^2))
}

# Get housing dataset
housing.df <- mlba::BostonHousing

# Check column names
colnames(housing.df)

print(housing.df)

# Plot percentage of lower status of the population (LSTAT) vs Median value of homes (MEDV)
g1 <- ggplot(housing.df) + geom_point(aes(x=LSTAT, y=MEDV), colour="navy", alpha=0.5)

# Compute meanMEDV per CHAS (Charles River dummy variable)
MEDV.per.CHAS <- aggregate(housing.df$MEDV, by=list(housing.df$CHAS), FUN=mean)
names(MEDV.per.CHAS) <- c("CHAS", "MeanMEDV")
MEDV.per.CHAS$CHAS <- factor(MEDV.per.CHAS$CHAS)

# Plot Median value by near the Charles river (CHAS) or not
g2 <- ggplot(MEDV.per.CHAS) + geom_bar(aes(x=CHAS, y=MeanMEDV, fill=CHAS), stat="identity")

# Distribution of median home values (MEDV)
g3 <- ggplot(housing.df) + geom_histogram(aes(x=MEDV), bins=9) + ylab("Count") +
  ggtitle("Distribution of Median Home Values (MEDV)")

# Simple heatmap of correlations
cor.mat <- round(cor(housing.df), 2)
melted.cor.mat <- melt(cor.mat)
g4 <- ggplot(melted.cor.mat, aes(x=Var1, y=Var2, fill=value)) + geom_tile() + xlab("") + ylab("") + 
  scale_fill_distiller(palette="RdBu", limits=c(-1, 1)) + ggtitle("Correlation of Features")

# Create grid of visualizations
grid.arrange(g1, g2, g3, g4)

# Feature scaling
housing.df$TAX <- scale(housing.df$TAX)
housing.df$AGE <- scale(housing.df$AGE)


print(housing.df)

# Check for missing values
print(sum(is.na(housing.df$LSTAT))) # 0
print(sum(is.na(housing.df$RM)))    # 0
print(sum(is.na(housing.df$MEDV)))  # 0

# Data is clean
set.seed(1) # Ensure reproducibility

# Split the data into training and test sets (60% training)
train.rows <- sample(rownames(housing.df), nrow(housing.df)*.6)
train.df <- housing.df[train.rows, ]
holdout.rows <- setdiff(rownames(housing.df), train.rows)
holdout.df <- housing.df[holdout.rows, ]

# Print counts of training and test data
print(nrow(holdout.df)) # 203
print(nrow(train.df))   # 303

# Create a linear model using all features to predict MEDV
house.lm <- lm(MEDV ~ ., data = train.df)

#####################################################################

# Backward stepwise regression (including all features)
house.lm.step <- step(house.lm, direction = "backward")

# Check the resulting model details
summary(house.lm.step)

# Predict on the holdout set
house.lm.step.pred <- predict(house.lm.step, newdata = holdout.df)

# calculate the RMSE for comparison
rmse_both <- rmse(house.lm.step.pred, holdout.df$MEDV)
print(paste("RMSE for Stepwise Regression (Backward Directions):", rmse_both))


#####################################################################

# forward stepwise regression (including all features)
house.lm.step <- step(house.lm, direction = "forward")

# Check the resulting model details
summary(house.lm.step)

# Predict on the holdout set
house.lm.step.pred <- predict(house.lm.step, newdata = holdout.df)

# calculate the RMSE for comparison
rmse_both <- rmse(house.lm.step.pred, holdout.df$MEDV)
print(paste("RMSE for Stepwise Regression (Forward):", rmse_both))


#####################################################################

# both stepwise regression (including all features)
house.lm.step <- step(house.lm, direction = "both")

# Check the resulting model details
summary(house.lm.step)

# Predict on the holdout set
house.lm.step.pred <- predict(house.lm.step, newdata = holdout.df)

# calculate the RMSE for comparison
rmse_both <- rmse(house.lm.step.pred, holdout.df$MEDV)
print(paste("RMSE for Stepwise Regression (Both Directions):", rmse_both))

#####################################################################
# changing the feature selection to only allow features with p-value 
# <= 0.05 in the model
#####################################################################

# Define a custom stepwise function that removes features with p-values >= 0.05
stepwise_with_pvalue <- function(model, data, direction) {
  model <- step(model, direction = direction, trace = 0)  # Perform stepwise regression
  p_values <- summary(model)$coefficients[, "Pr(>|t|)"]  # Get the p-values of predictors
  
  # Iteratively remove predictors with p-values >= 0.05
  while (any(p_values[-1] >= 0.05)) {  # Ignore the intercept
    # Find the predictor with the highest p-value
    max_p_value <- max(p_values[-1])
    if (max_p_value >= 0.05) {
      # Get the name of the predictor with the highest p-value
      drop_var <- names(p_values[p_values == max_p_value])
      # Update the formula by dropping the predictor
      formula_update <- as.formula(
        paste(". ~ . -", drop_var)
      )
      # Refit the model after dropping the predictor
      model <- update(model, formula_update)
      # Recompute p-values after removing the predictor
      p_values <- summary(model)$coefficients[, "Pr(>|t|)"]
    }
  }
  
  return(model)  # Return the final model
}

# Backward stepwise regression with p-value threshold
house.lm.step.backward <- stepwise_with_pvalue(house.lm, train.df, direction = "backward")

# Check the resulting model details
summary(house.lm.step.backward)

# Predict on the holdout set
house.lm.step.pred.backward <- predict(house.lm.step.backward, newdata = holdout.df)

# Calculate RMSE for backward stepwise regression with p-value filtering
rmse_backward <- rmse(house.lm.step.pred.backward, holdout.df$MEDV)
print(paste("RMSE for Backward Stepwise Regression (P-Value Based):", rmse_backward))


#####################################################################

# Forward stepwise regression with p-value threshold
house.lm.step.forward <- stepwise_with_pvalue(house.lm, train.df, direction = "forward")

# Check the resulting model details
summary(house.lm.step.forward)

# Predict on the holdout set
house.lm.step.pred.forward <- predict(house.lm.step.forward, newdata = holdout.df)

# Calculate RMSE for forward stepwise regression with p-value filtering
rmse_forward <- rmse(house.lm.step.pred.forward, holdout.df$MEDV)
print(paste("RMSE for Forward Stepwise Regression (P-Value Based):", rmse_forward))


#####################################################################

# Both stepwise regression with p-value threshold
house.lm.step.both <- stepwise_with_pvalue(house.lm, train.df, direction = "both")

# Check the resulting model details
summary(house.lm.step.both)

# Predict on the holdout set
house.lm.step.pred.both <- predict(house.lm.step.both, newdata = holdout.df)

# Calculate RMSE for both directions stepwise regression with p-value filtering
rmse_both <- rmse(house.lm.step.pred.both, holdout.df$MEDV)
print(paste("RMSE for Both Directions Stepwise Regression (P-Value Based):", rmse_both))


#####################################################################
# wanted to graph the linear relationships because I thought it was
# odd that age was not statistically relevant for some reason.
#####################################################################

# Load necessary libraries
library(ggplot2)
library(gridExtra)
library(scales)  # For scaling
library(reshape2)  # For reshaping data if needed

# Columns we want to scale and plot
column_sels <- colnames(housing.df)

# Subset the data for the selected columns
x <- housing.df[, column_sels]
y <- housing.df$MEDV

# Min-Max scaling for each selected column (range between 0 and 1)
x_scaled <- as.data.frame(lapply(x, function(col) {
  (col - min(col)) / (max(col) - min(col))
}))
names(x_scaled) <- column_sels

# Create regression plots for each feature against MEDV
plot_list <- list()

for (i in 1:length(column_sels)) {
  feature <- column_sels[i]
  
  # Create a scatter plot with regression line using ggplot2
  p <- ggplot(x_scaled, aes_string(x = feature, y = y)) +
    geom_point(alpha = 0.6) +
    geom_smooth(method = "lm", color = "blue") +
    ggtitle(paste("MEDV vs", feature)) +
    xlab(feature) +
    ylab("MEDV")
  
  plot_list[[i]] <- p  # Store the plot in a list
}

# Arrange all plots into a 2x4 grid
grid.arrange(grobs = plot_list, ncol = 4)


#####################################################################
# Just out of curiosity I went on kaggle and tried to find out
# the optimal feature combination so here it is:
# Interestingly Enough it performs worse
#####################################################################

# Define the features to include in the model
selected_features <- c("LSTAT", "INDUS", "NOX", "PTRATIO", "RM", "TAX", "DIS", "AGE")

# Create the formula for the linear model using only the selected features
formula <- as.formula(paste("MEDV ~", paste(selected_features, collapse = " + ")))

# Fit the linear model with only the selected features
house.lm <- lm(formula, data = train.df)

# Check the resulting model details
summary(house.lm)

# Predict on the holdout set
house.lm.pred <- predict(house.lm, newdata = holdout.df)

rmse_value <- rmse(house.lm.pred, holdout.df$MEDV)
print(paste("RMSE for Linear Model with Selected Features:", rmse_value))


#####################################################################
#####################################################################
#####################################################################
# Start HandsOn3
#####################################################################
#####################################################################
#####################################################################



#####################################################################
# 1 layer 5 neuron neural network
#####################################################################
# nn <- neuralnet(train.df$MEDV ~ ., data = train.df, hidden = c(5),
#                 linear.output = FALSE)  # Two hidden layers with 5 neurons
# 
# # Step 4: Plot the model
# plot(nn)
# 
# #predict test set
# pred <- compute(nn, holdout.df[, -which(names(holdout.df) == "MEDV")])
# rmse_value <- rmse(pred$net.result, holdout.df$MEDV)
# print(paste("RMSE for NN with 5 neurons:", rmse_value))
# 
# 
# pred <- unlist(pred)
# holdout.df$MEDV <- as.numeric(holdout.df$MEDV)
# 
# mse_value <- mean((pred - holdout.df$MEDV)^2)
# print(paste("MSE for NN with 5 neurons:", round(mse_value, 2)))


 
# #####################################################################
# # previous best features with BOTH stepwise function for 1 layer 5 neuron neural network
# #####################################################################
# nn <- neuralnet(MEDV ~ CRIM + INDUS + CHAS + NOX + DIS + RAD + TAX + PTRATIO + LSTAT + CAT.MEDV,
#                 data = train.df, hidden = c(5), linear.output = TRUE, stepmax = 1e6)  # Use TRUE for regression
# 
# # Plot the model
# plot(nn)
# 
# # Predict on test set
# pred <- compute(nn, holdout.df[, -which(names(holdout.df) == "MEDV")])$net.result
# 
# # Calculate RMSE manually
# rmse_value <- sqrt(mean((pred - holdout.df$MEDV)^2))
# print(paste("RMSE for NN with 5 neurons:", round(rmse_value, 2)))
# 
# pred <- unlist(pred)
# holdout.df$MEDV <- as.numeric(holdout.df$MEDV)
# 
# mse_value <- mean((pred - holdout.df$MEDV)^2)
# print(paste("MSE for NN with 5 neurons:", round(mse_value, 2)))


#####################################################################
# previous best features with Forward stepwise function for 1 layer 5 neuron neural network
#####################################################################
# nn <- neuralnet(MEDV ~ CRIM + INDUS + CHAS + NOX + DIS + RAD + TAX +
#                   PTRATIO + LSTAT + CAT.MEDV,
#                 data = train.df, hidden = c(5), linear.output = TRUE, stepmax = 1e6)  # Use TRUE for regression
# 
# # Plot the model
# plot(nn)
# 
# # Predict on test set
# pred <- compute(nn, holdout.df[, -which(names(holdout.df) == "MEDV")])$net.result
# 
# # Calculate RMSE manually
# rmse_value <- sqrt(mean((pred - holdout.df$MEDV)^2))
# print(paste("RMSE for NN with 5 neurons:", round(rmse_value, 2)))
# 
# pred <- unlist(pred)
# holdout.df$MEDV <- as.numeric(holdout.df$MEDV)
# 
# mse_value <- mean((pred - holdout.df$MEDV)^2)
# print(paste("MSE for NN with 5 neurons:", round(mse_value, 2)))


#####################################################################
# previous best features with Forward stepwise function for 1 layer 8 neuron neural network
#####################################################################
# nn <- neuralnet(MEDV ~ CRIM + INDUS + CHAS + NOX + DIS + RAD + TAX +
#                   PTRATIO + LSTAT + CAT.MEDV,
#                 data = train.df, hidden = c(8), linear.output = TRUE, stepmax = 1e6)  # Use TRUE for regression
# 
# # Plot the model
# plot(nn)
# 
# # Predict on test set
# pred <- compute(nn, holdout.df[, -which(names(holdout.df) == "MEDV")])$net.result
# 
# # Calculate RMSE manually
# rmse_value <- sqrt(mean((pred - holdout.df$MEDV)^2))
# print(paste("RMSE for NN with 8 neurons:", round(rmse_value, 2)))
# 
# pred <- unlist(pred)
# holdout.df$MEDV <- as.numeric(holdout.df$MEDV)
# 
# mse_value <- mean((pred - holdout.df$MEDV)^2)
# print(paste("MSE for NN with 8 neurons:", round(mse_value, 2)))


#####################################################################
# Tree Model 1: Basic Regression Tree
#####################################################################
tree_model1 <- rpart(MEDV ~ CRIM + INDUS + CHAS + NOX + DIS + RAD + TAX + PTRATIO + LSTAT + CAT.MEDV,
                     data = train.df, method = "anova")
print(tree_model1)  # Summary of the model

# Plot the tree
plot(tree_model1, uniform = TRUE, main = "Basic Tree Model 1")
text(tree_model1, use.n = TRUE, cex = 0.8)

# Predict on test set
pred1 <- predict(tree_model1, holdout.df)
rmse_value1 <- rmse(pred1, holdout.df$MEDV)
print(paste("RMSE for Basic Tree Model 1:", round(rmse_value1, 2)))


#####################################################################
# Tree Model 2: Deeper Tree with Lower cp
#####################################################################
tree_model2 <- rpart(MEDV ~ CRIM + INDUS + CHAS + NOX + DIS + RAD + TAX + PTRATIO + LSTAT + CAT.MEDV,
                     data = train.df, method = "anova", control = rpart.control(cp = 0.01, minsplit = 10))
print(tree_model2)  # Summary of the model

# Plot the tree
plot(tree_model2, uniform = TRUE, main = "Deeper Tree Model 2")
text(tree_model2, use.n = TRUE, cex = 0.8)

# Predict on test set
pred2 <- predict(tree_model2, holdout.df)
rmse_value2 <- rmse(pred2, holdout.df$MEDV)
print(paste("RMSE for Deeper Tree Model 2:", round(rmse_value2, 2)))


#####################################################################
# Tree Model 3: Shallow Tree with Limited Depth
#####################################################################
tree_model3 <- rpart(MEDV ~ CRIM + INDUS + CHAS + NOX + DIS + RAD + TAX + PTRATIO + LSTAT + CAT.MEDV,
                     data = train.df, method = "anova", control = rpart.control(maxdepth = 3, cp = 0.05))
print(tree_model3)  # Summary of the model

# Plot the tree
plot(tree_model3, uniform = TRUE, main = "Shallow Tree Model 3")
text(tree_model3, use.n = TRUE, cex = 0.8)

# Predict on test set
pred3 <- predict(tree_model3, holdout.df)
rmse_value3 <- rmse(pred3, holdout.df$MEDV)
print(paste("RMSE for Shallow Tree Model 3:", round(rmse_value3, 2)))



#####################################################################
### START HANDS ON 4 ###
#####################################################################

# Train the KNN model with caret
knn_model <- train(MEDV ~ ., data = train.df, method = "knn", tuneLength = 10)

# Print the model details
print(knn_model)

# Make predictions on the test set
knn_predictions <- predict(knn_model, newdata = holdout.df)

# Calculate residuals
rmse_value3 <- rmse(knn_predictions, holdout.df$MEDV)
print(paste("RMSE for KNN:", round(rmse_value3, 2)))

