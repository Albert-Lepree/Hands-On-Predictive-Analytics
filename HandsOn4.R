# install.packages(c("caret", "ipred", "MASS", "ggplot2"))

# install.packages("caret")
# install.packages("recipes")



library(caret)
library(ipred)
library(MASS)  
library(ggplot2)


# Load the required library
library(caret)
data("Boston")

# Set seed for reproducibility
set.seed(42)

# Create training and testing indices
trainIndex <- createDataPartition(Boston$medv, p = 0.8, list = FALSE)
trainData <- Boston[trainIndex, ]
testData <- Boston[-trainIndex, ]

# Separate predictors and target variable for training and testing sets
X_train <- trainData[, -14]  # Exclude the 14th column ('medv')
y_train <- trainData$medv
X_test <- testData[, -14]
y_test <- testData$medv



preProcValues <- preProcess(X_train, method = c("center", "scale"))
X_train_scaled <- predict(preProcValues, X_train)
X_test_scaled <- predict(preProcValues, X_test)


set.seed(42)
bagging_model <- bagging(
  medv ~ ., 
  data = trainData, 
  nbagg = 50,
  coob = TRUE
)




summary(bagging_model)


bagging_pred <- predict(bagging_model, newdata = testData)


mse_bagging <- mean((bagging_pred - y_test)^2)
r2_bagging <- 1 - sum((bagging_pred - y_test)^2) / sum((y_test - mean(y_test))^2)


cat("Bagging Model MSE:", mse_bagging, "\n")
cat("Bagging Model R²:", r2_bagging, "\n")




set.seed(42)
bagging_caret <- train(
  medv ~ ., 
  data = trainData, 
  method = "treebag",
  trControl = trainControl(method = "cv", number = 5)
)




print(bagging_caret)


bagging_caret_pred <- predict(bagging_caret, newdata = testData)


mse_bagging_caret <- mean((bagging_caret_pred - y_test)^2)
r2_bagging_caret <- 1 - sum((bagging_caret_pred - y_test)^2) / sum((y_test - mean(y_test))^2)


cat("Bagging (caret) Model MSE:", mse_bagging_caret, "\n")
cat("Bagging (caret) Model R²:", r2_bagging_caret, "\n")


varImp(bagging_caret)

# Train a gradient boosting model
boost_model_gbm <- train(medv ~ ., data = trainData, method = "gbm", 
                         trControl = trainControl(method = "cv", number = 10), 
                         verbose = FALSE)

# Print the model results
print(boost_model_gbm)

# Make predictions on the test data
boost_predictions_gbm <- predict(boost_model_gbm, testData)

# Evaluate performance
postResample(boost_predictions_gbm, testData$medv)



ggplot(data = data.frame(Actual = y_test, Predicted = bagging_caret_pred), aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Predicted vs Actual: Bagging Model", x = "Actual Values", y = "Predicted Values") +
  theme_minimal()

# install.packages("ggrepel")
library(ggrepel)

print(trainData)

ggplot(mapping = aes(x = medv, y = lstat, color = age)) +
  geom_point(data = trainData) +
  geom_text_repel(aes(label = rownames(trainData)), data = trainData, show.legend = FALSE) +
  scale_color_gradient(low = "blue", high = "red")  # Customize colors as desired

# Train a KNN model
knn_model <- train(medv ~ ., data = trainData, method = "knn",
                   tuneLength = 10,  # Number of K values to test
                   trControl = trainControl(method = "cv", number = 10))  # 10-fold cross-validation

# Print the model summary
print(knn_model)

# Plot the results (K vs RMSE)
plot(knn_model)

# Predict on the test data
knn_predictions <- predict(knn_model, testData)

# Evaluate performance
postResample(knn_predictions, testData$medv)



# Load required libraries
library(caret)
library(caretEnsemble)

# Train the bagging model with savePredictions
bagging_caret <- train(
  medv ~ ., 
  data = trainData, 
  method = "treebag",
  trControl = trainControl(method = "cv", number = 5, savePredictions = "all")
)

# Train the gradient boosting model with savePredictions
boost_model_gbm <- train(
  medv ~ ., 
  data = trainData, 
  method = "gbm", 
  trControl = trainControl(method = "cv", number = 10, savePredictions = "all"),
  verbose = FALSE
)

# Combine the models using caretEnsemble
models <- list(bagging = bagging_caret, boosting = boost_model_gbm)

# Stack the models
ensemble_model <- caretEnsemble(models)

# Summarize the ensemble model
summary(ensemble_model)

# Make predictions with the stacked ensemble model
ensemble_predictions <- predict(ensemble_model, testData)

# Evaluate the performance of the ensemble model
postResample(ensemble_predictions, testData$medv)

