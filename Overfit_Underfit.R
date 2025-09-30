# Load required libraries
library(tidyverse)       # For data wrangling
library(caret)           # For ML models
library(rpart)           # For Decision Trees
library(adabag)          # For AdaBoost
library(randomForest)    # For Random Forest
library(xgboost)         # For XGBoost
library(e1071)           # For confusion matrix

# Set working directory (Change this path to where your file is saved)
setwd("C:/Users/Prajwal/Desktop")

# Load CSV file
water_data <- read.csv("water_potability.csv")

# Check first few rows
head(water_data)
dim(water_data)

# Remove rows with missing values
water_data <- na.omit(water_data)

# Verify if NA values are gone
sum(is.na(water_data))

# Convert Potability column to factor (for classification)
water_data$Potability <- as.factor(water_data$Potability)
str(water_data)

set.seed(123)  # For reproducibility

# Create 80-20 train-test split
trainIndex <- createDataPartition(water_data$Potability, p = 0.8, list = FALSE)
train_data <- water_data[trainIndex, ]
test_data  <- water_data[-trainIndex, ]

log_model <- glm(Potability ~ ., data = train_data, family = binomial)
summary(log_model)

# Predict
log_pred <- predict(log_model, test_data, type = "response")
log_pred_class <- ifelse(log_pred > 0.5, 1, 0)

# Confusion Matrix
confusionMatrix(as.factor(log_pred_class), test_data$Potability)

dt_model <- rpart(Potability ~ ., data = train_data, method = "class")

# Predict
dt_pred <- predict(dt_model, test_data, type = "class")

# Confusion Matrix
confusionMatrix(dt_pred, test_data$Potability)

rf_model <- randomForest(Potability ~ ., data = train_data, ntree = 100)

# Predict
rf_pred <- predict(rf_model, test_data)

# Confusion Matrix
confusionMatrix(rf_pred, test_data$Potability)

ada_model <- boosting(Potability ~ ., data = train_data, boos = TRUE, mfinal = 50)

# Predict
ada_pred <- predict.boosting(ada_model, newdata = test_data)

# Confusion Matrix
confusionMatrix(as.factor(ada_pred$class), test_data$Potability)

# Convert data to matrix form for XGBoost
train_matrix <- xgb.DMatrix(data = as.matrix(train_data[,-9]), label = as.numeric(train_data$Potability)-1)
test_matrix  <- xgb.DMatrix(data = as.matrix(test_data[,-9]),  label = as.numeric(test_data$Potability)-1)

# Train XGBoost model
xgb_model <- xgboost(data = train_matrix, max.depth = 3, eta = 0.1, nrounds = 50, objective = "binary:logistic", verbose = 0)

# Predict
xgb_pred <- predict(xgb_model, test_matrix)
xgb_pred_class <- ifelse(xgb_pred > 0.5, 1, 0)

# Confusion Matrix
confusionMatrix(as.factor(xgb_pred_class), test_data$Potability)

# Function to calculate accuracy
get_accuracy <- function(model, data, type = "class") {
  pred <- predict(model, data, type = type)
  return(mean(pred == data$Potability))
}

# Decision Tree
train_acc_dt <- get_accuracy(dt_model, train_data)
test_acc_dt  <- get_accuracy(dt_model, test_data)

cat("Decision Tree Train Accuracy:", train_acc_dt, "\n")
cat("Decision Tree Test Accuracy:", test_acc_dt, "\n")

if(train_acc_dt - test_acc_dt > 0.15) {
  cat("Model is Overfitting\n")
} else if(abs(train_acc_dt - test_acc_dt) < 0.05) {
  cat("Model is Balanced\n")
} else {
  cat("Model might be Underfitting\n")
}

# Example accuracy values (replace with actual results)
results <- data.frame(
  Model = c("Logistic Regression", "Decision Tree", "Random Forest", "AdaBoost", "XGBoost"),
  Accuracy = c(0.78, 0.82, 0.90, 0.85, 0.88)
)

print(results)

# Bar plot for comparison
ggplot(results, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Model Accuracy Comparison", x = "Model", y = "Accuracy") +
  theme(axis.text.x = element_text(angle = 15, hjust = 1))


