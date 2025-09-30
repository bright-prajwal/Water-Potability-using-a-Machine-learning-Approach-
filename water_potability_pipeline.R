# 1. Load libraries
library(tidyverse)
library(caret)
library(randomForest)
library(rpart)
library(class)
library(adabag)
library(xgboost)
library(e1071)
library(pROC)

# 2. Load dataset
# Change working directory or provide full path to CSV
#setwd("C:/Users/Prajwal/Desktop/Water_Portability/")
setwd("C:/Users/Prajwal/Desktop/Water Portability/")
data_path <- "water_potability.csv"  # <-- change path here if needed

if (!file.exists(data_path)) {
  stop(paste("CSV file not found at:", data_path, "\nPlease put the file in working directory or update data_path."))
}

df <- read.csv(data_path, stringsAsFactors = FALSE)
cat("Loaded dataset with", nrow(df), "rows and", ncol(df), "columns\n")

# 3. Initial inspection
print(names(df))
print(summary(df))
print(sapply(df, function(x) sum(is.na(x))))

# 4. Handle missing values (median imputation for numeric columns used in original notebook)
num_cols <- names(df)[sapply(df, is.numeric)]
for (col in num_cols) {
  miss_pct <- sum(is.na(df[[col]])) / nrow(df) * 100
  if (miss_pct > 30) {
    message("Column ", col, " has ", round(miss_pct,2), "% missing. Dropping column.")
    df[[col]] <- NULL
  } else if (miss_pct > 0) {
    med <- median(df[[col]], na.rm = TRUE)
    df[[col]][is.na(df[[col]])] <- med
    message("Filled missing in ", col, " with median = ", med)
  }
}

# 5. Drop original Potability column (user requested to delete it and compute new labels)
if ("Potability" %in% names(df)) {
  df$Potability_original <- df$Potability   # optional backup
  df$Potability <- NULL
  message("Original Potability column moved to Potability_original and removed from active features.")
}

# 6. Create a new potability label using thresholds (user-provided safe ranges)

# Normalize column names (replace spaces, capitals to match common names)
names(df) <- make.names(names(df))

# Helper to check presence
has <- function(x) x %in% names(df)

potability_score <- function(row) {
  score <- 0
  # pH: ideal 6.5-8.5
  if (has("ph")) {
    ph <- as.numeric(row["ph"])
    if (!is.na(ph) && ph >= 6.5 && ph <= 8.5) score <- score + 2
  }
  # Hardness: ideal <=200, permissible <=600
  if (has("Hardness")) {
    h <- as.numeric(row["Hardness"])
    if (!is.na(h) && h <= 200) score <- score + 2
    else if (!is.na(h) && h <= 600) score <- score + 1
  }
  # Solids (TDS): ideal <=500, permissible <=2000
  if (has("Solids")) {
    s <- as.numeric(row["Solids"])
    if (!is.na(s) && s <= 500) score <- score + 2
    else if (!is.na(s) && s <= 2000) score <- score + 1
  }
  # Chloramines: ideal <=4
  if (has("Chloramines")) {
    c <- as.numeric(row["Chloramines"])
    if (!is.na(c) && c <= 4) score <- score + 2
  }
  # Sulfate: ideal <=200, permissible <=400
  if (has("Sulfate")) {
    su <- as.numeric(row["Sulfate"])
    if (!is.na(su) && su <= 200) score <- score + 2
    else if (!is.na(su) && su <= 400) score <- score + 1
  }
  # Conductivity: ideal <=600 (using user's permissible 400-600), permissable up to 800
  if (has("Conductivity")) {
    cond <- as.numeric(row["Conductivity"])
    if (!is.na(cond) && cond <= 600) score <- score + 2
    else if (!is.na(cond) && cond <= 800) score <- score + 1
  }
  # Organic_carbon: ideal <=5, permissible <=8
  # Column may be named Organic_carbon or OrganicCarbon depending on dataset
  if (has("Organic_carbon")) {
    oc <- as.numeric(row["Organic_carbon"])
    if (!is.na(oc) && oc <= 5) score <- score + 2
    else if (!is.na(oc) && oc <= 8) score <- score + 1
  } else if (has("OrganicCarbon")) {
    oc <- as.numeric(row["OrganicCarbon"])
    if (!is.na(oc) && oc <= 5) score <- score + 2
    else if (!is.na(oc) && oc <= 8) score <- score + 1
  }
  # Trihalomethanes: ideal <=100
  if (has("Trihalomethanes")) {
    th <- as.numeric(row["Trihalomethanes"])
    if (!is.na(th) && th <= 100) score <- score + 2
  }
  # Turbidity: ideal <=1, permissible <=5
  if (has("Turbidity")) {
    tu <- as.numeric(row["Turbidity"])
    if (!is.na(tu) && tu <= 1) score <- score + 2
    else if (!is.na(tu) && tu <= 5) score <- score + 1
  }
  return(score)
}

# Apply scoring
scores <- apply(df, 1, potability_score)
df$Potability_New_Score <- scores
# Decide threshold: use >=10 as potable (adjustable)
df$Potability_New <- ifelse(df$Potability_New_Score >= 10, 1, 0)
df$Potability_New <- factor(df$Potability_New, levels = c(0,1), labels = c("Not Potable","Potable"))

cat("New potability labels created. Distribution:\n")
print(table(df$Potability_New))

# 7. Save a cleaned CSV (features + new labels) for Power BI or reuse
clean_path <- "clean_water_potability.csv"
write.csv(df, clean_path, row.names = FALSE)
cat("Saved cleaned dataset to:", clean_path, "\n")

# 8. Prepare data for modeling: remove any non-feature columns
# Keep only numeric features and the new label
feature_cols <- names(df)[sapply(df, is.numeric)]
# Remove the score column from features if present
feature_cols <- setdiff(feature_cols, "Potability_New_Score")
# If you want to use all numeric features, proceed; else specify list manually
model_df <- df[, c(feature_cols, "Potability_New")]

# Convert label to factor with levels 0,1
model_df$Potability_New <- factor(model_df$Potability_New, levels = c("Not Potable","Potable"))

# 9. Train/Test split (80/20)
set.seed(42)
train_index <- createDataPartition(model_df$Potability_New, p = 0.8, list = FALSE)
train_df <- model_df[train_index, ]
test_df  <- model_df[-train_index, ]

cat("Train rows:", nrow(train_df), "Test rows:", nrow(test_df), "\n")

# 10. Train models
# We'll train: Logistic Regression, Decision Tree, Random Forest, KNN, AdaBoost, XGBoost

# Helper to compute metrics
compute_metrics <- function(true, pred, positive = "Potable") {
  cm <- confusionMatrix(pred, true, positive = positive)
  acc <- cm$overall["Accuracy"]
  prec <- cm$byClass["Precision"]
  rec <- cm$byClass["Recall"]
  f1 <- cm$byClass["F1"]
  return(list(confusion = cm$table, Accuracy = as.numeric(acc), Precision = as.numeric(prec),
              Recall = as.numeric(rec), F1 = as.numeric(f1)))
}

results_list <- list()

# 10.1 Logistic Regression (glm)
cat("Training Logistic Regression...\n")
glm_model <- glm(Potability_New ~ ., data = train_df, family = binomial)
glm_probs <- predict(glm_model, test_df, type = "response")
glm_pred <- factor(ifelse(glm_probs > 0.5, "Potable", "Not Potable"), levels = c("Not Potable","Potable"))
res_glm <- compute_metrics(test_df$Potability_New, glm_pred)
results_list$Logistic_Regression <- res_glm

# 10.2 Decision Tree (rpart)
cat("Training Decision Tree...\n")
dt_model <- rpart(Potability_New ~ ., data = train_df, method = "class")
dt_pred <- predict(dt_model, test_df, type = "class")
res_dt <- compute_metrics(test_df$Potability_New, dt_pred)
results_list$Decision_Tree <- res_dt

# 10.3 Random Forest
cat("Training Random Forest...\n")
rf_model <- randomForest(Potability_New ~ ., data = train_df, ntree = 200)
rf_pred <- predict(rf_model, test_df)
res_rf <- compute_metrics(test_df$Potability_New, rf_pred)
results_list$Random_Forest <- res_rf

# 10.4 K-Nearest Neighbors (class::knn) - requires scaled features
cat("Training KNN...\n")
# Scale numeric features using caret preProcess
preproc <- preProcess(train_df[, setdiff(names(train_df), "Potability_New")], method = c("center", "scale"))
train_scaled <- predict(preproc, train_df[, setdiff(names(train_df), "Potability_New")])
test_scaled  <- predict(preproc, test_df[, setdiff(names(test_df), "Potability_New")])
k <- 5
knn_pred <- knn(train = train_scaled, test = test_scaled, cl = train_df$Potability_New, k = k)
res_knn <- compute_metrics(test_df$Potability_New, knn_pred)
results_list$KNN <- res_knn

# 10.5 AdaBoost (adabag::boosting)
cat("Training AdaBoost...\n")
# adabag expects the label as a factor in the same data frame
ada_model <- boosting(Potability_New ~ ., data = train_df, boos = TRUE, mfinal = 50)
ada_pred_raw <- predict.boosting(ada_model, newdata = test_df)
ada_pred <- factor(ifelse(ada_pred_raw$class == 1 | ada_pred_raw$class == "Potable", "Potable", "Not Potable"),
                   levels = c("Not Potable","Potable"))
# Note: predict.boosting$class may be 1/0; ensure mapping
# Try mapping intelligently
if (is.numeric(ada_pred_raw$class)) {
  ada_pred <- factor(ifelse(ada_pred_raw$class == 1, "Potable", "Not Potable"), levels = c("Not Potable","Potable"))
} else {
  ada_pred <- factor(ada_pred_raw$class, levels = c("Not Potable","Potable"))
}
res_ada <- compute_metrics(test_df$Potability_New, ada_pred)
results_list$AdaBoost <- res_ada

# 10.6 XGBoost
cat("Training XGBoost...\n")
# Convert factors to 0/1
label_train <- ifelse(train_df$Potability_New == "Potable", 1, 0)
label_test  <- ifelse(test_df$Potability_New == "Potable", 1, 0)
dtrain <- xgb.DMatrix(data = as.matrix(train_df[, setdiff(names(train_df), "Potability_New")]), label = label_train)
dtest  <- xgb.DMatrix(data = as.matrix(test_df[, setdiff(names(test_df), "Potability_New")]), label = label_test)
params <- list(objective = "binary:logistic", eval_metric = "error")
xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 100, verbose = 0)
xgb_pred_prob <- predict(xgb_model, dtest)
xgb_pred <- factor(ifelse(xgb_pred_prob > 0.5, "Potable", "Not Potable"), levels = c("Not Potable","Potable"))
res_xgb <- compute_metrics(test_df$Potability_New, xgb_pred)
results_list$XGBoost <- res_xgb

# 11. Aggregate results and save CSV with metrics
models <- names(results_list)
metrics_df <- data.frame(
  Model = models,
  Accuracy = sapply(results_list, function(x) x$Accuracy),
  Precision = sapply(results_list, function(x) x$Precision),
  Recall = sapply(results_list, function(x) x$Recall),
  F1 = sapply(results_list, function(x) x$F1)
)
# Save CSV
metrics_path <- "water_models_metrics.csv"
write.csv(metrics_df, metrics_path, row.names = FALSE)
cat("Saved model metrics to:", metrics_path, "\n")
print(metrics_df)

# 12. Plot comparisons (Accuracy, Precision, Recall, F1)
# Save plots to PNG files for Power BI or reports

library(ggplot2)
metrics_long <- pivot_longer(metrics_df, cols = -Model, names_to = "Metric", values_to = "Value")

p_acc <- ggplot(metrics_df, aes(x = reorder(Model, -Accuracy), y = Accuracy, fill = Model)) +
  geom_col() +
  geom_text(aes(label = sprintf("%.2f%%", Accuracy*100)), vjust = -0.5) +
  ylim(0,1) +
  labs(title = "Model Accuracy Comparison", y = "Accuracy", x = "") +
  theme_minimal() +
  theme(legend.position = "none")

ggsave("model_accuracy_comparison.png", p_acc, width = 10, height = 6, dpi = 150)
cat("Saved plot: model_accuracy_comparison.png\n")

p_all <- ggplot(metrics_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_col(position = position_dodge()) +
  geom_text(aes(label = sprintf("%.2f", Value)), position = position_dodge(width = 0.9), vjust = -0.5, size = 3) +
  labs(title = "Model Metrics Comparison", y = "Value", x = "") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))

ggsave("model_metrics_comparison.png", p_all, width = 12, height = 6, dpi = 150)
cat("Saved plot: model_metrics_comparison.png\n")

# 13. Save cleaned data + new label for Power BI
write.csv(model_df, "model_ready_water_potability.csv", row.names = FALSE)
cat("Saved model-ready data to: model_ready_water_potability.csv\n")

# 14. End message
cat("Pipeline finished. Files created:\n -", clean_path, "\n -", metrics_path, "\n - model_accuracy_comparison.png\n - model_metrics_comparison.png\n - model_ready_water_potability.csv\n")




# 
# 15. Compute train accuracy for overfit/underfit visualization

train_accuracies <- c(
  Logistic_Regression = mean(ifelse(predict(glm_model, train_df, type="response") > 0.5, "Potable", "Not Potable") == train_df$Potability_New),
  Decision_Tree = mean(predict(dt_model, train_df, type="class") == train_df$Potability_New),
  Random_Forest = mean(predict(rf_model, train_df) == train_df$Potability_New),
  KNN = mean(knn(train_scaled, train_scaled, cl = train_df$Potability_New, k = k) == train_df$Potability_New),
  AdaBoost = {
    ada_pred_train_raw <- predict.boosting(ada_model, newdata = train_df)
    if (is.numeric(ada_pred_train_raw$class)) {
      factor(ifelse(ada_pred_train_raw$class == 1, "Potable", "Not Potable"), levels=c("Not Potable","Potable"))
    } else {
      factor(ada_pred_train_raw$class, levels=c("Not Potable","Potable"))
    } -> pred
    mean(pred == train_df$Potability_New)
  },
  XGBoost = {
    pred_train_prob <- predict(xgb_model, dtrain)
    pred_train <- factor(ifelse(pred_train_prob > 0.5, "Potable", "Not Potable"), levels=c("Not Potable","Potable"))
    mean(pred_train == train_df$Potability_New)
  }
)

# Combine train/test accuracy
overfit_df <- data.frame(
  Model = names(results_list),
  Train_Accuracy = as.numeric(train_accuracies),
  Test_Accuracy = metrics_df$Accuracy
)

# 16. Plot overfit/underfit comparison
library(reshape2)
overfit_long <- melt(overfit_df, id.vars="Model", variable.name="Dataset", value.name="Accuracy")

p_overfit <- ggplot(overfit_long, aes(x=Model, y=Accuracy, fill=Dataset)) +
  geom_col(position = position_dodge()) +
  geom_text(aes(label = sprintf("%.2f%%", Accuracy*100)), position = position_dodge(width=0.9), vjust=-0.5, size=3) +
  labs(title="Train vs Test Accuracy (Overfit / Underfit Check)", y="Accuracy", x="Model") +
  ylim(0,1) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle=20, hjust=1))

ggsave("model_overfit_underfit.png", p_overfit, width=12, height=6, dpi=150)
cat("Saved overfit/underfit plot: model_overfit_underfit.png\n")


# Load libraries
library(ggplot2)
library(caret)

# Generate synthetic data
set.seed(42)
x <- seq(-3, 3, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)
data <- data.frame(x = x, y = y)

# Split data
trainIndex <- createDataPartition(data$y, p = 0.8, list = FALSE)
train <- data[trainIndex, ]
test <- data[-trainIndex, ]

# Linear model (underfitting)
lm_model <- lm(y ~ x, data = train)
lm_pred <- predict(lm_model, test)

# MSE
lm_mse <- mean((test$y - lm_pred)^2)
cat("Linear Model MSE (Underfitting):", lm_mse, "\n")

# Plot
p_under <- ggplot(data, aes(x, y)) +
  geom_point(color = "gray") +
  geom_line(aes(y = predict(lm_model, data)), color = "blue", size = 1.2) +
  labs(title = "Underfitting Example (Linear Model)", y = "Predicted y") +
  theme_minimal()

print(p_under)

# Save plot
ggsave("underfitting_plot.png", plot = p_under, width = 8, height = 5, dpi = 150)
cat("Underfitting plot saved as underfitting_plot.png\n")

# Overfitting Example - Decision Tree

# Load libraries
library(ggplot2)
library(rpart)
library(caret)

# Generate synthetic data
set.seed(42)
x <- seq(-3, 3, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)
data <- data.frame(x = x, y = y)

# Split data
trainIndex <- createDataPartition(data$y, p = 0.8, list = FALSE)
train <- data[trainIndex, ]
test <- data[-trainIndex, ]

# Decision tree (overfitting)
tree_model <- rpart(y ~ x, data = train, control = rpart.control(cp = 0.0001))
tree_pred <- predict(tree_model, test)

# MSE
tree_mse <- mean((test$y - tree_pred)^2)
cat("Decision Tree MSE (Overfitting):", tree_mse, "\n")

# Plot
p_over <- ggplot(data, aes(x, y)) +
  geom_point(color = "gray") +
  geom_line(aes(y = predict(tree_model, data)), color = "red", size = 1.2) +
  labs(title = "Overfitting Example (Decision Tree)", y = "Predicted y") +
  theme_minimal()

print(p_over)

# Save plot
ggsave("overfitting_plot.png", plot = p_over, width = 8, height = 5, dpi = 150)
cat("Overfitting plot saved as overfitting_plot.png\n")

# Underfitting Example - Linear Model

# Load libraries
library(ggplot2)
library(caret)

# Generate synthetic data
set.seed(42)
x <- seq(-3, 3, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)
data <- data.frame(x = x, y = y)

# Split data
trainIndex <- createDataPartition(data$y, p = 0.8, list = FALSE)
train <- data[trainIndex, ]
test <- data[-trainIndex, ]

# Linear model (underfitting)
lm_model <- lm(y ~ x, data = train)
lm_pred <- predict(lm_model, test)

# MSE
lm_mse <- mean((test$y - lm_pred)^2)
cat("Linear Model MSE (Underfitting):", lm_mse, "\n")

# Plot
p_under <- ggplot(data, aes(x, y)) +
  geom_point(color = "gray") +
  geom_line(aes(y = predict(lm_model, data)), color = "blue", size = 1.2) +
  labs(title = "Underfitting Example (Linear Model)", y = "Predicted y") +
  theme_minimal()

print(p_under)

# Save plot
ggsave("RF_underfitting_plot.png", plot = p_under, width = 8, height = 5, dpi = 150)
cat("Underfitting plot saved as underfitting_plot.png\n")

# Overfitting Example - Decision Tree

# Load libraries
library(ggplot2)
library(rpart)
library(caret)

# Generate synthetic data
set.seed(42)
x <- seq(-3, 3, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)
data <- data.frame(x = x, y = y)

# Split data
trainIndex <- createDataPartition(data$y, p = 0.8, list = FALSE)
train <- data[trainIndex, ]
test <- data[-trainIndex, ]

# Decision tree (overfitting)
tree_model <- rpart(y ~ x, data = train, control = rpart.control(cp = 0.0001))
tree_pred <- predict(tree_model, test)

# MSE
tree_mse <- mean((test$y - tree_pred)^2)
cat("Decision Tree MSE (Overfitting):", tree_mse, "\n")

# Plot
p_over <- ggplot(data, aes(x, y)) +
  geom_point(color = "gray") +
  geom_line(aes(y = predict(tree_model, data)), color = "red", size = 1.2) +
  labs(title = "Overfitting Example (Decision Tree)", y = "Predicted y") +
  theme_minimal()

print(p_over)

# Save plot
ggsave("RF_overfitting_plot.png", plot = p_over, width = 8, height = 5, dpi = 150)
cat("Overfitting plot saved as overfitting_plot.png\n")

# Random Forest Example

# Load libraries
library(ggplot2)
library(randomForest)
library(caret)

# Generate synthetic data
set.seed(42)
x <- seq(-3, 3, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)
data <- data.frame(x = x, y = y)

# Split data
trainIndex <- createDataPartition(data$y, p = 0.8, list = FALSE)
train <- data[trainIndex, ]
test <- data[-trainIndex, ]

# Train Random Forest
set.seed(42)
rf_model <- randomForest(y ~ x, data = train, ntree = 200)
rf_pred <- predict(rf_model, test)

# MSE
rf_mse <- mean((test$y - rf_pred)^2)
cat("Random Forest MSE:", rf_mse, "\n")

# Plot
p_rf <- ggplot(data, aes(x, y)) +
  geom_point(color = "gray") +
  geom_line(aes(y = predict(rf_model, data)), color = "green", size = 1.2) +
  labs(title = "Random Forest Fit", y = "Predicted y") +
  theme_minimal()

print(p_rf)

# Save plot
ggsave("random_forest_plot.png", plot = p_rf, width = 8, height = 5, dpi = 150)
cat("Random Forest plot saved as random_forest_plot.png\n")

# Random Forest Example with Name and Style

# Load libraries
library(ggplot2)
library(randomForest)
library(caret)
library(extrafont) # Optional: for custom fonts

# Generate synthetic data
set.seed(42)
x <- seq(-3, 3, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)
data <- data.frame(x = x, y = y)

# Split data
trainIndex <- createDataPartition(data$y, p = 0.8, list = FALSE)
train <- data[trainIndex, ]
test <- data[-trainIndex, ]

# Train Random Forest
set.seed(42)
rf_model <- randomForest(y ~ x, data = train, ntree = 200)
rf_pred <- predict(rf_model, test)

# MSE
rf_mse <- mean((test$y - rf_pred)^2)
cat("Random Forest MSE:", rf_mse, "\n")

# Plot with custom style
p_rf <- ggplot(data, aes(x, y)) +
  geom_point(color = "darkgray", size = 2, alpha = 0.6) +
  geom_line(aes(y = predict(rf_model, data)), color = "forestgreen", size = 1.5) +
  labs(
    title = "Random Forest Fit on Synthetic Data",
    subtitle = "Model Prediction Example | Your Name Here",
    x = "X Values",
    y = "Predicted Y"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", color = "darkblue", size = 16, hjust = 0.5),
    plot.subtitle = element_text(face = "italic", color = "darkred", size = 12, hjust = 0.5),
    axis.title = element_text(face = "bold", color = "black"),
    axis.text = element_text(color = "black")
  )

print(p_rf)

# Save plot
ggsave("New_random_2_forest_plot_styled.png", plot = p_rf, width = 8, height = 5, dpi = 150)
cat("Random Forest plot saved as random_forest_plot_styled.png\n")

# Load libraries
library(ggplot2)
library(rpart)
library(randomForest)
library(caret)

# Generate synthetic data
set.seed(42)
x <- seq(-3, 3, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)
data <- data.frame(x = x, y = y)

# Split data
trainIndex <- createDataPartition(data$y, p = 0.8, list = FALSE)
train <- data[trainIndex, ]
test <- data[-trainIndex, ]

# Models
lm_model <- lm(y ~ x, data = train)                     # Underfit
tree_model <- rpart(y ~ x, data = train, control = rpart.control(cp = 0.0001))  # Overfit
rf_model <- randomForest(y ~ x, data = train, ntree = 200)                        # Balanced

# Predictions on full dataset
data$lm_pred <- predict(lm_model, data)
data$tree_pred <- predict(tree_model, data)
data$rf_pred <- predict(rf_model, data)

# Plot with annotations
p <- ggplot(data, aes(x, y)) +
  geom_point(color = "gray", alpha = 0.6) +
  geom_line(aes(y = lm_pred, color = "Linear (Underfit)"), size = 1.2) +
  geom_line(aes(y = tree_pred, color = "Tree (Overfit)"), size = 1.2) +
  geom_line(aes(y = rf_pred, color = "Random Forest (Balanced)"), size = 1.2) +
  annotate("point", x = -2, y = lm_model$coef[1] + lm_model$coef[2]*(-2), shape = 21, size = 4, color = "blue", fill = "blue") +
  annotate("text", x = -2, y = lm_model$coef[1] + lm_model$coef[2]*(-2) - 0.5, label = "Underfit", color = "blue") +
  annotate("point", x = 2, y = predict(tree_model, newdata = data.frame(x=2)), shape = 24, size = 4, color = "red", fill = "red") +
  annotate("text", x = 2, y = predict(tree_model, newdata = data.frame(x=2)) + 0.5, label = "Overfit", color = "red") +
  scale_color_manual(values = c("Linear (Underfit)" = "blue",
                                "Tree (Overfit)" = "red",
                                "Random Forest (Balanced)" = "green")) +
  labs(title = "Underfit vs Overfit vs Random Forest",
       subtitle = "Annotated points highlight Underfit and Overfit regions",
       y = "Predicted y",
       x = "x",
       color = "Model") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(face = "italic", hjust = 0.5))

# Print plot
print(p)

# Save plot
ggsave("combined_model_fit.png", plot = p, width = 10, height = 6, dpi = 150)
cat("Combined plot saved as combined_model_fit.png\n")

# Underfit Example - Linear Model

library(ggplot2)
library(caret)

# Generate synthetic data
set.seed(42)
x <- seq(-3, 3, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)
data <- data.frame(x = x, y = y)

# Split data
trainIndex <- createDataPartition(data$y, p = 0.8, list = FALSE)
train <- data[trainIndex, ]
test <- data[-trainIndex, ]

# Linear model
lm_model <- lm(y ~ x, data = train)

# Predictions
data$lm_pred <- predict(lm_model, data)

# MSE
lm_mse <- mean((test$y - predict(lm_model, test))^2)
cat("Linear Model MSE (Underfitting):", lm_mse, "\n")

# Plot
p_under <- ggplot(data, aes(x, y)) +
  geom_point(color = "gray", alpha = 0.6) +
  geom_line(aes(y = lm_pred), color = "blue", size = 1.2) +
  annotate("point", x = 0, y = predict(lm_model, newdata = data.frame(x=0)), shape = 21, size = 4, color = "blue", fill = "blue") +
  annotate("text", x = 0, y = predict(lm_model, newdata = data.frame(x=0)) - 0.5, label = "Underfit", color = "blue") +
  labs(title = "Underfitting Example (Linear Model)", y = "Predicted y", x = "x") +
  theme_minimal(base_size = 14)

print(p_under)

# Save plot
ggsave("New_underfit_linear.png", plot = p_under, width = 8, height = 5, dpi = 150)
cat("Underfit plot saved as underfit_linear.png\n")

# Balanced Fit Example - Random Forest

library(ggplot2)
library(randomForest)
library(caret)

# Generate synthetic data
set.seed(42)
x <- seq(-3, 3, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)
data <- data.frame(x = x, y = y)

# Split data
trainIndex <- createDataPartition(data$y, p = 0.8, list = FALSE)
train <- data[trainIndex, ]
test <- data[-trainIndex, ]

# Random Forest model
set.seed(42)
rf_model <- randomForest(y ~ x, data = train, ntree = 200)

# Predictions
data$rf_pred <- predict(rf_model, data)

# MSE
rf_mse <- mean((test$y - predict(rf_model, test))^2)
cat("Random Forest MSE:", rf_mse, "\n")

# Plot
p_rf <- ggplot(data, aes(x, y)) +
  geom_point(color = "gray", alpha = 0.6) +
  geom_line(aes(y = rf_pred), color = "green", size = 1.2) +
  labs(title = "Balanced Fit Example (Random Forest)", y = "Predicted y", x = "x") +
  theme_minimal(base_size = 14)

print(p_rf)

# Save plot
ggsave("NEW_balanced_rf.png", plot = p_rf, width = 8, height = 5, dpi = 150)
cat("Random Forest plot saved as balanced_rf.png\n")

# Overfit Example - Decision Tree

library(ggplot2)
library(rpart)
library(caret)

# Generate synthetic data
set.seed(42)
x <- seq(-3, 3, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)
data <- data.frame(x = x, y = y)

# Split data
trainIndex <- createDataPartition(data$y, p = 0.8, list = FALSE)
train <- data[trainIndex, ]
test <- data[-trainIndex, ]

# Decision Tree (high complexity)
tree_model <- rpart(y ~ x, data = train, control = rpart.control(cp = 0.0001))

# Predictions
data$tree_pred <- predict(tree_model, data)

# MSE
tree_mse <- mean((test$y - predict(tree_model, test))^2)
cat("Decision Tree MSE (Overfitting):", tree_mse, "\n")

# Plot
p_over <- ggplot(data, aes(x, y)) +
  geom_point(color = "gray", alpha = 0.6) +
  geom_line(aes(y = tree_pred), color = "red", size = 1.2) +
  annotate("point", x = 1, y = predict(tree_model, newdata = data.frame(x=1)), shape = 24, size = 4, color = "red", fill = "red") +
  annotate("text", x = 1, y = predict(tree_model, newdata = data.frame(x=1)) + 0.5, label = "Overfit", color = "red") +
  labs(title = "Overfitting Example (Decision Tree)", y = "Predicted y", x = "x") +
  theme_minimal(base_size = 14)

print(p_over)

# Save plot
ggsave("NEW_overfit_tree.png", plot = p_over, width = 8, height = 5, dpi = 150)
cat("Overfit plot saved as overfit_tree.png\n")
