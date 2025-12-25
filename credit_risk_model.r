#STEP 1: Loading data and analyze

#loading training dataset
credit_data <- read.csv("cs-training.csv", header = TRUE, sep = ",")
credit_data$X <- NULL

#Print number of columns and rows
print("Number of Columns: ")
ncol(credit_data)
print("Number of Rows: ")
nrow(credit_data)

#head(credit_data)

#checking columns var types
data.frame(Column = names(credit_data), type = sapply(credit_data, class))
# count total rows
n <- nrow(credit_data)

# count number of default cases (where target variable = 1)
default_count <- sum(credit_data$SeriousDlqin2yrs == 1)

# calculate imbalance percentage
percentage <- (default_count / n) 

print(default_count)
print(percentage * 100)

#checking missing values in all columns
missing_values <- data.frame(Column = names(credit_data), Missing = as.vector(colSums(is.na(credit_data))),
                            Percent = as.vector(colSums(is.na(credit_data))/nrow(credit_data)*100))
print(missing_values)

#we have two columns with missing data (Monthly_Income - 29731, Number_Of_Dependents - 3924)
#now i am gonna if missing values in these columns are important or not.

#install.packages("dplyr")
library(dplyr)

# 1. Check if missing income is random or relates to other financial variables - seriousdeli.. = past 90 days or worse = default rate
#and debt ratio, avg credit line, no of real estate loans or lines.
#the code below will calculate mean of each group where the income is missing vs income is not missing
income_analysis <- credit_data %>%
  mutate(Income_Missing = is.na(MonthlyIncome)) %>%
  group_by(Income_Missing) %>%
  summarise(
    Count = n(),
    Default_Rate = mean(SeriousDlqin2yrs, na.rm = TRUE),
    Avg_DebtRatio = mean(DebtRatio, na.rm = TRUE),
    Avg_CreditLines = mean(NumberOfOpenCreditLinesAndLoans, na.rm = TRUE),
    Avg_RealEstateLoans = mean(NumberRealEstateLoansOrLines, na.rm = TRUE),
      Avg_Age = mean(age, na.rm = TRUE)
  )

print(" INCOME MISSINGNESS ANALYSIS ")
print(income_analysis, n = Inf, width = Inf)

#intrpretation: for the missing income group, the debt service ratio is extremely high 1673 compare to only 26.6 for other group
#but it make sense since denominator is zero, so it pushes the number to infinity
#From the income missingness analysis:When income is not missing (FALSE): 120269 cases, default rate 6.95%, average debt ratio 26.6,
#average credit lines 8.76, average real estate loans (not shown in the output but in the data).
#When income is missing (TRUE): 29731 cases, default rate 5.61%, average debt ratio 1673, average credit lines 7.22.


# 2. Check if missing dependents relates to age or other factors
dependents_analysis <- credit_data %>%
  mutate(Dependents_Missing = is.na(NumberOfDependents)) %>%
  group_by(Dependents_Missing) %>%
  summarise(
    Count = n(),
    Default_Rate = mean(SeriousDlqin2yrs, na.rm = TRUE),
    Avg_Age = mean(age, na.rm = TRUE),
    Avg_Income = mean(MonthlyIncome, na.rm = TRUE)
  )

print(" DEPENDENTS MISSINGNESS ANALYSIS ")
print(dependents_analysis)

#From the dependents missingness analysis:
#When dependents are not missing (FALSE): 146076 cases, default rate 6.74%, average age 52.1, average income 6670.
#When dependents are missing (TRUE): 3924 cases, default rate 4.56%, average age 59.6, average income NaN (will investigate later first let me fix
#Income part)

#checking duplicate rows
sum(duplicated(credit_data))

#data cleaning

# Load required libraries
library(caret)
library(dplyr)

#-----------------------------------------------
# 1. STRATIFIED SPLIT (FIXED) and data cleaning
# ------------------------------------------------

set.seed(143)
#creating partition train and test data
trainIndex <- createDataPartition(credit_data$SeriousDlqin2yrs, p = 0.85, list = FALSE)
train.data <- credit_data[trainIndex, ]
test.data <- credit_data[-trainIndex, ]
cat("Training samples:", nrow(train.data), "\n")
cat("Test samples:", nrow(test.data), "\n")
cat("Training class distribution:", table(train.data$SeriousDlqin2yrs), "\n")

#average age for missing dependents is 62, so we can safely input as 0 and create binary flag
#to avoid data leakge first  create binary missing flag in training set and replace NA with 0
train.data$Dependents_Missing <- ifelse(is.na(train.data$NumberOfDependents), 1, 0)
train.data$NumberOfDependents[is.na(train.data$NumberOfDependents)] <- 0

#similar process for testing set
test.data$Dependents_Missing <- ifelse(is.na(test.data$NumberOfDependents), 1, 0)
test.data$NumberOfDependents[is.na(test.data$NumberOfDependents)] <- 0

#missing income rows have high debt ratio
#creating missing income flag for training and testing set
train.data$Income_Missing <- ifelse(is.na(train.data$MonthlyIncome), 1, 0)
test.data$Income_Missing <- ifelse(is.na(test.data$MonthlyIncome), 1, 0)

#creating median from training set and applies to both
median_income <- median(train.data$MonthlyIncome, na.rm = TRUE)
train.data$MonthlyIncome[is.na(train.data$MonthlyIncome)] <- median_income
test.data$MonthlyIncome[is.na(test.data$MonthlyIncome)] <- median_income  

#fixing high debt ratio
#creating high debt ratio flag in training and testing set
train.data$highdebtflag <- ifelse(train.data$DebtRatio > 1, 1, 0)
test.data$highdebtflag <- ifelse(test.data$DebtRatio > 1, 1, 0)
#capping extreme values
cap_value <- quantile(train.data$DebtRatio, 0.99, na.rm = TRUE)
#apply to both set
train.data$DebtRatio <- pmin(train.data$DebtRatio, cap_value)
test.data$DebtRatio <- pmin(test.data$DebtRatio, cap_value)

#capping pastdue col
#creating list of 3 col
past_due_cols <- c(
  "NumberOfTime30.59DaysPastDueNotWorse",
  "NumberOfTime60.89DaysPastDueNotWorse",
  "NumberOfTimes90DaysLate"
)
# Apply to TRAIN
#looping through each col
for (col in past_due_cols) {
  train.data[[paste0(col, "_extreme")]] <- ifelse(train.data[[col]] > 10, 1, 0)  #creating extreme value flag
  train.data[[col]] <- ifelse(train.data[[col]] > 10, 10, train.data[[col]])  #capping original value at 10
}
# Apply SAME transformation to TEST
for (col in past_due_cols) {
  test.data[[paste0(col, "_extreme")]] <- ifelse(test.data[[col]] > 10, 1, 0)
  test.data[[col]] <- ifelse(test.data[[col]] > 10, 10, test.data[[col]])
}

# Fix unrealistic ages < 18 using training median
age_median_train <- median(train.data$age[train.data$age >= 18], na.rm = TRUE)
train.data$age[train.data$age < 18] <- age_median_train
test.data$age[test.data$age < 18] <- age_median_train

#capping revolving utilization of credit after creating flag
cap <- 1.5
# TRAIN
train.data$RevolvingUtilization_extreme <- 
  ifelse(train.data$RevolvingUtilizationOfUnsecuredLines > cap_value, 1, 0)

train.data$RevolvingUtilizationOfUnsecuredLines <- 
  ifelse(train.data$RevolvingUtilizationOfUnsecuredLines > cap_value,
         cap_value,
         train.data$RevolvingUtilizationOfUnsecuredLines)

# TEST
test.data$RevolvingUtilization_extreme <- 
  ifelse(test.data$RevolvingUtilizationOfUnsecuredLines > cap_value, 1, 0)

test.data$RevolvingUtilizationOfUnsecuredLines <- 
  ifelse(test.data$RevolvingUtilizationOfUnsecuredLines > cap_value,
         cap_value,
         test.data$RevolvingUtilizationOfUnsecuredLines)

#applying log transformation to income
train.data$MonthlyIncome_log <- log(train.data$MonthlyIncome + 1)
test.data$MonthlyIncome_log <- log(test.data$MonthlyIncome + 1)

#----------------------------------correlation matrix --------------------------
#-------------------------------------------------------------------------------

print(" CORRELATION ANALYSIS (TRAINING DATA) ")

vars <- c(
  "SeriousDlqin2yrs",  
  "age", 
  "DebtRatio", 
  "MonthlyIncome",
  "NumberOfOpenCreditLinesAndLoans",
  "NumberRealEstateLoansOrLines",
  "NumberOfDependents",
  "Income_Missing",
  "MonthlyIncome_log",
  "Dependents_Missing", 
  "RevolvingUtilizationOfUnsecuredLines",
  "NumberOfTime30.59DaysPastDueNotWorse",
  "NumberOfTimes90DaysLate",
  "NumberOfTime60.89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate_extreme",
    "NumberOfTime30.59DaysPastDueNotWorse_extreme",
    "NumberOfTime60.89DaysPastDueNotWorse_extreme",
    "RevolvingUtilization_extreme"
)

train_subset <- train.data[vars]

corr_matrix <- cor(train_subset, method = "pearson", use = "complete.obs")
round(corr_matrix, 4)

#-------------------------------BASELINE WEIGHTED LOGISTIC MODEL---------------------
#---------------------------------------------------------------------------------------

# WEIGHTED LOGISTIC REGRESSION
# Convert with explicit levels to ensure both are always present
train.data$SeriousDlqin2yrs <- factor(train.data$SeriousDlqin2yrs, levels = c(0,1), labels = c("NonDefault","Default"))
test.data$SeriousDlqin2yrs  <- factor(test.data$SeriousDlqin2yrs, levels = c(0,1), labels = c("NonDefault","Default"))

# Load  necessary libraries
library(caret)                          #For Machine learning and model training
library(pROC)                            #for ROC curves and AUC
library(dplyr)                        #for data manipulation 


#  Automatically calculate weight ratio - Calculates imbalance ratio: non-default count ÷ default count
ratio <- sum(train.data$SeriousDlqin2yrs == "NonDefault") / 
          sum(train.data$SeriousDlqin2yrs == "Default")

# Assigns higher weight to default cases (ratio) and weight 1 to non-defaults
model_weights <- ifelse(train.data$SeriousDlqin2yrs == "Default", ratio, 1)
cat(sprintf("🔹 Class weight ratio: %.2f\n", ratio))  #printing ratio
#ratio is 13.98 means default are weighted 14 times more

#Default cases get weight = 13.98
#NonDefault cases get weight = 1
#Effect: Each default "counts" as 14 non-defaults in the model training

train.data$age_sq <- train.data$age^2
train.data$debtratio_sq <- train.data$DebtRatio^2

test.data$age_sq <- test.data$age^2
test.data$debtratio_sq <- test.data$DebtRatio^2
# Train weighted logistic regression model
weighted_model <- train(
  x = train.data[, c( "RevolvingUtilizationOfUnsecuredLines", 
  "RevolvingUtilization_extreme", 
  "DebtRatio", "debtratio_sq",
  "highdebtflag",
  "MonthlyIncome_log",
  "Income_Missing", 
  "NumberOfDependents",
  "Dependents_Missing",
  "age",  "age_sq",
  "NumberOfOpenCreditLinesAndLoans",
  "NumberRealEstateLoansOrLines",
  "NumberOfTimes90DaysLate",
  "NumberOfTimes90DaysLate_extreme" )],   
  y = train.data$SeriousDlqin2yrs,   #specifying target variables
  method = "glm",                    #logistic 
  family = "binomial",
  weights = model_weights,    #applies calculated class weight
  trControl = trainControl(method = "cv", number = 5, classProbs = TRUE)   #5-fold cross-validation, returns class probabilities
)


# Predict probabilities on test set - Generates probability scores for "Default" class on test data
test.data$weighted_pred <- predict(weighted_model, newdata = test.data, type = "prob")[, "Default"]

# Function to calculate metrics
# Converts probabilities to binary predictions using threshold
calculate_metrics <- function(predictions, actual, threshold = 0.1) {
  pred_class <- ifelse(predictions > threshold, 1, 0)  
  pred_class <- factor(pred_class, levels = c(0, 1))      # Ensures factor has both levels (0 and 1)
  actual <- factor(ifelse(actual == "Default", 1, 0), levels = c(0, 1))   # Converts actual values to 1/0 and ensures factor levels
  
  cm <- table(Predicted = pred_class, Actual = actual)   #creates confusion matrix
  
  TP <- ifelse(nrow(cm) >= 2 && ncol(cm) >= 2, cm[2, 2], 0)
  TN <- ifelse(nrow(cm) >= 1 && ncol(cm) >= 1, cm[1, 1], 0)
  FP <- ifelse(nrow(cm) >= 2 && ncol(cm) >= 1, cm[2, 1], 0)
  FN <- ifelse(nrow(cm) >= 1 && ncol(cm) >= 2, cm[1, 2], 0)
  
  accuracy <- (TP + TN) / (TP + TN + FP + FN)   #calculates performance matrix
  precision <- ifelse((TP + FP) > 0, TP / (TP + FP), 0)
  recall <- ifelse((TP + FN) > 0, TP / (TP + FN), 0)
  f1 <- ifelse((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0)
  
  return(c(accuracy, precision, recall, f1))
}

#--------------------------------THRESHOLD EVALUATION--------------------------------------

# Evaluate across thresholds - Creates sequence of thresholds from 0.05 to 0.50 in 0.05 increments
thresholds <- seq(0.01, 0.65, by = 0.03)
results <- data.frame()  # Initializes empty data frame to store results

for (thresh in thresholds) {
  m <- calculate_metrics(test.data$weighted_pred, test.data$SeriousDlqin2yrs, thresh)   # Calculates metrics for current threshold
  results <- rbind(results, data.frame(
    Threshold = thresh,
    Accuracy = m[1],
    Precision = m[2],
    Recall = m[3],
    F1 = m[4]
  ))
}

# --------------------------------------AUC AND ROC PLOT--------------------------------------------
actual_numeric <- as.numeric(test.data$SeriousDlqin2yrs) - 1  # Converts "NonDefault"=0, "Default"=1
roc_weighted <- roc(actual_numeric, test.data$weighted_pred)
auc_val <- auc(roc_weighted)

# Print results
cat("\n=== WEIGHTED LOGISTIC REGRESSION PERFORMANCE ===\n")
print(results %>% mutate(across(-Threshold, ~ round(. * 100, 2))))
cat(sprintf("\nAUC: %.4f\n", auc_val))

# Plot ROC
plot(roc_weighted, main = "ROC Curve - Weighted Logistic Regression")
text(0.5, 0.3, paste("AUC =", round(auc_val, 4)), col = "blue")
thresholds <- seq(0.05, 0.5, by = 0.05)
results <- data.frame()

# -------------------------------------------WEIGHTED AND BALANCED RANDOM FOREST ---------------------------------------
# --------------------------------------------

library(randomForest)
library(ranger)
library(pROC)
library(dplyr)


# Define predictor columns - ALL FEATURES from train_final (except target)
predictor_cols <- names(train.data)[names(train.data) != "SeriousDlqin2yrs"]
cat("Using", length(predictor_cols), "features for Random Forest\n")
predictor_cols <- predictor_cols[predictor_cols != "X"]
# Calculate class weights
ratio <- sum(train.data$SeriousDlqin2yrs == "NonDefault") / 
         sum(train.data$SeriousDlqin2yrs == "Default")
cat(sprintf("Class imbalance ratio: %.2f\n", ratio))

# ===== WEIGHTED RANDOM FOREST =====
cat("\n=== TRAINING WEIGHTED RANDOM FOREST ===\n")

class_weights <- c(
  "NonDefault" = 1,
  "Default" = ratio
)

set.seed(143)
weighted_rf <- randomForest(
  x = train.data[, predictor_cols],           # ALL FEATURES from train_final
  y = train.data$SeriousDlqin2yrs,
  ntree = 150,
  mtry = floor(sqrt(length(predictor_cols))),
  classwt = class_weights,
  importance = TRUE,
  strata = train.data$SeriousDlqin2yrs,
  sampsize = rep(min(table(train.data$SeriousDlqin2yrs)), 2)  # Balanced sampling
)

# Predict
test.data$weighted_rf_pred <- predict(weighted_rf, newdata = test.data, type = "prob")[, "Default"]

# ===== BALANCED RANDOM FOREST =====
cat("\n=== TRAINING BALANCED RANDOM FOREST ===\n")

# Create formula using ALL FEATURES
formula <- as.formula(paste("SeriousDlqin2yrs ~", paste(predictor_cols, collapse = " + ")))

set.seed(123)
balanced_rf <- ranger(
  formula = formula,
  data = train.data,                          # Using train_final
  num.trees = 150,
  mtry = floor(sqrt(length(predictor_cols))),
  probability = TRUE,
  replace = TRUE,
  sample.fraction = 0.7,
  strata = train.data$SeriousDlqin2yrs,
  classification = TRUE,
  importance = "impurity"
)

# Predict
test.data$balanced_rf_pred <- predict(balanced_rf, data = test.data)$predictions[, "Default"]

# ===== EVALUATION =====
cat("\n=== EVALUATING MODELS ===\n")

# Use your existing evaluation function
calculate_metrics <- function(predictions, actual, threshold = 0.1) {
  pred_class <- ifelse(predictions > threshold, 1, 0)  
  pred_class <- factor(pred_class, levels = c(0, 1))
  actual <- factor(ifelse(actual == "Default", 1, 0), levels = c(0, 1))
  
  cm <- table(Predicted = pred_class, Actual = actual)
  
  TP <- ifelse(nrow(cm) >= 2 && ncol(cm) >= 2, cm[2, 2], 0)
  TN <- ifelse(nrow(cm) >= 1 && ncol(cm) >= 1, cm[1, 1], 0)
  FP <- ifelse(nrow(cm) >= 2 && ncol(cm) >= 1, cm[2, 1], 0)
  FN <- ifelse(nrow(cm) >= 1 && ncol(cm) >= 2, cm[1, 2], 0)
  
  accuracy <- (TP + TN) / (TP + TN + FP + FN)
  precision <- ifelse((TP + FP) > 0, TP / (TP + FP), 0)
  recall <- ifelse((TP + FN) > 0, TP / (TP + FN), 0)
  f1 <- ifelse((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0)
  
  return(c(accuracy, precision, recall, f1))
}

evaluate_model <- function(predictions, model_name) {
  thresholds <- seq(0.05, 0.65, by = 0.01)
  results <- data.frame()
  
  for (thresh in thresholds) {
    m <- calculate_metrics(predictions, test.data$SeriousDlqin2yrs, thresh)
    results <- rbind(results, data.frame(
      Threshold = thresh,
      Accuracy = m[1],
      Precision = m[2],
      Recall = m[3],
      F1 = m[4]
    ))
  }
  
  # Calculate AUC
  actual_numeric <- ifelse(test.data$SeriousDlqin2yrs == "Default", 1, 0)
  auc_val <- auc(roc(actual_numeric, predictions))
  
  cat(sprintf("\n%s PERFORMANCE:\n", toupper(model_name)))
  print(results %>% mutate(across(-Threshold, ~ round(. * 100, 2))))
  cat(sprintf("AUC: %.4f\n", auc_val))
  
  return(list(results = results, auc = auc_val))
}

# Evaluate both models
weighted_results <- evaluate_model(test.data$weighted_rf_pred, "Weighted Random Forest")
balanced_results <- evaluate_model(test.data$balanced_rf_pred, "Balanced Random Forest")

# ===== COMPARISON =====
cat("\n=== MODEL COMPARISON ===\n")
cat(sprintf("Weighted RF Best F1: %.2f%%\n", max(weighted_results$results$F1) * 100))
cat(sprintf("Balanced RF Best F1: %.2f%%\n", max(balanced_results$results$F1) * 100))
cat(sprintf("Weighted RF AUC: %.4f\n", weighted_results$auc))
cat(sprintf("Balanced RF AUC: %.4f\n", balanced_results$auc))

# Feature importance
cat("\n=== FEATURE IMPORTANCE (Weighted RF) ===\n")
importance_df <- data.frame(
  Feature = rownames(weighted_rf$importance),
  Importance = weighted_rf$importance[, "MeanDecreaseGini"]
) %>% arrange(desc(Importance))

print(head(importance_df, 15))  # Show top 15 features

# Plot feature importance
varImpPlot(weighted_rf, n.var = 15, main = "Top 15 Feature Importance - Random Forest")

# ====================================
# XGBOOST WITH THRESHOLD EVALUATION
# ====================================

library(xgboost)
library(pROC)
library(dplyr)

# 1. PREPARE DATA FOR XGBOOST
# ============================

predictor_cols <- setdiff(names(train.data), "SeriousDlqin2yrs")

x_train <- as.matrix(train.data[, predictor_cols])
x_test  <- as.matrix(test.data[, predictor_cols])

y_train <- as.numeric(train.data$SeriousDlqin2yrs == "Default")
y_test  <- as.numeric(test.data$SeriousDlqin2yrs == "Default")

cat("Target distribution - Train:", table(y_train), "\n")
cat("Target distribution - Test:", table(y_test), "\n")

# 2. RUN XGBOOST
# ============================

scale_pos_weight <- sum(y_train == 0) / sum(y_train == 1)
cat("Scale_pos_weight:", round(scale_pos_weight, 2), "\n")

params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  max_depth = 6,
  eta = 0.1,
  scale_pos_weight = scale_pos_weight
)

set.seed(143)
xgb_model <- xgboost(
  data = x_train,
  label = y_train,
  params = params,
  nrounds = 100,
  verbose = 0
)

# 3. PREDICT
# ============================

xgb_pred_probs <- predict(xgb_model, x_test)

cat("\nPrediction range:", range(xgb_pred_probs), "\n")
cat("Predictions:", length(xgb_pred_probs), "\n")

# 4. THRESHOLD EVALUATION
# ============================

thresholds <- seq(0.01, 0.6, by = 0.01)
results <- data.frame()

for (thresh in thresholds) {
  preds <- ifelse(xgb_pred_probs >= thresh, 1, 0)
  cm <- table(Predicted = preds, Actual = y_test)

  if (all(dim(cm) == c(2, 2))) {
    TP <- cm[2,2]; FP <- cm[2,1]; FN <- cm[1,2]; TN <- cm[1,1]

    precision <- TP / (TP + FP)
    recall <- TP / (TP + FN)
    f1 <- 2 * precision * recall / (precision + recall)

    results <- rbind(results, data.frame(
      Threshold = thresh,
      Precision = precision,
      Recall = recall,
      F1 = f1,
      True_Positives = TP,
      False_Positives = FP,
      False_Negatives = FN
    ))
  }
}

# 5. SHOW PERFORMANCE SUMMARY
# ============================

cat("\n=== XGBOOST PERFORMANCE SUMMARY ===\n")
cat("Actual defaults in test:", sum(y_test), "\n")
cat("AUC:", round(auc(roc(y_test, xgb_pred_probs)), 4), "\n\n")

key_thresholds <- c(0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.53, 0.56, 0.58, 0.60)

cat("Threshold | Recall | Precision | F1 | TP | FP | FN\n")
cat("----------|--------|-----------|----|----|----|----\n")

for (t in key_thresholds) {
  idx <- which.min(abs(results$Threshold - t))
  row <- results[idx, ]

  cat(sprintf("   %.2f   | %5.1f%% | %8.1f%% | %4.1f | %3d | %3d | %3d\n",
              row$Threshold,
              row$Recall * 100,
              row$Precision * 100,
              row$F1 * 100,
              row$True_Positives,
              row$False_Positives,
              row$False_Negatives))
}

#-----------------------------------------LIGHTGBM - BEST PERFORMER---------------------------
#-------------------------------------------------------------------------------------------------

library(lightgbm)
library(pROC)
library(dplyr)

# ---------------------------
# 1. Prepare data
# ---------------------------
predictor_cols <- setdiff(names(train.data), "SeriousDlqin2yrs")

X <- as.matrix(train.data[, predictor_cols])
y <- as.numeric(train.data$SeriousDlqin2yrs == "Default")

dtrain <- lgb.Dataset(data = X, label = y)

# Class imbalance handling
scale_pos_weight <- sum(y == 0) / sum(y == 1)

# ---------------------------
# 2. Define hyperparameters
# ---------------------------
params <- list(
  objective = "binary", boosting_type = 'gbdt', 
  metric = "auc",
  learning_rate = 0.02,
  num_leaves = 40,
  max_depth = -1,
  min_data_in_leaf = 40,
  feature_fraction = 0.9,
  bagging_fraction = 0.9,
  bagging_freq = 1,
  lambda_l1 = 0.0,
  lambda_l2 = 1.0,
  scale_pos_weight = scale_pos_weight
)

# ---------------------------
# 3. Run 5-fold cross-validation
# ---------------------------
set.seed(143)

cv_results <- lgb.cv(
  params = params,
  data = dtrain,
  nrounds = 3000,
  nfold = 5,
  stratified = TRUE,   # IMPORTANT for class imbalance
  early_stopping_rounds = 80,
  verbose = 1
)

# Best iteration (based on validation folds)
best_iter <- cv_results$best_iter
best_iter
final_model <- lgb.train(
  params = params,
  data = dtrain,
  nrounds = best_iter
)
x_test <- as.matrix(test.data[, predictor_cols])
y_test <- as.numeric(test.data$SeriousDlqin2yrs == "Default")

pred_test <- predict(final_model, x_test)

test_auc <- roc(y_test, pred_test)$auc
test_auc
cv_auc <- cv_results$best_score
test_auc <- roc(y_test, pred_test)$auc
cat(sprintf("CV AUC: %.4f, Test AUC: %.4f\n", cv_auc, test_auc))

# ------------------------------------------
# Threshold metrics
# ------------------------------------------
thresholds <- seq(0.1, 0.9, by = 0.01)

results <- data.frame()

for (t in thresholds) {
  pred_class <- ifelse(pred_test >= t, 1, 0)
  
  TP <- sum(pred_class == 1 & y_test == 1)
  FP <- sum(pred_class == 1 & y_test == 0)
  FN <- sum(pred_class == 0 & y_test == 1)
  TN <- sum(pred_class == 0 & y_test == 0)
  
  Accuracy  <- (TP + TN) / (TP + FP + FN + TN)
  Precision <- TP / (TP + FP)
  Recall    <- TP / (TP + FN)
  F1        <- 2 * Precision * Recall / (Precision + Recall)
  
  results <- rbind(results, data.frame(
    Threshold = t,
    Accuracy  = round(Accuracy, 4),
    Precision = round(Precision, 4),
    Recall    = round(Recall, 4),
    F1        = round(F1, 4),
    TP = TP, FP = FP, FN = FN, TN = TN
  ))
}

print(results)
# ------------------------------------------
# ROC Curve
# ------------------------------------------
roc_obj <- roc(y_test, pred_test)

plot(
  roc_obj,
  main = sprintf("ROC Curve (AUC = %.4f)", auc(roc_obj)),
  col = "blue",
  lwd = 2
)

abline(a = 0, b = 1, col = "red", lty = 2)

# ------------------------------------------
# Precision-Recall Curve
# ------------------------------------------
library(PRROC)

pr_obj <- pr.curve(
  scores.class0 = pred_test[y_test == 1],
  scores.class1 = pred_test[y_test == 0],
  curve = TRUE
)

plot(
  pr_obj$curve[, 1], pr_obj$curve[, 2],
  type = "l", col = "darkgreen", lwd = 2,
  xlab = "Recall", ylab = "Precision",
  main = sprintf("Precision-Recall Curve (AUC = %.4f)", pr_obj$auc.integral)
)
