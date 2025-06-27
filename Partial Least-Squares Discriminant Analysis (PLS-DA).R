#===============================================================================
# PLS-DA Performance Comparison with L2 Normalization and
# Savitzky–Golay 2nd Derivative Preprocessing
#-------------------------------------------------------------------------------
# This script reads in training and test data, applies two preprocessing pipelines:
#  1. L2 normalization only
#  2. Savitzky–Golay 2nd derivative followed by L2 normalization
# It then performs 3-fold cross-validated PLS-DA model tuning using macro F1 score,
# and compares training, CV, and test F1 performances for each pipeline.
#===============================================================================

# 0. Working directory & package loading --------------------------------------
# Set your working directory as needed
setwd("C:/Users/paper/OneDrive/Desktop/PaperLab/Manuscripts/2024_Forensic/Red_Stamp")

library(openxlsx)    # for reading Excel files
library(prospectr)    # for Savitzky–Golay filtering
library(caret)        # for model training and tuning
library(MLmetrics)    # for F1 score computation
library(pls)          # for PLS-DA support

# 1. Data import ---------------------------------------------------------------
train_file <- "Selected_1.xlsx"
data.Train  <- read.xlsx(train_file, sheet = "Train")
data.Test   <- read.xlsx(train_file, sheet = "Test")

# Ensure 'Sample' column is a factor
data.Train$Sample <- factor(data.Train$Sample)
data.Test$Sample  <- factor(data.Test$Sample)

# 2. Preprocessing functions --------------------------------------------------
# L2 normalization: scales each sample vector to unit length
normalize_l2 <- function(x) {
  x / sqrt(sum(x^2))
}

# Pipeline A: L2 normalization only
dpreprocess_norm_only <- function(df) {
  X_norm <- t(apply(as.matrix(df[,-1]), 1, normalize_l2))
  data.frame(Sample = df$Sample, X_norm, check.names = FALSE)
}

# Pipeline B: Savitzky–Golay 2nd derivative + L2 normalization
dpreprocess_sg2_l2 <- function(df) {
  X         <- as.matrix(df[,-1])
  sg2_matrix <- t(apply(X, 1, function(x) savitzkyGolay(x, m = 2, p = 3, w = 13)))
  X_norm    <- t(apply(sg2_matrix, 1, normalize_l2))
  data.frame(Sample = df$Sample, X_norm, check.names = FALSE)
}

# 3. Macro F1 summary function ------------------------------------------------
f1_macro_summary <- function(data, lev = NULL, model = NULL) {
  f1_scores <- sapply(lev, function(cl) {
    obs_bin  <- factor(ifelse(data$obs == cl, cl, "other"), levels = c(cl, "other"))
    pred_bin <- factor(ifelse(data$pred == cl, cl, "other"), levels = c(cl, "other"))
    F1_Score(y_true = obs_bin, y_pred = pred_bin, positive = cl)
  })
  c(F1 = mean(f1_scores, na.rm = TRUE))
}

# 4. Cross-validation setup and tuning grid -----------------------------------
set.seed(2025)
train_ctrl <- trainControl(
  method          = "cv",
  number          = 3,
  classProbs      = TRUE,
  summaryFunction = f1_macro_summary,
  savePredictions = "final"
)

tune_grid <- expand.grid(ncomp = 1:10)

# 5. Run pipelines and collect results ----------------------------------------
pipelines <- list(
  "L2_normalization_only"        = dpreprocess_norm_only,
  "SavitzkyGolay2nd + L2_norm" = dpreprocess_sg2_l2
)

results <- data.frame(
  Pipeline   = character(),
  Best_ncomp = integer(),
  Train_F1   = numeric(),
  CV_F1      = numeric(),
  Test_F1    = numeric(),
  stringsAsFactors = FALSE
)

for (name in names(pipelines)) {
  cat("Processing pipeline:", name, "\n")
  
  # Apply preprocessing
  train_df <- pipelines[[name]](data.Train)
  test_df  <- pipelines[[name]](data.Test)
  
  # Train PLS-DA with CV tuning
  set.seed(2025)
  fit <- train(
    Sample ~ ., data = train_df,
    method    = "pls",
    preProc   = c("center", "scale"),
    tuneGrid  = tune_grid,
    metric    = "F1",
    trControl = train_ctrl
  )
  
  # Extract best component and CV F1
  best_n <- fit$bestTune$ncomp
  cv_f1  <- fit$results[fit$results$ncomp == best_n, "F1"]
  
  # Compute training F1
  pred_train <- predict(fit, newdata = train_df)
  df_train   <- data.frame(obs = train_df$Sample, pred = pred_train)
  train_f1   <- f1_macro_summary(df_train, lev = levels(df_train$obs))
  
  # Compute test F1
  # Align factor levels with training
  test_df$Sample <- factor(test_df$Sample, levels = levels(train_df$Sample))
  pred_test <- predict(fit, newdata = test_df)
  df_test   <- data.frame(obs = test_df$Sample, pred = pred_test)
  test_f1   <- f1_macro_summary(df_test, lev = levels(df_test$obs))
  
  # Store results
  results <- rbind(
    results,
    data.frame(
      Pipeline   = name,
      Best_ncomp = best_n,
      Train_F1   = round(train_f1, 4),
      CV_F1      = round(cv_f1,    4),
      Test_F1    = round(test_f1,  4),
      stringsAsFactors = FALSE
    )
  )
}

# 6. Display comparison table ------------------------------------------------
print(results)