#===============================================================================
# SVM (RBF) Performance Comparison with L2 Normalization and
# Savitzky–Golay 2nd Derivative Preprocessing
#===============================================================================

# 0. Set working directory and load packages
# (Adjust path if needed; here we assume you run from repo root)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path) %||% ".")
library(openxlsx)    # reading Excel files
library(prospectr)    # Savitzky–Golay filtering
library(caret)       # model training and tuning
library(MLmetrics)   # F1 score computation
library(kernlab)     # SVM (RBF)

# 1. Load data
file_path  <- "../data/Selected_1.xlsx"
data.Train <- read.xlsx(file_path, sheet = "Train")
data.Test  <- read.xlsx(file_path, sheet = "Test")
data.Train$Sample <- factor(data.Train$Sample)
data.Test$Sample  <- factor(data.Test$Sample)

# 2. Define preprocessing functions
normalize_l2 <- function(x) x / sqrt(sum(x^2))

preprocess_norm_only <- function(df) {
  X_norm <- t(apply(as.matrix(df[,-1]), 1, normalize_l2))
  data.frame(Sample = df$Sample, X_norm, check.names = FALSE)
}

preprocess_sg2_l2 <- function(df) {
  X          <- as.matrix(df[,-1])
  sg2_matrix <- t(apply(X, 1, function(x) savitzkyGolay(x, m=2, p=3, w=13)))
  X_norm     <- t(apply(sg2_matrix, 1, normalize_l2))
  data.frame(Sample = df$Sample, X_norm, check.names = FALSE)
}

# 3. Macro‐F1 summary function for caret
f1_macro_summary <- function(data, lev = NULL, model = NULL) {
  f1_scores <- sapply(lev, function(cl) {
    obs_bin  <- factor(ifelse(data$obs == cl, cl, "other"), levels = c(cl, "other"))
    pred_bin <- factor(ifelse(data$pred == cl, cl, "other"), levels = c(cl, "other"))
    F1_Score(y_true = obs_bin, y_pred = pred_bin, positive = cl)
  })
  c(F1 = mean(f1_scores, na.rm = TRUE))
}

# 4. 3‐fold CV setup & tuning grid
set.seed(2025)
train_ctrl <- trainControl(
  method           = "cv",
  number           = 3,
  classProbs       = TRUE,
  summaryFunction  = f1_macro_summary,
  savePredictions  = "final"
)

tune_grid_svm <- expand.grid(
  sigma = 10^seq(-1, -6, by = -1),
  C     = 10^seq( 0,  5, by =  1)
)

# 5. Run pipelines and collect results
pipelines <- list(
  "L2_norm_only"     = preprocess_norm_only,
  "SG2_deriv + L2"   = preprocess_sg2_l2
)

results_svm <- data.frame(
  Pipeline   = character(),
  Best_sigma = numeric(),
  Best_C     = numeric(),
  Train_F1   = numeric(),
  CV_F1      = numeric(),
  Test_F1    = numeric(),
  stringsAsFactors = FALSE
)

for (name in names(pipelines)) {
  cat("Running pipeline:", name, "\n")
  
  train_df <- pipelines[[name]](data.Train)
  test_df  <- pipelines[[name]](data.Test)
  
  set.seed(2025)
  fit_svm <- train(
    Sample ~ ., data      = train_df,
    method    = "svmRadial",
    tuneGrid  = tune_grid_svm,
    metric    = "F1",
    trControl = train_ctrl
  )
  
  best_sigma <- fit_svm$bestTune$sigma
  best_C     <- fit_svm$bestTune$C
  cv_f1      <- fit_svm$results[
    fit_svm$results$sigma == best_sigma &
      fit_svm$results$C     == best_C, "F1"]
  
  # Compute Train F1
  pred_train <- predict(fit_svm, newdata = train_df)
  train_f1   <- f1_macro_summary(
    data.frame(obs = train_df$Sample, pred = pred_train),
    lev = levels(train_df$Sample)
  )
  
  # Compute Test F1
  test_df$Sample <- factor(test_df$Sample, levels = levels(train_df$Sample))
  pred_test      <- predict(fit_svm, newdata = test_df)
  test_f1        <- f1_macro_summary(
    data.frame(obs = test_df$Sample, pred = pred_test),
    lev = levels(train_df$Sample)
  )
  
  results_svm <- rbind(
    results_svm,
    data.frame(
      Pipeline   = name,
      Best_sigma = best_sigma,
      Best_C     = best_C,
      Train_F1   = round(train_f1, 4),
      CV_F1      = round(cv_f1,    4),
      Test_F1    = round(test_f1,  4),
      stringsAsFactors = FALSE
    )
  )
}

# 6. Print results
print(results_svm)
