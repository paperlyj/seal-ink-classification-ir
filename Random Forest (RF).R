#===============================================================================
# Random Forest Classification with Optimal Tree Count and F1 Metrics
#  • Preprocessing: L2 normalization only vs. Savitzky–Golay 2nd derivative + L2
#  • mtry ∈ {sqrt(p), log2(p), p/3}
#  • ntree_max = 200 (grow this many trees to find optimum)
#  • Metrics: Train F1, OOB (validation) F1, Test F1, and optimal tree count
#===============================================================================

# 0. Load required packages
library(openxlsx)     # read.xlsx
library(prospectr)    # savitzkyGolay
library(randomForest) # randomForest
library(MLmetrics)    # F1_Score
library(dplyr)        # data manipulation

# 1. Read Train/Test data
data.Train <- read.xlsx("Selected_2.xlsx", sheet = "Train")
data.Test  <- read.xlsx("Selected_2.xlsx", sheet = "Test")
data.Train$Sample <- factor(data.Train$Sample)
data.Test$Sample  <- factor(data.Test$Sample)

X_train_raw <- as.matrix(data.Train[ , -1])
y_train     <- data.Train$Sample

X_test_raw  <- as.matrix(data.Test[ , -1])
y_test      <- data.Test$Sample

p <- ncol(X_train_raw)

# 2. Preprocessing functions
normalize_l2 <- function(x) x / sqrt(sum(x^2))

pre_L2 <- function(mat) {
  t(apply(mat, 1, normalize_l2))
}

pre_SG2_L2 <- function(mat) {
  sg <- t(apply(mat, 1, function(x) savitzkyGolay(x, m = 2, p = 3, w = 13)))
  t(apply(sg, 1, normalize_l2))
}

pipelines <- list(
  L2_only    = pre_L2,
  SG2_plus_L2 = pre_SG2_L2
)

# 3. Macro-F1 summary (one-vs-rest)
f1_macro_summary <- function(df, lev) {
  mean(sapply(lev, function(cl) {
    obs_bin  <- factor(ifelse(df$obs  == cl, cl, "other"),
                       levels = c(cl, "other"))
    pred_bin <- factor(ifelse(df$pred == cl, cl, "other"),
                       levels = c(cl, "other"))
    F1_Score(y_true = obs_bin, y_pred = pred_bin, positive = cl)
  }), na.rm = TRUE)
}

# 4. Hyperparameters
mtry_types <- c("sqrt", "log2", "div3")
ntree_max  <- 300

# 5. Prepare results storage
results_rf <- tibble(
  Pipeline    = character(),
  mtry_type   = character(),
  opt_trees   = integer(),
  Train_F1    = double(),
  Val_F1      = double(),  # OOB-based
  Test_F1     = double()
)

# 6. Loop over pipelines and mtry types
for (pl in names(pipelines)) {
  # Apply preprocessing
  Xtr <- pipelines[[pl]](X_train_raw)
  Xte <- pipelines[[pl]](X_test_raw)
  
  for (mtry_type in mtry_types) {
    # Determine mtry value
    mtry_val <- switch(mtry_type,
                       sqrt = floor(sqrt(p)),
                       log2 = floor(log2(p)),
                       div3 = floor(p / 3))
    
    # 6.1 Grow max trees to track OOB error
    rf_max <- randomForest(
      x          = Xtr,
      y          = y_train,
      ntree      = ntree_max,
      mtry       = mtry_val,
      keep.inbag = TRUE
    )
    
    # 6.2 Find optimal tree count (min OOB error)
    errs    <- rf_max$err.rate[ , "OOB"]
    opt_t   <- which.min(errs)
    
    # 6.3 Retrain RF with optimal tree count
    rf_opt <- randomForest(
      x     = Xtr,
      y     = y_train,
      ntree = opt_t,
      mtry  = mtry_val
    )
    
    # 6.4 Predictions
    pred_tr    <- predict(rf_opt, Xtr)
    pred_oob   <- rf_opt$predicted     # OOB predictions
    pred_te    <- predict(rf_opt, Xte)
    
    # 6.5 Compute F1 scores
    levs <- levels(y_train)
    f1_tr  <- f1_macro_summary(data.frame(obs = y_train,    pred = pred_tr),  levs)
    f1_val <- f1_macro_summary(data.frame(obs = y_train,    pred = pred_oob), levs)
    f1_te  <- f1_macro_summary(data.frame(obs = y_test,     pred = pred_te),  levs)
    
    # 6.6 Store results
    results_rf <- results_rf %>%
      add_row(
        Pipeline  = pl,
        mtry_type = mtry_type,
        opt_trees = opt_t,
        Train_F1  = round(f1_tr, 4),
        Val_F1    = round(f1_val,4),
        Test_F1   = round(f1_te, 4)
      )
  }
}

# 7. Display summary table
print(results_rf)
