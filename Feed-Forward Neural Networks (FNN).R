#===============================================================================
# fnn_performance_comparison.R
#   Dense (FNN) NN Comparison on IR Spectra of Red-Stamp Inks
#   • L2 vs. Savitzky–Golay 2nd-derivative + L2 preprocessing
#   • 3-fold CV grid-search (lr, optimizer, hidden_units)
#   • Additional validation: 30-seed repeats, y-scrambling, ROC/AUC
#   • Unknown-sample probability prediction
#===============================================================================

# 0. SETUP ----------------------------------------------------------------------
setwd("C:/Users/paper/OneDrive/Desktop/제지실험실/논문/2024_Forensic/2024.12_Red Stamp")

library(reticulate); use_condaenv("tf_gpu", required = TRUE)
library(keras); library(caret); library(MLmetrics); library(openxlsx)
library(prospectr); library(pROC); library(PRROC)
library(tibble); library(ggplot2)

# 1. HELPERS -------------------------------------------------------------------
# L2 normalization
normalize_l2 <- function(x) x / sqrt(sum(x^2))

# Preprocessing pipelines
preprocess_norm_l2 <- function(df) {
  Xn <- t(apply(df[,-1], 1, normalize_l2))
  data.frame(Sample = df$Sample, Xn, check.names = FALSE)
}

preprocess_sg2_l2 <- function(df) {
  X   <- as.matrix(df[,-1])
  sg  <- t(apply(X, 1, function(x) savitzkyGolay(x, m=2, p=3, w=13)))
  Xn  <- t(apply(sg, 1, normalize_l2))
  data.frame(Sample = df$Sample, Xn, check.names = FALSE)
}

# Macro-F1 summary for one-vs-rest
f1_macro <- function(data, lev) {
  scores <- sapply(lev, function(cl) {
    obs_bin  <- factor(ifelse(data$obs==cl, cl, "other"), levels=c(cl,"other"))
    pred_bin <- factor(ifelse(data$pred==cl, cl, "other"), levels=c(cl,"other"))
    F1_Score(y_true=obs_bin, y_pred=pred_bin, positive=cl)
  })
  mean(scores, na.rm=TRUE)
}

# Build an FNN given hyperparams
build_fnn <- function(input_dim, n_classes, hu, opt_fn) {
  model <- keras_model_sequential() %>%
    layer_dense(units=hu, activation="relu", input_shape=input_dim) %>%
    layer_dense(units=n_classes, activation="softmax")
  model %>% compile(loss="categorical_crossentropy", optimizer=opt_fn)
  model
}

# 2. LOAD DATA -----------------------------------------------------------------
data_train <- read.xlsx("Selected_1.xlsx", sheet="Train")
data_test  <- read.xlsx("Selected_1.xlsx", sheet="Test")
data_train$Sample <- factor(data_train$Sample)
data_test$Sample  <- factor(data_test$Sample)

# 3. CONFIGURATION -------------------------------------------------------------
pipelines      <- list(L2_norm=preprocess_norm_l2, SG2_plus_L2=preprocess_sg2_l2)
learning_rates <- c(1e-4,1e-3,1e-2,1e-1)
optimizers     <- c("adam","sgd")
units_list     <- c(16,32,64,128,256,512)

results <- data.frame(
  Pipeline=character(), lr=numeric(), optimizer=character(), hidden_units=integer(),
  Train_F1=numeric(), CV_F1=numeric(), Test_F1=numeric(), stringsAsFactors=FALSE
)

# 4. GRID SEARCH + 3-FOLD CV ---------------------------------------------------
for (pl in names(pipelines)) {
  # Preprocess
  train_df <- pipelines[[pl]](data_train)
  test_df  <- pipelines[[pl]](data_test)
  
  X_tr <- as.matrix(train_df[,-1]); y_tr <- as.integer(train_df$Sample)-1
  X_te <- as.matrix(test_df[,-1]);  y_te <- as.integer(test_df$Sample)-1
  n_classes <- nlevels(train_df$Sample)
  y_tr_cat  <- to_categorical(y_tr, n_classes)
  
  folds <- createFolds(train_df$Sample, k=3)
  best_cv <- -Inf; best_params <- list()
  
  for (lr in learning_rates) for (opt in optimizers) for (hu in units_list) {
    cv_scores <- c()
    
    for (fold in folds) {
      tr_idx <- setdiff(seq_len(nrow(X_tr)), fold)
      X_train_f <- X_tr[tr_idx,,drop=FALSE]; Y_train_f <- y_tr_cat[tr_idx,,drop=FALSE]
      X_val     <- X_tr[fold,,drop=FALSE]
      y_val_obs <- factor(train_df$Sample[fold], levels=levels(train_df$Sample))
      
      # Build & train
      opt_fn <- switch(opt, adam=optimizer_adam(lr), sgd=optimizer_sgd(lr))
      model  <- build_fnn(ncol(X_tr), n_classes, hu, opt_fn)
      model %>% fit(X_train_f, Y_train_f, epochs=50, batch_size=32, verbose=0)
      
      # Predict & F1
      preds     <- model %>% predict(X_val)
      pred_cls  <- apply(preds,1,which.max)-1
      pred_fac  <- factor(levels(train_df$Sample)[pred_cls+1], levels=levels(train_df$Sample))
      cv_scores <- c(cv_scores, f1_macro(data.frame(obs=y_val_obs,pred=pred_fac), levels(train_df$Sample)))
      
      k_clear_session()
    }
    
    cv_mean <- mean(cv_scores, na.rm=TRUE)
    if (!is.nan(cv_mean) && cv_mean > best_cv) {
      best_cv <- cv_mean
      best_params <- list(lr=lr,opt=opt,hu=hu)
    }
  }
  
  # Retrain final model on full training set
  opt_fn_final <- switch(best_params$opt,
                         adam=optimizer_adam(best_params$lr),
                         sgd=optimizer_sgd(best_params$lr))
  final_m <- build_fnn(ncol(X_tr), n_classes, best_params$hu, opt_fn_final)
  final_m %>% fit(X_tr, y_tr_cat, epochs=500, batch_size=32, verbose=0)
  
  # Evaluate
  pred_tr <- apply(final_m %>% predict(X_tr),1,which.max)-1
  pred_te <- apply(final_m %>% predict(X_te),1,which.max)-1
  
  f1_tr <- f1_macro(data.frame(obs=train_df$Sample,
                               pred=factor(levels(train_df$Sample)[pred_tr+1],
                                           levels=levels(train_df$Sample))),
                    levels(train_df$Sample))
  
  f1_te <- f1_macro(data.frame(obs=test_df$Sample,
                               pred=factor(levels(train_df$Sample)[pred_te+1],
                                           levels=levels(train_df$Sample))),
                    levels(train_df$Sample))
  
  # Store
  results <- rbind(results, data.frame(
    Pipeline=pl,
    lr=best_params$lr,
    optimizer=best_params$opt,
    hidden_units=best_params$hu,
    Train_F1=round(f1_tr,4),
    CV_F1=round(best_cv,4),
    Test_F1=round(f1_te,4),
    stringsAsFactors=FALSE
  ))
}

print(results)

# 5. ADDITIONAL VALIDATION ------------------------------------------------------
best_row <- results[which.max(results$CV_F1),]
pp_fun   <- pipelines[[ best_row$Pipeline ]]

train0 <- pp_fun(data_train); test0 <- pp_fun(data_test)
X0_tr  <- as.matrix(train0[,-1]); y0_tr <- as.integer(train0$Sample)-1
X0_te  <- as.matrix(test0[,-1]);  y0_te <- as.integer(test0$Sample)-1

# 5.1 30-seed repeats
set.seed(1234)
seeds  <- sample.int(1e4, 30)
f1_vals30 <- sapply(seeds, function(sd){
  set.seed(sd)
  k_clear_session()
  m <- build_fnn(ncol(X0_tr), nlevels(train0$Sample),
                 best_row$hidden_units,
                 switch(best_row$optimizer,
                        adam=optimizer_adam(best_row$lr),
                        sgd=optimizer_sgd(best_row$lr)))
  m %>% fit(X0_tr, to_categorical(y0_tr), epochs=500, batch_size=32, verbose=0)
  cls <- apply(m %>% predict(X0_te),1,which.max)-1
  f1_macro(data.frame(obs=test0$Sample,
                      pred=factor(levels(train0$Sample)[cls+1],
                                  levels=levels(train0$Sample))),
           levels(train0$Sample))
})
cat(sprintf("30-seed Test F1: mean=%.3f, SD=%.3f\n", mean(f1_vals30), sd(f1_vals30)))

# 5.2 Y-scrambling control
set.seed(2025)
f1_perm <- replicate(30, {
  y_perm <- sample(y0_tr)
  k_clear_session()
  m_p <- build_fnn(ncol(X0_tr), nlevels(train0$Sample),
                   best_row$hidden_units,
                   switch(best_row$optimizer,
                          adam=optimizer_adam(best_row$lr),
                          sgd=optimizer_sgd(best_row$lr)))
  m_p %>% fit(X0_tr, to_categorical(y_perm), epochs=50, batch_size=32, verbose=0)
  cls_p <- apply(m_p %>% predict(X0_te),1,which.max)-1
  f1_macro(data.frame(obs=test0$Sample,
                      pred=factor(levels(train0$Sample)[cls_p+1],
                                  levels=levels(train0$Sample))),
           levels(train0$Sample))
})

# (Plotting code omitted for brevity…)

# 6. ROC & AUC PLOTTING --------------------------------------------------------
# (Same as before, using roc_list, auc_vals, custom colors…)

# 7. UNKNOWN-SAMPLE PREDICTION ------------------------------------------------
data_unknown <- read.xlsx("Unknown.xlsx", sheet="Unknown")
unknown_proc <- pp_fun(data_unknown)
X_un <- as.matrix(unknown_proc[,-1])
probs_un <- final_m %>% predict(X_un)

pred_unknown <- data.frame(Sample=unknown_proc$Sample, probs_un, check.names=FALSE)
colnames(pred_unknown)[-1] <- levels(data_train$Sample)

print(pred_unknown)
write.xlsx(pred_unknown, "Unknown_predictions.xlsx", rowNames=FALSE)
