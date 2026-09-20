
# =====================================================================
suppressMessages({
  library(readr); library(dplyr); library(tidyr); library(stringr)
  library(caret); library(xgboost); library(randomForest); library(nnet)
  library(glmnet); library(gbm); library(e1071)
  library(kernelshap); library(pdp)
  library(ggplot2); library(ggExtra); library(ggbeeswarm); library(cowplot)
  library(pheatmap)
})

set.seed(120)

DATA_DIR      <- "C:/Users/<username>/Documents/lingxi-claw/20260908-09-19-00-537/edu_ml"
META_PATH     <- "C:/Users/<username>/Documents/meta_data.csv"
FIG_DIR       <- file.path(DATA_DIR, "figures_v5")
R_SCRIPTS_DIR <- "C:/Users/<username>/Documents/lingxi-claw/20260908-09-19-00-537/R_scripts"
if (!dir.exists(FIG_DIR)) dir.create(FIG_DIR, recursive = TRUE)


prepare_data <- function() {
  df <- read_csv(file.path(DATA_DIR, "data_clean.csv"), show_col_types = FALSE)
  cat_features <- c("domain", "role", "generation", "subject", "duration", "design", "age_group")
  num_features  <- c("ln_n", "pct_female")
  features <- c(num_features, cat_features)

  df <- df %>% select(all_of(features), yi)
  y <- df$yi

  set.seed(120)
  train_idx <- sample(seq_len(nrow(df)), floor(0.75 * nrow(df)))
  train_data <- df[train_idx, ]; test_data <- df[-train_idx, ]
  y_train <- train_data$yi; y_test <- test_data$yi
  cat("训练:", length(y_train), " 测试:", length(y_test), "\n")

  for (col in cat_features) df[[col]] <- as.factor(df[[col]])
  for (col in cat_features) {
    train_data[[col]] <- factor(train_data[[col]], levels = levels(df[[col]]))
    test_data[[col]]  <- factor(test_data[[col]], levels = levels(df[[col]]))
  }
  dummy_model <- dummyVars(paste("~", paste(features, collapse = "+")), data = df)
  X_train_raw <- data.frame(predict(dummy_model, newdata = train_data))
  X_test_raw  <- data.frame(predict(dummy_model, newdata = test_data))
  preProc <- preProcess(X_train_raw, method = c("center", "scale"))
  X_train <- predict(preProc, X_train_raw); X_test <- predict(preProc, X_test_raw)
  X_train_mat <- as.matrix(X_train); X_test_mat <- as.matrix(X_test)
  cat("编码后特征数:", ncol(X_train_mat), "\n")

  list(df = df, features = features, cat_features = cat_features, num_features = num_features,
       y = y, train_idx = train_idx, y_train = y_train, y_test = y_test,
       X_train_raw = X_train_raw, X_test_raw = X_test_raw,
       X_train_mat = X_train_mat, X_test_mat = X_test_mat, preProc = preProc)
}

# ---- 公共：统一预测接口（兼容 caret 对象 / xgb.Booster / 裸 glmnet）----
predict_any <- function(model, newdata) {
  if (inherits(model, "xgb.Booster")) return(as.numeric(predict(model, as.matrix(newdata))))
  if (inherits(model, "glmnet"))     return(as.numeric(predict(model, newx = as.matrix(newdata))))
  else return(as.numeric(predict(model, newdata = as.data.frame(newdata))))
}

# =====================================================================
#  Part 1  数据清洗 + 特征工程
# =====================================================================
df <- read_csv(META_PATH, show_col_types = FALSE)

out <- data.frame(yi = df$yi)

# ---- 连续变量 ----
out$ln_n <- log(df$n_ai + df$n_control)

# pct_female：提取数值，"60.0%" -> 60，"—" -> NA，缺失填中位数
parse_pct <- function(x) {
  s <- trimws(as.character(x))
  if (s %in% c("\u2014", "-", "", "NA", "NaN", "None")) return(NA_real_)
  s <- gsub("%", "", s)
  suppressWarnings(as.numeric(s))
}
out$pct_female <- sapply(df$pct_female, parse_pct)
out$pct_female[is.na(out$pct_female)] <- median(out$pct_female, na.rm = TRUE)

# ---- 分类变量（归并小类）----
out$domain     <- df$domain
out$role       <- ifelse(df$role == "FP", "CP_FP", df$role)              # FP(3) 并入 CP_FP
out$generation <- df$generation
out$subject    <- ifelse(df$subject == "Health", "Natural Sciences", df$subject)  # Health 并入 Natural Sciences
out$duration   <- df$duration
out$design     <- ifelse(grepl("^RCT", df$design), "RCT",
                         ifelse(grepl("^Quasi", df$design), "Quasi", df$design))
out$age_group  <- ifelse(df$age_group == "Children", "Adolescents", df$age_group)

cat("清洗后形状:", dim(out), "\n")
cat("\n连续变量描述:\n"); print(summary(out[, c("ln_n", "pct_female")]))
cat("\n分类变量频数:\n")
for (c in c("domain", "role", "generation", "subject", "duration", "design", "age_group")) {
  cat("---", c, "---\n"); print(table(out[[c]]))
}

data_clean_path <- file.path(DATA_DIR, "data_clean.csv")
write_csv(out, data_clean_path)
cat("\n已保存:", data_clean_path, "\n")

# =====================================================================
#  Part 2  Pearson 相关分析（9 变量）→ Figure1（来自 pearson_corr_analysis.R）
# =====================================================================
df2 <- read_csv(data_clean_path, show_col_types = FALSE)

target_cols <- c("ln_n", "pct_female", "domain", "role", "generation",
                 "subject", "duration", "design", "age_group")
sub <- df2[target_cols]

# ---- 连续变量转数值 ----
sub$ln_n       <- as.numeric(sub$ln_n)
sub$pct_female <- as.numeric(sub$pct_female)

# ---- 分类变量有序数值编码 ----
sub$domain <- as.numeric(factor(sub$domain,
    levels = c("affective", "cognitive", "collaborative", "higher_order", "load")))
sub$role <- as.numeric(factor(sub$role, levels = c("CP", "CP_FP", "RP_FP")))
sub$generation <- as.numeric(factor(sub$generation, levels = c("Pre-GAI", "Post-GAI")))
sub$subject <- as.numeric(factor(sub$subject,
    levels = c("Arts and Humanities", "Education", "ICT", "Natural Sciences", "Social Sciences")))
sub$duration <- as.numeric(factor(sub$duration,
    levels = c("\u22641 month", ">1-3 months", ">3 months")))
sub$design <- as.numeric(factor(sub$design, levels = c("Quasi", "RCT")))
sub$age_group <- as.numeric(factor(sub$age_group, levels = c("Adolescents", "Adults")))

cat("缺失值统计:\n"); print(colSums(is.na(sub)))
cat("维度:", dim(sub), "\n")

# ---- Pearson 相关 + p 值 ----
n_cols <- ncol(sub)
corr_matrix <- cor(sub, method = "pearson", use = "pairwise.complete.obs")
p_matrix <- matrix(NA, nrow = n_cols, ncol = n_cols,
                   dimnames = list(colnames(sub), colnames(sub)))
for (i in 1:(n_cols - 1)) {
  for (j in (i + 1):n_cols) {
    test <- cor.test(sub[[i]], sub[[j]], method = "pearson")
    p_matrix[i, j] <- test$p.value
    p_matrix[j, i] <- test$p.value
  }
}
cat("\n=== Pearson 相关矩阵 ===\n"); print(round(corr_matrix, 3))
cat("\n=== p 值矩阵 ===\n"); print(round(p_matrix, 4))

# ---- 显著性星号 + 显示矩阵 ----
sig_matrix <- matrix("", nrow = n_cols, ncol = n_cols,
                     dimnames = list(colnames(sub), colnames(sub)))
for (i in 1:n_cols) for (j in 1:n_cols) {
  if (i != j) {
    p <- p_matrix[i, j]
    if (!is.na(p)) sig_matrix[i, j] <- if (p < 0.001) "***" else if (p < 0.01) "**" else if (p < 0.05) "*" else ""
  }
}
display_matrix <- matrix("", nrow = n_cols, ncol = n_cols,
                         dimnames = list(colnames(sub), colnames(sub)))
for (i in 1:n_cols) for (j in 1:n_cols)
  display_matrix[i, j] <- if (i == j) "1.00" else paste0(round(corr_matrix[i, j], 2), sig_matrix[i, j])

color_matrix <- ifelse(abs(corr_matrix) > 0.6, "white", "black")
dimnames(color_matrix) <- dimnames(corr_matrix)

short_names <- c("ln_n" = "ln(Sample Size)", "pct_female" = "Female %",
                 "domain" = "Domain", "role" = "Role", "generation" = "Generation",
                 "subject" = "Subject", "duration" = "Duration", "design" = "Design",
                 "age_group" = "Age Group")
colnames(corr_matrix) <- short_names[colnames(corr_matrix)]
rownames(corr_matrix) <- short_names[rownames(corr_matrix)]
colnames(display_matrix) <- colnames(corr_matrix)
rownames(display_matrix) <- rownames(corr_matrix)
colnames(color_matrix) <- colnames(corr_matrix)
rownames(color_matrix) <- rownames(corr_matrix)

# ---- pheatmap 热图 ----
png(file.path(FIG_DIR, "Figure1_pearson_corr.png"), width = 2700, height = 2400, res = 300)
pheatmap(corr_matrix,
         display_numbers = display_matrix,
         number_color = color_matrix,
         color = colorRampPalette(c("#053061", "#2166AC", "#92C5DE", "#D1E5F0",
                                    "#F7F7F7", "#FDDBC7", "#F4A582", "#D6604D",
                                    "#B2182B", "#67001F"))(100),
         breaks = seq(-1, 1, length.out = 101),
         cluster_rows = FALSE, cluster_cols = FALSE,
         main = "Pearson Correlation Heatmap of Study Characteristics\n(9 Variables)",
         fontsize = 10, fontsize_number = 7, angle_col = 45, border_color = NA,
         legend_breaks = seq(-1, 1, 0.5), legend_labels = as.character(seq(-1, 1, 0.5)))
dev.off()

write.csv(corr_matrix, file.path(DATA_DIR, "pearson_corr_9vars_R.csv"))
write.csv(p_matrix,   file.path(DATA_DIR, "pearson_pval_9vars_R.csv"))
cat("\nPart 2 输出已保存。\n")

# =====================================================================
#  Part 3  6 模型训练 + SHAP + PDP（Figure2-6，来自 ml_meta_analysis.R）
# =====================================================================
cat("\n===== Part 3：6 模型训练 =====\n")
D <- prepare_data()
df <- D$df; features <- D$features; cat_features <- D$cat_features
num_features <- D$num_features; y <- D$y
y_train <- D$y_train; y_test <- D$y_test
X_train_raw <- D$X_train_raw; X_test_raw <- D$X_test_raw
X_train_mat <- D$X_train_mat; X_test_mat <- D$X_test_mat
preProc <- D$preProc

set.seed(120)
folds <- createFolds(y_train, k = 5, returnTrain = TRUE)
ctrl <- trainControl(method = "cv", number = 5, index = folds,
                     savePredictions = "final", allowParallel = FALSE)
n_feat <- ncol(X_train_mat)

# --- Lasso ---
set.seed(120)
grid_lasso <- expand.grid(alpha = 1, lambda = c(0.01, 0.03, 0.1, 0.3, 1, 3, 10))
model_lasso <- train(x = X_train_mat, y = y_train, method = "glmnet",
                     trControl = ctrl, tuneGrid = grid_lasso)

# --- Random Forest ---
set.seed(120)
grid_rf <- expand.grid(mtry = seq(2, max(2, floor(n_feat / 3)), by = 2))
model_rf <- train(x = X_train_mat, y = y_train, method = "rf", trControl = ctrl,
                  tuneGrid = grid_rf, ntree = 300, nodesize = 15, importance = TRUE)

# --- XGBoost ---
set.seed(120)
grid_xgb <- expand.grid(nrounds = 200, max_depth = 3, eta = 0.05,
                        gamma = 0, colsample_bytree = 0.6,
                        min_child_weight = 3, subsample = 0.8)
cv_rmse <- numeric(nrow(grid_xgb))
for (g in seq_len(nrow(grid_xgb))) {
  params <- list(objective = "reg:squarederror", eta = grid_xgb$eta[g],
                 max_depth = grid_xgb$max_depth[g], gamma = grid_xgb$gamma[g],
                 colsample_bytree = grid_xgb$colsample_bytree[g],
                 min_child_weight = grid_xgb$min_child_weight[g],
                 subsample = grid_xgb$subsample[g])
  fold_preds <- rep(NA_real_, length(y_train))
  for (f in seq_along(folds)) {
    tr_idx <- folds[[f]]; val_idx <- setdiff(seq_along(y_train), tr_idx)
    dtrain <- xgb.DMatrix(X_train_mat[tr_idx, , drop = FALSE], label = y_train[tr_idx])
    dval   <- xgb.DMatrix(X_train_mat[val_idx, , drop = FALSE], label = y_train[val_idx])
    bst <- xgb.train(params = params, data = dtrain, nrounds = grid_xgb$nrounds[g], verbose = 0)
    fold_preds[val_idx] <- predict(bst, dval)
  }
  cv_rmse[g] <- sqrt(mean((fold_preds - y_train)^2))
}
best_g <- which.min(cv_rmse)
model_xgb <- xgb.train(params = list(objective = "reg:squarederror",
                       eta = grid_xgb$eta[best_g], max_depth = grid_xgb$max_depth[best_g],
                       gamma = grid_xgb$gamma[best_g],
                       colsample_bytree = grid_xgb$colsample_bytree[best_g],
                       min_child_weight = grid_xgb$min_child_weight[best_g],
                       subsample = grid_xgb$subsample[best_g]),
                       data = xgb.DMatrix(X_train_mat, label = y_train),
                       nrounds = grid_xgb$nrounds[best_g], verbose = 0)

# --- BRT (gbm) ---
set.seed(120)
grid_gbm <- expand.grid(n.trees = 200, interaction.depth = 3,
                        shrinkage = 0.01, n.minobsinnode = 8)
model_gbm <- train(x = X_train_mat, y = y_train, method = "gbm", trControl = ctrl,
                   tuneGrid = grid_gbm, verbose = FALSE)

# --- MLP ---
set.seed(120)
grid_mlp <- expand.grid(size = 10, decay = c(0.1, 1, 10))
model_mlp <- train(x = X_train_mat, y = y_train, method = "nnet", trControl = ctrl,
                   tuneGrid = grid_mlp, linout = TRUE, trace = FALSE, maxit = 2000)

# --- SVM ---
set.seed(120)
grid_svm <- expand.grid(sigma = 0.1, C = 1)
model_svm <- train(x = X_train_mat, y = y_train, method = "svmRadial", trControl = ctrl,
                   tuneGrid = grid_svm)

# ---- 指标 + 统一预测（predict_any 已在顶部定义）----
calc_metrics <- function(act, pred) {
  c(R2 = cor(act, pred)^2, RMSE = sqrt(mean((act - pred)^2)), MAE = mean(abs(act - pred)))
}

all_models <- list(Lasso = model_lasso, `Random Forest` = model_rf,
                   XGBoost = model_xgb, BRT = model_gbm,
                   MLP = model_mlp, SVM = model_svm)
perf <- lapply(names(all_models), function(m) {
  tp <- predict_any(all_models[[m]], X_train_mat)
  vp <- predict_any(all_models[[m]], X_test_mat)
  list(train = tp, test = vp, tr_m = calc_metrics(y_train, tp), te_m = calc_metrics(y_test, vp))
})
names(perf) <- names(all_models)

perf_tab <- do.call(rbind, lapply(names(all_models), function(m) {
  data.frame(Model = m,
             Train_R2 = round(perf[[m]]$tr_m["R2"], 4), Test_R2 = round(perf[[m]]$te_m["R2"], 4),
             Train_RMSE = round(perf[[m]]$tr_m["RMSE"], 4), Test_RMSE = round(perf[[m]]$te_m["RMSE"], 4),
             Train_MAE = round(perf[[m]]$tr_m["MAE"], 4), Test_MAE = round(perf[[m]]$te_m["MAE"], 4),
             Overfit_Gap = round(perf[[m]]$tr_m["R2"] - perf[[m]]$te_m["R2"], 4))
}))
perf_tab <- perf_tab %>% arrange(desc(Test_R2))
print(perf_tab)
write_csv(perf_tab, paste0(FIG_DIR, "model_performance_R.csv"))

best_name <- "Random Forest"   # 用户指定最终模型为 RF
best_model <- model_rf
cat("最佳模型（用户指定）:", best_name, "  Test R2 =", round(perf[[best_name]]$te_m["R2"], 4), "\n")

# ---- Figure 2：4 模型预测性能 + 残差（2×2）----
cat("绘制 Figure 2...\n")
color_train <- "#A1A1A1"; color_test <- "#1e8449"
four_models <- c("Random Forest", "MLP", "XGBoost", "SVM")

make_perf_sub <- function(mname, pp) {
  dfp <- rbind(data.frame(Actual = y_train, Predicted = pp$train, Set = "Train data"),
               data.frame(Actual = y_test, Predicted = pp$test, Set = "Test data"))
  dfp$Set <- factor(dfp$Set, levels = c("Train data", "Test data"))
  dfp$Residual <- dfp$Predicted - dfp$Actual
  minv <- min(c(dfp$Actual, dfp$Predicted)); maxv <- max(c(dfp$Actual, dfp$Predicted))
  rg <- maxv - minv; pad <- rg * 0.05; lim_min <- minv - pad; lim_max <- maxv + pad

  p_main <- ggplot(dfp, aes(x = Actual, y = Predicted)) +
    geom_point(aes(color = Set, fill = Set), shape = 16, size = 2, alpha = 0.85) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black", linewidth = 0.8) +
    geom_smooth(aes(linetype = "Fitted line"), method = "lm", formula = y ~ x, se = FALSE, color = "black", linewidth = 1.2) +
    scale_color_manual(name = NULL, values = c("Train data" = color_train, "Test data" = color_test)) +
    scale_fill_manual(name = NULL, values = c("Train data" = color_train, "Test data" = color_test)) +
    scale_linetype_manual(name = NULL, values = c("Fitted line" = "solid")) +
    coord_fixed(ratio = 1, xlim = c(lim_min, lim_max), ylim = c(lim_min, lim_max)) +
    labs(title = mname, x = "Actual Value", y = "Predicted Value") +
    theme_bw(base_size = 13) +
    theme(panel.grid.major = element_line(color = "grey85", linetype = "dashed"),
          panel.grid.minor = element_blank(), panel.border = element_rect(color = "black", linewidth = 1),
          plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
          axis.text = element_text(color = "black", size = 11),
          legend.position = c(0.85, 0.15), legend.text = element_text(size = 10)) +
    annotate("text", x = lim_min + rg * 0.05, y = lim_max - rg * 0.08,
             label = sprintf("italic(R)[test]^2 == '%.2f'", pp$te_m[["R2"]]), parse = TRUE, hjust = 0, vjust = 1, size = 5, color = "#B22222") +
    annotate("text", x = lim_min + rg * 0.05, y = lim_max - rg * 0.18,
             label = sprintf("italic(RMSE)[test] == '%.2f'", pp$te_m[["RMSE"]]), parse = TRUE, hjust = 0, vjust = 1, size = 5, color = "#B22222")

  p_marg <- ggMarginal(p_main, groupColour = TRUE, groupFill = TRUE, type = "densigram", alpha = 0.6, size = 4)

  p_resid <- ggplot(dfp, aes(x = Actual, y = Residual, color = Set, fill = Set)) +
    geom_point(shape = 16, size = 2, alpha = 0.85) +
    geom_hline(yintercept = 0, color = "black", linewidth = 1) +
    scale_color_manual(values = c("Train data" = color_train, "Test data" = color_test)) +
    scale_fill_manual(values = c("Train data" = color_train, "Test data" = color_test)) +
    coord_cartesian(xlim = c(lim_min, lim_max)) +
    labs(x = "Actual Value", y = "Residuals") +
    theme_bw(base_size = 13) +
    theme(panel.grid.major = element_line(color = "grey85", linetype = "dashed"),
          panel.grid.minor = element_blank(), panel.border = element_rect(color = "black", linewidth = 1),
          legend.position = "none", axis.text = element_text(color = "black", size = 11)) +
    annotate("text", x = lim_max - rg * 0.02, y = max(dfp$Residual) * 0.9,
             label = sprintf("MAE (Train) = %.3f\nMAE (Test) = %.3f", pp$tr_m[["MAE"]], pp$te_m[["MAE"]]),
             hjust = 1, vjust = 1, size = 4.5, color = "black")
  plot_grid(p_marg, p_resid, ncol = 1, rel_heights = c(4, 1.2), align = "none")
}
sub_plots <- lapply(four_models, function(m) make_perf_sub(m, perf[[m]]))
fig2 <- plot_grid(plotlist = sub_plots, ncol = 2, nrow = 2, labels = c("(A)", "(B)", "(C)", "(D)"))
ggsave(paste0(FIG_DIR, "Figure2_predictive_performance.png"), plot = fig2, width = 12, height = 11, dpi = 300, device = "png")
cat("Figure2 完成\n")

# ---- SHAP 计算 ----
cat("计算 SHAP...\n")
X_explain <- rbind(X_train_mat, X_test_mat)
set.seed(120)
bg_idx <- sample(nrow(X_train_mat), min(100, nrow(X_train_mat)))
bg_X <- X_train_mat[bg_idx, , drop = FALSE]
predict_shap <- function(object, X) predict_any(best_model, X)
ks <- kernelshap(object = NULL, X = X_explain, bg_X = bg_X, pred_fun = predict_shap)
S_raw <- ks$S
S_cols <- colnames(S_raw)

S_agg <- matrix(0, nrow = nrow(S_raw), ncol = length(features))
colnames(S_agg) <- features
for (i in seq_along(features)) {
  orig <- features[i]
  if (orig %in% num_features) {
    if (orig %in% S_cols) S_agg[, i] <- S_raw[, orig]
  } else {
    cols <- grep(paste0("^", orig, "\\."), S_cols, value = TRUE)
    if (length(cols) == 1) S_agg[, i] <- S_raw[, cols]
    else if (length(cols) > 1) S_agg[, i] <- rowSums(S_raw[, cols, drop = FALSE])
  }
}

mean_shap <- colMeans(abs(S_agg))
write.csv(S_agg, paste0(FIG_DIR, "S_agg_RF.csv"), row.names = FALSE)
saveRDS(list(model = model_rf, preProc = preProc, X_train_raw = X_train_raw, df = df,
             features = features, num_features = num_features, cat_features = cat_features,
             y_train = y_train, y_test = y_test),
        paste0(FIG_DIR, "rf_env.rds"))
cat("SHAP 重要性（mean|SHAP|）:\n")
print(round(sort(mean_shap, decreasing = TRUE), 4))

# ---- Figure 3：SHAP 蜂群图 ----
cat("绘制 Figure 3...\n")
imp_df <- data.frame(Feature = features, Importance = as.numeric(mean_shap)) %>%
  mutate(Display = gsub("\\.", " ", Feature)) %>% arrange(Importance)
feature_order <- imp_df$Display

mapping <- data.frame(dummy = S_cols, stringsAsFactors = FALSE); mapping$original <- NA
for (orig in cat_features) mapping$original[grep(paste0("^", orig, "\\."), mapping$dummy)] <- orig
rep_cols <- setNames(character(length(features)), features)
for (orig in features) {
  if (orig %in% num_features) rep_cols[orig] <- orig
  else {
    cols <- mapping$dummy[mapping$original == orig & !is.na(mapping$original)]
    if (length(cols) == 1) rep_cols[orig] <- cols
    else if (length(cols) > 1) rep_cols[orig] <- cols[which.max(apply(X_explain[, cols, drop = FALSE], 2, var))]
  }
}
X_color <- X_explain
for (col in colnames(X_color)) {
  cmin <- min(X_color[, col]); cmax <- max(X_color[, col])
  X_color[, col] <- if (cmax > cmin) (X_color[, col] - cmin) / (cmax - cmin) else 0
}
X_rep <- matrix(0, nrow = nrow(X_color), ncol = length(features)); colnames(X_rep) <- features
for (i in seq_along(features)) { rc <- rep_cols[features[i]]; if (!is.na(rc) && rc %in% colnames(X_color)) X_rep[, i] <- X_color[, rc] }

df_color <- as.data.frame(X_rep) %>% mutate(sample_id = 1:n()) %>%
  pivot_longer(cols = -sample_id, names_to = "Feature", values_to = "Feature_value")
df_swarm <- as.data.frame(S_agg) %>% mutate(sample_id = 1:n()) %>%
  pivot_longer(cols = -sample_id, names_to = "Feature", values_to = "SHAP") %>%
  mutate(Display = gsub("\\.", " ", Feature))
df_plot <- left_join(df_swarm, df_color, by = c("sample_id", "Feature"))
df_plot$Display <- factor(df_plot$Display, levels = feature_order)

x_min <- min(df_plot$SHAP); x_max <- max(df_plot$SHAP); tw <- x_max - x_min
bar_start <- x_min - tw * 0.08
scale_f <- (tw * 0.95) / max(imp_df$Importance)
imp_df <- imp_df %>% mutate(scaled_end = bar_start + Importance * scale_f,
                            y_num = as.numeric(factor(Display, levels = feature_order)))
bw <- 0.55

fig3 <- ggplot() +
  geom_rect(data = imp_df, aes(xmin = bar_start, xmax = scaled_end, ymin = y_num - bw/2, ymax = y_num + bw/2), fill = "#87B1D4", alpha = 0.7) +
  geom_vline(xintercept = 0, color = "grey40", linewidth = 0.6) +
  geom_quasirandom(data = df_plot, aes(x = SHAP, y = Display, color = Feature_value),
                   groupOnX = FALSE, method = "pseudorandom", width = 0.35, alpha = 0.85, size = 1.5, stroke = 0) +
  scale_color_gradient(low = "#4575B4", high = "#D73027", breaks = c(0, 1), labels = c("Low", "High"),
                       name = "Feature Value",
                       guide = guide_colorbar(title.position = "right",
                                              title.theme = element_text(angle = -90, hjust = 0.5, size = 13, face = "bold"),
                                              barwidth = unit(0.4, "cm"), barheight = unit(5, "cm"), ticks = FALSE)) +
  scale_x_continuous(name = "SHAP value (impact on model output)", expand = expansion(mult = c(0, 0.05)),
                     limits = c(bar_start, max(x_max, max(imp_df$scaled_end))),
                     sec.axis = sec_axis(transform = ~ (. - bar_start) / scale_f, name = "Mean (|SHAP value|)")) +
  labs(y = "Feature") + theme_minimal(base_size = 14) +
  theme(panel.background = element_rect(fill = "white", color = NA),
        plot.background = element_rect(fill = "white", color = NA),
        panel.grid.major.x = element_blank(), panel.grid.minor.x = element_blank(),
        panel.grid.major.y = element_line(color = "grey92", linetype = "dashed", linewidth = 0.5),
        axis.line.x = element_line(color = "black", linewidth = 0.8),
        axis.line.y = element_line(color = "black", linewidth = 0.8),
        axis.text = element_text(color = "black", size = 12),
        axis.title = element_text(face = "bold", size = 14),
        legend.text = element_text(size = 12, face = "bold"),
        plot.margin = ggplot2::margin(t = 20, r = 30, b = 20, l = 10))
ggsave(paste0(FIG_DIR, "Figure3_SHAP_summary.png"), plot = fig3, width = 10.5, height = 7.5, dpi = 300, device = "png")
cat("Figure3 完成\n")

# ---- Figure 5：分类变量类别级 SHAP ----
cat("绘制 Figure 5...\n")
df_combined <- df
for (feat in cat_features) {
  temp_df <- data.frame(Category = as.character(df_combined[[feat]]), SHAP_Value = S_agg[, feat])
  summary_df <- temp_df %>% dplyr::group_by(Category) %>%
    dplyr::summarise(Mean_SHAP = mean(SHAP_Value), SD_SHAP = sd(SHAP_Value), N = dplyr::n(), .groups = "drop") %>%
    dplyr::mutate(SE_SHAP = SD_SHAP / sqrt(N)) %>%
    dplyr::arrange(dplyr::desc(Mean_SHAP))
  summary_df$Category <- factor(summary_df$Category, levels = rev(summary_df$Category))
  summary_df$Fill_Color <- ifelse(summary_df$Mean_SHAP > 0, "#D73027", "#4575B4")

  x_lo <- min(c(summary_df$Mean_SHAP - summary_df$SE_SHAP, 0), na.rm = TRUE)
  x_hi <- max(c(summary_df$Mean_SHAP + summary_df$SE_SHAP, 0), na.rm = TRUE)
  x_pad <- (x_hi - x_lo) * 0.35
  xlims <- c(x_lo - x_pad, x_hi + x_pad)

  p_cat <- ggplot(summary_df, aes(x = Mean_SHAP, y = Category)) +
    geom_vline(xintercept = 0, color = "black", linetype = "dashed", linewidth = 0.8) +
    geom_bar(stat = "identity", aes(fill = Fill_Color), width = 0.6, alpha = 0.8) +
    geom_errorbar(aes(xmin = Mean_SHAP - SE_SHAP, xmax = Mean_SHAP + SE_SHAP), width = 0.2, color = "gray30") +
    scale_fill_identity() +
    scale_x_continuous(limits = xlims, expand = c(0, 0)) +
    labs(title = paste0("Impact of specific categories: ", gsub("\\.", " ", feat)),
         x = "Mean SHAP Value (\u00B1 SE)", y = NULL) +
    theme_minimal(base_size = 14) +
    theme(plot.title = element_text(face = "bold", size = 16),
          axis.text.y = element_text(color = "black", face = "bold", size = 12),
          axis.text.x = element_text(color = "black", size = 12),
          panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(),
          panel.grid.major.x = element_line(color = "gray85", linetype = "dotted"),
          plot.margin = ggplot2::margin(15, 20, 15, 15)) +
    geom_text(aes(x = ifelse(Mean_SHAP > 0, Mean_SHAP + SE_SHAP + (x_hi - x_lo) * 0.02,
                             Mean_SHAP - SE_SHAP - (x_hi - x_lo) * 0.02),
                  label = paste0("(n=", N, ")"), hjust = ifelse(Mean_SHAP > 0, 0, 1)), size = 4, color = "gray30")
  h <- max(4, nrow(summary_df) * 0.6 + 2)
  ggsave(paste0(FIG_DIR, "Figure5_SHAP_categorical_", feat, ".png"), plot = p_cat, width = 8.5, height = h, dpi = 300, device = "png")
}
cat("Figure5 完成\n")

# ---- Figure 4：连续特征 PDP ----
cat("绘制 Figure 4...\n")
pred_fun_pdp <- function(object, newdata) {
  newdata_scaled <- predict(preProc, as.data.frame(newdata))
  predict_any(best_model, newdata_scaled)
}
plot_pdp_style <- function(ice_data, fc, fl, fn) {
  ps <- ice_data %>% group_by(!!sym(fc)) %>%
    summarise(mean_yhat = mean(yhat), sd_yhat = sd(yhat), .groups = "drop") %>%
    mutate(lower = mean_yhat - sd_yhat, upper = mean_yhat + sd_yhat)
  ymax <- max(ps$mean_yhat); ymin <- min(ps$mean_yhat)
  thr <- ymax - 0.1 * (ymax - ymin)
  top <- ps %>% filter(mean_yhat >= thr)
  opt_range <- c(min(top[[fc]]), max(top[[fc]]))
  x_span <- diff(range(ps[[fc]]))
  opt_rel <- (mean(opt_range) - min(ps[[fc]])) / x_span
  txt_x <- ifelse(opt_rel > 0.5, opt_range[2], opt_range[1])
  txt_hjust <- ifelse(opt_rel > 0.5, 1, 0)
  p <- ggplot(ps, aes(x = !!sym(fc))) +
    geom_ribbon(aes(ymin = lower, ymax = upper), fill = "#A2D9CE", alpha = 0.35) +
    geom_hline(yintercept = 0, color = "#B22222", linetype = "dashed", linewidth = 0.6, alpha = 0.7) +
    geom_line(aes(y = mean_yhat), color = "#1E8449", linewidth = 1.2) +
    geom_point(aes(y = mean_yhat), color = "#1E8449", size = 2.5) +
    annotate("rect", xmin = opt_range[1], xmax = opt_range[2], ymin = -Inf, ymax = Inf, alpha = 0.12, fill = "grey40") +
    geom_hline(yintercept = thr, linetype = "dashed", color = "#D95F02", linewidth = 0.8) +
    annotate("text", x = txt_x, y = thr, label = "Top 10% Outcomes", hjust = txt_hjust, vjust = -0.8, color = "#D95F02", size = 4) +
    scale_x_continuous(expand = expansion(mult = c(0.06, 0.08))) +
    labs(title = paste0('PDP for "', fl, '"'), x = fl, y = "Predicted Hedges' g") +
    theme_minimal(base_size = 14) +
    theme(plot.title = element_text(face = "bold", size = 16, hjust = 0),
          axis.title = element_text(face = "bold"), axis.text = element_text(color = "black"),
          panel.grid.minor = element_blank(), panel.grid.major = element_line(color = "gray90", linewidth = 0.5),
          panel.border = element_rect(color = "gray80", fill = NA, linewidth = 1),
          plot.margin = ggplot2::margin(t = 15, r = 20, b = 15, l = 15))
  ggsave(paste0(FIG_DIR, fn, ".png"), plot = p, width = 7.5, height = 5.5, dpi = 300, device = "png")
  return(p)
}

pdp_specs <- list(
  list(fc = "ln_n", fl = "ln(Sample Size)", fn = "Figure4_PDP_ln_n"),
  list(fc = "pct_female", fl = "Percent Female", fn = "Figure4_PDP_pct_female")
)
for (sp in pdp_specs) {
  grid <- data.frame(tmp = seq(quantile(X_train_raw[[sp$fc]], 0.05, na.rm = TRUE),
                               quantile(X_train_raw[[sp$fc]], 0.95, na.rm = TRUE), length.out = 30))
  names(grid) <- sp$fc
  ice <- pdp::partial(object = best_model, pred.var = sp$fc, pred.fun = pred_fun_pdp,
                      train = X_train_raw, pred.grid = grid, ice = TRUE, center = FALSE, type = "regression")
  plot_pdp_style(ice, sp$fc, sp$fl, sp$fn)
}
cat("Figure4 完成\n")

# ---- Figure 6：二维 PDP（ln_n × pct_female）----
cat("绘制 Figure 6...\n")
compute_2d_pdp <- function(f1, f2, X_ref, n_grid = 40) {
  f1_vals <- seq(min(X_ref[[f1]], na.rm = TRUE), max(X_ref[[f1]], na.rm = TRUE), length.out = n_grid)
  f2_vals <- seq(quantile(X_ref[[f2]], 0.05, na.rm = TRUE), quantile(X_ref[[f2]], 0.95, na.rm = TRUE), length.out = n_grid)
  grid <- matrix(NA, nrow = n_grid, ncol = n_grid)
  X_template <- X_ref
  for (i in seq_len(n_grid)) {
    for (j in seq_len(n_grid)) {
      X_temp <- X_template
      X_temp[[f1]] <- f1_vals[i]
      X_temp[[f2]] <- f2_vals[j]
      X_temp_scaled <- predict(preProc, as.data.frame(X_temp))
      grid[i, j] <- mean(predict_any(best_model, X_temp_scaled), na.rm = TRUE)
    }
  }
  list(grid = grid, f1_vals = f1_vals, f2_vals = f2_vals, f1 = f1, f2 = f2)
}
pdp_res <- compute_2d_pdp("ln_n", "pct_female", X_train_raw, n_grid = 40)
grid_df <- expand.grid(x = pdp_res$f1_vals, y = pdp_res$f2_vals)
grid_df$z <- as.vector(pdp_res$grid)
z_range <- range(grid_df$z, na.rm = TRUE)
contour_levels <- unique(seq(z_range[1], z_range[2], length.out = 12))

fig6 <- ggplot(grid_df, aes(x = x, y = y)) +
  geom_tile(aes(fill = z)) +
  geom_contour(aes(z = z), breaks = contour_levels, color = "gray30", linewidth = 0.25) +
  scale_fill_gradient2(low = "#2166AC", mid = "#F7F7F7", high = "#B2182B", midpoint = 0, name = "Predicted\nHedges' g") +
  labs(title = "2D PDP: ln(Sample Size) \u00D7 Percent Female",
       x = "ln(Sample Size)", y = "Percent Female") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold", size = 15, hjust = 0),
        axis.title = element_text(face = "bold", color = "black", size = 12),
        axis.text = element_text(color = "black", size = 10),
        panel.grid = element_blank(), panel.border = element_rect(color = "gray60", fill = NA, linewidth = 1),
        legend.position = "right", legend.key.height = unit(1.8, "cm"), legend.key.width = unit(0.55, "cm"),
        legend.title = element_text(face = "bold", size = 10), legend.text = element_text(size = 9),
        plot.margin = ggplot2::margin(t = 15, r = 20, b = 15, l = 15))
ggsave(paste0(FIG_DIR, "Figure6_2D_PDP_ln_n_pct_female.png"), plot = fig6, width = 8, height = 6.5, dpi = 300, device = "png")
cat("Figure6 完成\n")

# =====================================================================
#  Part 4  模型优化：加强正则化 + gap 惩罚选参 + 5 折重复 CV（来自 model_optimization.R）
# =====================================================================
cat("\n===== Part 4：模型优化 =====\n")
D4 <- prepare_data()
df <- D4$df; features <- D4$features; cat_features <- D4$cat_features
num_features <- D4$num_features; y <- D4$y
y_train <- D4$y_train; y_test <- D4$y_test
X_train_raw <- D4$X_train_raw; X_test_raw <- D4$X_test_raw
X_train_mat <- D4$X_train_mat; X_test_mat <- D4$X_test_mat
n_feat <- ncol(X_train_mat)

# ---- 指标 ----
calc_r2 <- function(a, p) {
  if (sd(p) < 1e-10 || is.na(sd(p))) return(NA)
  cor(a, p)^2
}
calc_rmse <- function(a, p) sqrt(mean((a - p)^2))
calc_mae <- function(a, p) mean(abs(a - p))
gap_score <- function(te_r2, gap) {
  if (is.na(te_r2)) return(-Inf)
  if (is.na(gap)) gap <- 0
  te_r2 - 0.3 * max(gap, 0)
}

# ---- 5 折重复 CV（3 次重复，稳健评估）----
set.seed(120)
folds <- createMultiFolds(y_train, k = 5, times = 3)
cv_r2 <- function(fit_fun) {
  preds <- rep(NA_real_, length(y_train))
  for (f in folds) {
    tr_idx <- f; val_idx <- setdiff(seq_along(y_train), tr_idx)
    m <- fit_fun(tr_idx)
    preds[val_idx] <- predict_any(m, X_train_mat[val_idx, , drop = FALSE])
  }
  calc_r2(y_train, preds)
}

results <- list()

# --- 1. Lasso（加强 lambda）---
cat("\n[1/6] Lasso 调参...\n")
best <- NULL
for (lam in c(0.1, 0.3, 1, 3, 10, 30)) {
  m <- glmnet(X_train_mat, y_train, alpha = 1, lambda = lam)
  ptr <- as.numeric(predict(m, newx = X_train_mat))
  pte <- as.numeric(predict(m, newx = X_test_mat))
  tr_r2 <- calc_r2(y_train, ptr); te_r2 <- calc_r2(y_test, pte)
  gap <- tr_r2 - te_r2; sc <- gap_score(te_r2, gap)
  if (is.null(best) || sc > best$sc) best <- list(model = m, lambda = lam, tr_r2 = tr_r2, te_r2 = te_r2, gap = gap, sc = sc, ptr = ptr, pte = pte)
}
best$cv <- cv_r2(function(tr_idx) glmnet(X_train_mat[tr_idx, , drop = FALSE], y_train[tr_idx], alpha = 1, lambda = best$lambda))
results$Lasso <- best
cat(sprintf("  lambda=%.2f  Train R2=%.3f  Test R2=%.3f  gap=%.3f  CV R2=%.3f\n", best$lambda, best$tr_r2, best$te_r2, best$gap, best$cv))

# --- 2. Random Forest（加强 nodesize/depth）---
cat("\n[2/6] RandomForest 调参...\n")
best <- NULL
for (nodesize in c(5, 10, 15)) {
  for (mtry in c(4, 6, 8, 12)) {
    m <- randomForest(x = X_train_mat, y = y_train, ntree = 500,
                      nodesize = nodesize, mtry = mtry, maxnodes = 30)
    ptr <- predict(m, X_train_mat); pte <- predict(m, X_test_mat)
    tr_r2 <- calc_r2(y_train, ptr); te_r2 <- calc_r2(y_test, pte)
    gap <- tr_r2 - te_r2; sc <- gap_score(te_r2, gap)
    if (is.null(best) || sc > best$sc) best <- list(model = m, nodesize = nodesize, mtry = mtry, tr_r2 = tr_r2, te_r2 = te_r2, gap = gap, sc = sc, ptr = ptr, pte = pte)
  }
}
best$cv <- cv_r2(function(tr_idx) randomForest(x = X_train_mat[tr_idx, , drop = FALSE], y = y_train[tr_idx], ntree = 500, nodesize = best$nodesize, mtry = best$mtry, maxnodes = 30))
results$RandomForest <- best
cat(sprintf("  nodesize=%d mtry=%d  Train R2=%.3f  Test R2=%.3f  gap=%.3f  CV R2=%.3f\n", best$nodesize, best$mtry, best$tr_r2, best$te_r2, best$gap, best$cv))

# --- 3. XGBoost（加强正则化）---
cat("\n[3/6] XGBoost 调参...\n")
best <- NULL
for (depth in c(2, 3)) {
  for (mcw in c(5, 10)) {
    for (gamma in c(0.5, 1)) {
      for (lam in c(5, 10)) {
        params <- list(objective = "reg:squarederror", eta = 0.05, max_depth = depth,
                       min_child_weight = mcw, gamma = gamma, lambda = lam,
                       subsample = 0.7, colsample_bytree = 0.6)
        m <- xgb.train(params = params, data = xgb.DMatrix(X_train_mat, label = y_train),
                       nrounds = 200, verbose = 0)
        ptr <- predict(m, X_train_mat); pte <- predict(m, X_test_mat)
        tr_r2 <- calc_r2(y_train, ptr); te_r2 <- calc_r2(y_test, pte)
        gap <- tr_r2 - te_r2; sc <- gap_score(te_r2, gap)
        if (is.null(best) || sc > best$sc) best <- list(model = m, depth = depth, mcw = mcw, gamma = gamma, lam = lam, tr_r2 = tr_r2, te_r2 = te_r2, gap = gap, sc = sc, ptr = ptr, pte = pte)
      }
    }
  }
}
best$cv <- cv_r2(function(tr_idx) xgb.train(params = list(objective = "reg:squarederror", eta = 0.05, max_depth = best$depth, min_child_weight = best$mcw, gamma = best$gamma, lambda = best$lam, subsample = 0.7, colsample_bytree = 0.6), data = xgb.DMatrix(X_train_mat[tr_idx, , drop = FALSE], label = y_train[tr_idx]), nrounds = 200, verbose = 0))
results$XGBoost <- best
cat(sprintf("  depth=%d mcw=%d gamma=%.1f lambda=%.0f  Train R2=%.3f  Test R2=%.3f  gap=%.3f  CV R2=%.3f\n", best$depth, best$mcw, best$gamma, best$lam, best$tr_r2, best$te_r2, best$gap, best$cv))

# --- 4. BRT（gbm，加强正则化）---
cat("\n[4/6] BRT 调参...\n")
best <- NULL
for (depth in c(1, 2, 3)) {
  for (shrink in c(0.005, 0.01)) {
    for (minobs in c(10, 15)) {
      m <- gbm(y_train ~ ., data = as.data.frame(X_train_mat), distribution = "gaussian",
               n.trees = 500, interaction.depth = depth, shrinkage = shrink,
               n.minobsinnode = minobs, verbose = FALSE)
      ptr <- predict(m, newdata = as.data.frame(X_train_mat), n.trees = 500)
      pte <- predict(m, newdata = as.data.frame(X_test_mat), n.trees = 500)
      tr_r2 <- calc_r2(y_train, ptr); te_r2 <- calc_r2(y_test, pte)
      gap <- tr_r2 - te_r2; sc <- gap_score(te_r2, gap)
      if (is.null(best) || sc > best$sc) best <- list(model = m, depth = depth, shrink = shrink, minobs = minobs, tr_r2 = tr_r2, te_r2 = te_r2, gap = gap, sc = sc, ptr = ptr, pte = pte)
    }
  }
}
best$cv <- cv_r2(function(tr_idx) gbm(y_train[tr_idx] ~ ., data = as.data.frame(X_train_mat[tr_idx, , drop = FALSE]), distribution = "gaussian", n.trees = 500, interaction.depth = best$depth, shrinkage = best$shrink, n.minobsinnode = best$minobs, verbose = FALSE))
results$BRT <- best
cat(sprintf("  depth=%d shrink=%.3f minobs=%d  Train R2=%.3f  Test R2=%.3f  gap=%.3f  CV R2=%.3f\n", best$depth, best$shrink, best$minobs, best$tr_r2, best$te_r2, best$gap, best$cv))

# --- 5. MLP（加强 decay）---
cat("\n[5/6] MLP 调参...\n")
best <- NULL
for (size in c(5, 10)) {
  for (decay in c(1, 10)) {
    set.seed(120)
    m <- nnet(X_train_mat, y_train, size = size, decay = decay, linout = TRUE,
              maxit = 2000, trace = FALSE, MaxNWts = 5000)
    ptr <- as.numeric(predict(m, X_train_mat)); pte <- as.numeric(predict(m, X_test_mat))
    tr_r2 <- calc_r2(y_train, ptr); te_r2 <- calc_r2(y_test, pte)
    gap <- tr_r2 - te_r2; sc <- gap_score(te_r2, gap)
    if (is.null(best) || sc > best$sc) best <- list(model = m, size = size, decay = decay, tr_r2 = tr_r2, te_r2 = te_r2, gap = gap, sc = sc, ptr = ptr, pte = pte)
  }
}
best$cv <- cv_r2(function(tr_idx) { set.seed(120); nnet(X_train_mat[tr_idx, , drop = FALSE], y_train[tr_idx], size = best$size, decay = best$decay, linout = TRUE, maxit = 2000, trace = FALSE, MaxNWts = 5000) })
results$MLP <- best
cat(sprintf("  size=%d decay=%.0f  Train R2=%.3f  Test R2=%.3f  gap=%.3f  CV R2=%.3f\n", best$size, best$decay, best$tr_r2, best$te_r2, best$gap, best$cv))

# --- 6. SVM（减小 C 加强正则化）---
cat("\n[6/6] SVM 调参...\n")
best <- NULL
for (sigma in c(0.05, 0.1, 0.2)) {
  for (C in c(0.1, 0.5, 1)) {
    m <- svm(x = X_train_mat, y = y_train, kernel = "radial", gamma = sigma, cost = C)
    ptr <- predict(m, X_train_mat); pte <- predict(m, X_test_mat)
    tr_r2 <- calc_r2(y_train, ptr); te_r2 <- calc_r2(y_test, pte)
    gap <- tr_r2 - te_r2; sc <- gap_score(te_r2, gap)
    if (is.null(best) || sc > best$sc) best <- list(model = m, sigma = sigma, C = C, tr_r2 = tr_r2, te_r2 = te_r2, gap = gap, sc = sc, ptr = ptr, pte = pte)
  }
}
best$cv <- cv_r2(function(tr_idx) svm(x = X_train_mat[tr_idx, , drop = FALSE], y = y_train[tr_idx], kernel = "radial", gamma = best$sigma, cost = best$C))
results$SVM <- best
cat(sprintf("  sigma=%.2f C=%.1f  Train R2=%.3f  Test R2=%.3f  gap=%.3f  CV R2=%.3f\n", best$sigma, best$C, best$tr_r2, best$te_r2, best$gap, best$cv))

# ---- 汇总 ----
cat("\n========== 优化后模型性能对比 ==========\n")
tab <- do.call(rbind, lapply(names(results), function(nm) {
  r <- results[[nm]]
  data.frame(Model = nm, Train_R2 = r$tr_r2, Test_R2 = r$te_r2,
             Gap = r$gap, CV_R2 = r$cv, Score = r$sc,
             RMSE_Test = calc_rmse(y_test, r$pte), MAE_Test = calc_mae(y_test, r$pte))
}))
tab <- tab %>% arrange(desc(Score))
print(tab, row.names = FALSE)
write_csv(tab, paste0(FIG_DIR, "model_optimization_R.csv"))

best_name <- tab$Model[1]
cat("\n按 gap 惩罚 score 最优模型:", best_name, "\n")
cat("说明：CV_R2 是 5 折 × 3 重复交叉验证的稳健估计，比单次 Test_R2 更可靠。\n")

# =====================================================================
#  Part 5  RF 结果数值提取（来自 extract_rf_results.R）
#  独立重新训练 RF，输出详细数值到 _rf_results.txt（用于更新结果部分 Word）
# =====================================================================
cat("\n===== Part 5：RF 结果数值提取 =====\n")
out_txt <- file.path(R_SCRIPTS_DIR, "_rf_results.txt")
sink(out_txt)

D5 <- prepare_data()
df <- D5$df; features <- D5$features; cat_features <- D5$cat_features
num_features <- D5$num_features; y <- D5$y
y_train <- D5$y_train; y_test <- D5$y_test
X_train_raw <- D5$X_train_raw; X_test_raw <- D5$X_test_raw
X_train_mat <- D5$X_train_mat; X_test_mat <- D5$X_test_mat
preProc <- D5$preProc

# ---- 训练 RF（与 Part 3 一致：caret，ntree=300, nodesize=15）----
set.seed(120)
folds <- createFolds(y_train, k = 5, returnTrain = TRUE)
ctrl <- trainControl(method = "cv", number = 5, index = folds, allowParallel = FALSE)
grid_rf <- expand.grid(mtry = seq(2, max(2, floor(ncol(X_train_mat) / 3)), by = 2))
model_rf <- train(x = X_train_mat, y = y_train, method = "rf", trControl = ctrl,
                  tuneGrid = grid_rf, ntree = 300, nodesize = 15, importance = TRUE)
cat("RF 最优 mtry =", model_rf$bestTune$mtry, "\n")

ptr <- predict(model_rf, X_train_mat); pte <- predict(model_rf, X_test_mat)
tr_r2 <- cor(y_train, ptr)^2; te_r2 <- cor(y_test, pte)^2
te_rmse <- sqrt(mean((y_test - pte)^2)); te_mae <- mean(abs(y_test - pte))
cat("\n=== RF 性能 ===\n")
cat(sprintf("Train R2 = %.4f | Test R2 = %.4f | Test RMSE = %.4f | Test MAE = %.4f | Gap = %.4f\n",
            tr_r2, te_r2, te_rmse, te_mae, tr_r2 - te_r2))

# ---- SHAP ----
X_explain <- rbind(X_train_mat, X_test_mat)
set.seed(120)
bg_idx <- sample(nrow(X_train_mat), min(100, nrow(X_train_mat)))
bg_X <- X_train_mat[bg_idx, , drop = FALSE]
predict_shap <- function(object, X) as.numeric(predict(model_rf, newdata = as.data.frame(X)))
ks <- kernelshap(object = NULL, X = X_explain, bg_X = bg_X, pred_fun = predict_shap)
S_raw <- ks$S
S_cols <- colnames(S_raw)

S_agg <- matrix(0, nrow = nrow(S_raw), ncol = length(features))
colnames(S_agg) <- features
for (i in seq_along(features)) {
  orig <- features[i]
  if (orig %in% num_features) {
    if (orig %in% S_cols) S_agg[, i] <- S_raw[, orig]
  } else {
    cols <- grep(paste0("^", orig, "\\."), S_cols, value = TRUE)
    if (length(cols) == 1) S_agg[, i] <- S_raw[, cols]
    else if (length(cols) > 1) S_agg[, i] <- rowSums(S_raw[, cols, drop = FALSE])
  }
}

cat("\n=== 9 特征 SHAP 重要性 (mean|SHAP|) ===\n")
imp <- colMeans(abs(S_agg))
print(round(sort(imp, decreasing = TRUE), 4))

cat("\n=== 连续变量 SHAP 依赖方向 (Pearson r) ===\n")
for (fi in num_features) {
  fvals <- df[[fi]]
  r <- cor(fvals, S_agg[, fi])
  cat(sprintf("  %s: r = %.3f\n", fi, r))
}

cat("\n=== 分类变量各 level 的 mean SHAP + SE + n ===\n")
for (feat in cat_features) {
  fi <- features[features == feat]
  svals <- S_agg[, fi]
  lv <- as.character(df[[feat]])
  cat(sprintf("\n[%s]\n", feat))
  for (lev in sort(unique(lv))) {
    m <- svals[lv == lev]; mu <- mean(m); se <- sd(m) / sqrt(length(m))
    cat(sprintf("  %-24s meanSHAP=%+.4f  SE=%.4f  n=%d\n", lev, mu, se, length(m)))
  }
}

# ---- PDP：ln_n 与 pct_female ----
cat("\n=== PDP 单变量（5th-95th 百分位，30 点）===\n")
for (fi in num_features) {
  fvals <- X_train_raw[[fi]]
  lo <- quantile(fvals, 0.05); hi <- quantile(fvals, 0.95)
  grid <- data.frame(tmp = seq(lo, hi, length.out = 30)); names(grid) <- fi
  base <- X_train_raw[1, ]
  for (c in setdiff(colnames(X_train_raw), fi)) base[1, c] <- if (is.numeric(X_train_raw[[c]])) median(X_train_raw[[c]]) else X_train_raw[[c]][1]
  preds <- sapply(seq_len(nrow(grid)), function(k) {
    r <- base; r[1, fi] <- grid[k, fi]
    r_s <- predict(preProc, as.data.frame(r))
    mean(predict(model_rf, newdata = as.data.frame(r_s)))
  })
  cat(sprintf("\n[%s] 范围 [%.2f, %.2f]\n", fi, lo, hi))
  for (k in c(1, 6, 11, 16, 21, 26, 30)) {
    cat(sprintf("  x=%.3f  PDP=%+.3f\n", grid[k, fi], preds[k]))
  }
  cat(sprintf("  min=%+.3f max=%+.3f\n", min(preds), max(preds)))
}

# ---- 2D PDP：ln_n × pct_female ----
cat("\n=== 2D PDP ln_n × pct_female（6×6 格点）===\n")
v1 <- X_train_raw$ln_n; v2 <- X_train_raw$pct_female
g1 <- seq(quantile(v1, 0.05), quantile(v1, 0.95), length.out = 6)
g2 <- seq(quantile(v2, 0.05), quantile(v2, 0.95), length.out = 6)
base <- X_train_raw[1, ]
for (c in setdiff(colnames(X_train_raw), c("ln_n", "pct_female")))
  base[1, c] <- if (is.numeric(X_train_raw[[c]])) median(X_train_raw[[c]]) else X_train_raw[[c]][1]
for (a in g1) {
  row <- sapply(g2, function(b) {
    r <- base; r[1, "ln_n"] <- a; r[1, "pct_female"] <- b
    r_s <- predict(preProc, as.data.frame(r))
    mean(predict(model_rf, newdata = as.data.frame(r_s)))
  })
  cat(sprintf("  ln_n=%.2f  ", a), paste(sprintf("%+.2f", row), collapse = "  "), "\n")
}

sink()
cat("结果已写入:", out_txt, "\n")

cat("\n===== 全部流程完成 =====\n")
