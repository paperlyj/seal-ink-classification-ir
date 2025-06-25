# PCA Analysis of IR Spectra
# --------------------------------
# Load required packages
library(openxlsx)
library(prospectr)
library(ggplot2)

# Function to perform L2 normalization
normalize_l2 <- function(x) {
  x / sqrt(sum(x^2))
}

#--------------------------------
# 1. Raw Data PCA
#--------------------------------
# Read summary data
raw_file <- "Summary.xlsx"
data_raw <- read.xlsx(xlsxFile = raw_file, sheet = "Summary")

# Prepare data
data_raw$Sample <- as.factor(data_raw$Sample)
samples_raw <- data_raw$Sample

# L2 normalization
norm_raw_matrix <- t(apply(data_raw[ , -1], 1, normalize_l2))
data_raw_norm <- data.frame(Sample = samples_raw, norm_raw_matrix)

# PCA
pr_raw <- prcomp(data_raw_norm[ , -1], rank. = 7, scale. = FALSE)
print(summary(pr_raw))

# Plot PCA
plot_pca <- function(pr, data, pc1_perc, pc2_perc) {
  ggplot(data, aes(x = pr$x[,1], y = pr$x[,2], color = Sample, shape = Sample)) +
    geom_point(shape = 1, size = 6, stroke = 1.25) +
    scale_shape_manual(values = rep(1, length(levels(data$Sample)))) +
    scale_color_manual(values = c(
      "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
      "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
      "#bcbd22", "#17becf"
    )) +
    theme_light() +
    theme(
      panel.border    = element_rect(colour = "black", fill = NA, size = 1),
      axis.title      = element_text(size = 18, face = "bold"),
      axis.text       = element_text(size = 16, colour = "black"),
      legend.title    = element_blank(),
      legend.position = "none"
    ) +
    xlab(expression(bold(paste("PC1 (", pc1_perc, "%)")))) +
    ylab(expression(bold(paste("PC2 (", pc2_perc, "%)"))))
}

# Display Raw PCA plot
print(plot_pca(pr_raw, data_raw_norm,
               round(100 * pr_raw$sdev[1]^2 / sum(pr_raw$sdev^2), 1),
               round(100 * pr_raw$sdev[2]^2 / sum(pr_raw$sdev^2), 1)))

#--------------------------------
# 2. Second-Derivative (Savitzky–Golay) Data PCA
#--------------------------------
# Read selected data
data_sg <- read.xlsx(xlsxFile = raw_file, sheet = "Selected")

data_sg$Sample <- as.factor(data_sg$Sample)
samples_sg <- data_sg$Sample

# Savitzky–Golay smoothing (2nd derivative)
sg_matrix <- t(apply(data_sg[ , -1], 1, function(x) savitzkyGolay(x, m = 2, p = 3, w = 13)))

# L2 normalization on SG data
norm_sg_matrix <- t(apply(sg_matrix, 1, normalize_l2))
data_sg_norm <- data.frame(Sample = samples_sg, norm_sg_matrix)

# PCA on SG data
pr_sg <- prcomp(data_sg_norm[ , -1], rank. = 7, scale. = FALSE)
print(summary(pr_sg))

# Display SG PCA plot
print(plot_pca(pr_sg, data_sg_norm,
               round(100 * pr_sg$sdev[1]^2 / sum(pr_sg$sdev^2), 1),
               round(100 * pr_sg$sdev[2]^2 / sum(pr_sg$sdev^2), 1)))

# End of script
