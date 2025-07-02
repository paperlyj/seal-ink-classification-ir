# Infrared Spectroscopy & Machine Learning Classification of Red-Stamp Inks

## Overview
A forensic workflow to classify commercial red-stamp inks on Hansol Copy™ paper using ATR-FTIR spectroscopy and machine learning. We compare five classifiers—PLS-DA, k-NN, SVM, RF, and a feed-forward neural network (FNN)—across multiple spectral ranges and preprocessing methods, validate model robustness, and predict the origins of unknown samples.

## Data  
All spectra are stored in **Spectra.xlsx** with four worksheets:

- **Entire** (4000–700 cm⁻¹)  
  Raw full-range spectra (1877 variables).  
- **Selected1** (1700–900 cm⁻¹)  
  Savitzky–Golay 2nd-derivative spectra (457 variables).  
- **Selected2** (1650–1100 cm⁻¹)  
  VIP-selected sub-range (314 variables).  
- **Unknown** (1700–900 cm⁻¹)  
  Second-derivative spectra of three unlabelled ink samples for final prediction.

## Key Components

1. **Preprocessing**  
   - L2 normalization  
   - Savitzky–Golay 2nd-derivative + L2

2. **Modeling Pipeline**  
   - 3-fold cross-validation grid-search over learning rate (1e-4–1e-1), optimizer (`adam`/`sgd`), hidden units (16–512)  
   - Final FNN retrained on full training set (500 epochs)

3. **Validation & Metrics**  
   - Macro-F1 score  
   - 30 random-seed repeats → mean ± SD of Test F1  
   - Y-scrambling control  
   - One-vs-rest ROC & AUC curves

4. **Feature Selection**  
   - VIP analysis highlights 1650–1100 cm⁻¹ as most informative

5. **Unknown-Sample Prediction**  
   - Apply final FNN to **Unknown** sheet  
   - Export predicted softmax probabilities for each manufacturer

## File Structure

├── fnn_performance_comparison.R # Main R script
├── Spectra.xlsx # Spectral data sheets: Entire, Selected1, Selected2, Unknown
├── Unknown_predictions.xlsx # Output: predicted probabilities for unknown inks
└── README.md # This file

**Usage**

1. **Clone the repository**
git clone https://github.com/your-username/seal-ink-classification-ir.git
cd seal-ink-classification-ir
Install R packages

2. **Install R packages**
install.packages(c(
  "reticulate","keras","caret","MLmetrics","openxlsx",
  "prospectr","pROC","PRROC","tibble","ggplot2"
))

3. **Activate Conda environment**
library(reticulate)
use_condaenv("tf_gpu", required = TRUE)
Run the analysis

4. **Run the analysis**
source("fnn_performance_comparison.R")
Inspect predictions

5. **Inspect predictions**
View Unknown_predictions.xlsx for softmax probabilities.

**Key Findings**
- Best model: FNN on 2nd-derivative spectra in 1700–900 cm⁻¹ (Test F1 = 1.000; AUC = 1.000)

- Interpretable alternatives: PLS-DA & RF (F1 ≥ 0.933)

- Critical sub-range: 1650–1100 cm⁻¹ (VIP analysis)

- Unknown inks: High-confidence manufacturer predictions
