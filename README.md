# Infrared Spectroscopy & Machine Learning Classification of Red-Stamp Inks

A forensic workflow to classify ten commercial red-stamp ink brands on Hansol Copy™ paper using ATR-FTIR spectroscopy and machine learning. Five classifiers are compared—PLS-DA, k-NN, SVM, RF, and a feed-forward neural network (FNN)—across multiple spectral ranges and preprocessing methods. Model robustness is validated via cross-validation, random-seed repeats, y-scrambling, and ROC/AUC curves. Finally, the workflow predicts the manufacturers of three unknown ink samples.

## Data  
All spectral data are provided as Excel files in the repository root:

- **Entire.xlsx**  
  Full raw spectra (4000–700 cm⁻¹; 1877 variables)  
- **Selected_1.xlsx**  
  Savitzky–Golay 2nd-derivative spectra (1700–900 cm⁻¹; 457 variables)  
- **Selected_2.xlsx**  
  VIP-selected sub-range spectra (1650–1100 cm⁻¹; 314 variables)  
- **Unknown.xlsx**  
  2nd-derivative spectra (1700–900 cm⁻¹) of three unlabelled ink samples  

## Analysis Scripts  
- `Principal Component Analysis (PCA).R`  
- `Partial Least-Squares Discriminant Analysis (PLS-DA).R`  
- `K-Nearest Neighbor (KNN).R`  
- `Support Vector Machine (SVM).R`  
- `Random Forest (RF).R`  
- `Feed-Forward Neural Networks (FNN).R`  

Each script loads its matching Excel sheet, applies preprocessing, trains and evaluates its model, and—in the case of `FNN.R`—produces `Unknown_predictions.xlsx`.

## File Structure  
Entire.xlsx
Selected_1.xlsx
Selected_2.xlsx
Unknown.xlsx
Principal Component Analysis (PCA).R
Partial Least-Squares Discriminant Analysis (PLS-DA).R
K-Nearest Neighbor (KNN).R
Support Vector Machine (SVM).R
Random Forest (RF).R
Feed-Forward Neural Networks (FNN).R
Unknown_predictions.xlsx
LICENSE
README.md


## Usage  
1. **Install required R packages**  
   ```r
   install.packages(c(
     "prospectr","caret","MLmetrics","openxlsx",
     "pROC","PRROC","reticulate","keras",
     "tibble","ggplot2"
   ))

2. **(Optional) Activate TensorFlow GPU environment**
    ```r
   library(reticulate)
   use_condaenv("tf_gpu", required = TRUE)

4. **Run scripts in sequence**
    ```r
   source("Principal Component Analysis (PCA).R")
   source("Partial Least-Squares Discriminant Analysis (PLS-DA).R")
   source("K-Nearest Neighbor (KNN).R")
   source("Support Vector Machine (SVM).R")
   source("Random Forest (RF).R")
   source("Feed-Forward Neural Networks (FNN).R")

6. **Inspect predictions**
   Open Unknown_predictions.xlsx to view the softmax probabilities for the unknown ink samples.

## Key Findings
- Best model: FNN on 2nd-derivative spectra (1700–900 cm⁻¹) achieved perfect Test F1 (= 1.000) and AUC (= 1.000)
- Interpretable alternatives: PLS-DA & RF also performed strongly (F1 ≥ 0.933)
- Critical sub-range: 1650–1100 cm⁻¹ identified by VIP analysis
- Unknown inks: High-confidence manufacturer probability predictions 
